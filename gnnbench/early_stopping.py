import numpy as np
import tensorflow as tf


class EarlyStoppingCriterion(object):
    def __init__(self, patience, _log):
        self.patience = patience
        self._log = _log

    def should_stop(self, epoch, val_loss, val_accuracy):
        raise NotImplementedError

    def after_stopping_ops(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class NoStoppingCriterion(EarlyStoppingCriterion):

    def should_stop(self, epoch, val_loss, val_accuracy):
        return False

    def after_stopping_ops(self):
        pass

    def reset(self):
        pass


class GCNCriterion(EarlyStoppingCriterion):
    def __init__(self, patience, _log):
        super().__init__(patience, _log)
        self.val_losses = []

    def should_stop(self, epoch, val_loss, val_accuracy):
        self.val_losses.append(val_loss)

        return epoch >= self.patience and self.val_losses[-1] > np.mean(
            self.val_losses[-(self.patience + 1):-1])

    def after_stopping_ops(self):
        pass

    def reset(self):
        self.val_losses = []


class CriterionWithVariablesReset(EarlyStoppingCriterion):
    def __init__(self, patience, _log):
        super().__init__(patience, _log)
        self.best_step = 0
        self.best_variable_state = None

    def after_stopping_ops(self):
        self._log.debug(f"Resetting to best state of variables which occurred at step {self.best_step + 1}.")
        set_trainable_variables(self.best_variable_state)

    def reset(self):
        self.best_step = 0
        self.best_variable_state = extract_variables_state()


class GATCriterion(CriterionWithVariablesReset):
    def __init__(self, patience, _log):
        super().__init__(patience, _log)
        self.val_accuracy_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_accuracy >= self.val_accuracy_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved
            self.val_accuracy_max = np.max((val_accuracy, self.val_accuracy_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = extract_variables_state()
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_accuracy_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0


class KDDCriterion(CriterionWithVariablesReset):
    def __init__(self, patience, _log):
        super().__init__(patience, _log)
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        # only pay attention to validation loss
        if val_loss <= self.val_loss_min:
            # val loss improved
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = extract_variables_state()
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience

    def reset(self):
        super().reset()
        self.val_loss_min = np.inf
        self.patience_step = 0


class GATCriterionWithTolerance(GATCriterion):
    def __init__(self, patience, tolerance, _log):
        super().__init__(patience, _log)
        self.tolerance = tolerance

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_accuracy >= self.val_accuracy_max or val_loss <= self.val_loss_min:
            # either val accuracy or val loss improved, so we have a new best state
            self.val_accuracy_max = np.max((val_accuracy, self.val_accuracy_max))
            self.val_loss_min = np.min((val_loss, self.val_loss_min))
            self.best_step = epoch
            self.best_variable_state = extract_variables_state()

            # But only reset patience if accuracy or loss improved by a certain degree. This avoids long-running
            # convergence processes like for the LabelPropagation algorithm.
            if val_accuracy >= self.val_accuracy_max + self.tolerance or val_loss <= self.val_loss_min - self.tolerance:
                self.patience_step = 0
            else:
                self.patience_step += 1
        else:
            self.patience_step += 1

        return self.patience_step >= self.patience


def extract_variables_state():
    """Return a list of current trainable variable values as numpy arrays (TF2 eager mode)."""
    return [var.numpy() for var in tf.compat.v1.trainable_variables()]


def set_trainable_variables(variables_state):
    """Assign the provided state (list of numpy arrays) to the current trainable variables (TF2 eager mode)."""
    for var, value in zip(tf.compat.v1.trainable_variables(), variables_state):
        var.assign(value)
