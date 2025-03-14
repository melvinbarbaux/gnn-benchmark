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
        self.extract_variables_state_fn, self.set_trainable_variables_fn = get_reset_variable_ops()
        self.best_step = 0
        self.best_variable_state = None

    def should_stop(self, epoch, val_loss, val_accuracy):
        raise NotImplementedError

    def after_stopping_ops(self):
        self._log.debug(f"Resetting to best state of variables which occurred at step {self.best_step + 1}.")
        self.set_trainable_variables_fn(self.best_variable_state)

    def reset(self):
        self.best_step = 0
        self.best_variable_state = self.extract_variables_state_fn()


class GATCriterion(CriterionWithVariablesReset):
    def __init__(self, patience, _log):
        super().__init__(patience, _log)
        self.val_accuracy_max = 0.0
        self.val_loss_min = np.inf
        self.patience_step = 0

    def should_stop(self, epoch, val_loss, val_accuracy):
        if val_accuracy >= self.val_accuracy_max or val_loss <= self.val_loss_min:
            self.val_accuracy_max = np.maximum(val_accuracy, self.val_accuracy_max)
            self.val_loss_min = np.minimum(val_loss, self.val_loss_min)
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = self.extract_variables_state_fn()
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
        if val_loss <= self.val_loss_min:
            self.val_loss_min = np.minimum(val_loss, self.val_loss_min)
            self.patience_step = 0
            self.best_step = epoch
            self.best_variable_state = self.extract_variables_state_fn()
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
            self.val_accuracy_max = np.maximum(val_accuracy, self.val_accuracy_max)
            self.val_loss_min = np.minimum(val_loss, self.val_loss_min)
            self.best_step = epoch
            self.best_variable_state = self.extract_variables_state_fn()
            if val_accuracy >= self.val_accuracy_max + self.tolerance or val_loss <= self.val_loss_min - self.tolerance:
                self.patience_step = 0
            else:
                self.patience_step += 1
        else:
            self.patience_step += 1
        return self.patience_step >= self.patience


def get_reset_variable_ops():
    return extract_variables_state, set_trainable_variables


def extract_variables_state():
    return [var.numpy() for var in tf.trainable_variables()]


def set_trainable_variables(variables_state):
    for i, var in enumerate(tf.trainable_variables()):
        var.assign(variables_state[i])