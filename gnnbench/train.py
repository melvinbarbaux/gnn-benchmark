import numpy as np
import tensorflow as tf

import gnnbench.metrics
from gnnbench.early_stopping import GCNCriterion, GATCriterion, KDDCriterion, NoStoppingCriterion, \
    GATCriterionWithTolerance


def build_train_ops(model, early_stopping_tolerance, early_stopping_criterion, improvement_tolerance, learning_rate,_log):
    # In TF2, we no longer use placeholders or sessions. Create global_step as a tf.Variable.
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

    # Use the provided learning_rate instead of None.
    train_step = model.optimize(learning_rate, global_step)  

    # Instantiate early stopping criterion without session dependency
    if early_stopping_criterion == 'gcn':
        early_stopping = GCNCriterion(early_stopping_tolerance, _log)
    elif early_stopping_criterion == 'gat':
        early_stopping = GATCriterion(early_stopping_tolerance, _log)
    elif early_stopping_criterion == 'gnnbench':
        early_stopping = KDDCriterion(early_stopping_tolerance, _log)
    elif early_stopping_criterion == "gat_tol":
        early_stopping = GATCriterionWithTolerance(early_stopping_tolerance, improvement_tolerance, _log)
    else:
        _log.warn("Disabled early stopping.")
        early_stopping = NoStoppingCriterion(early_stopping_tolerance, _log)

    # In eager execution, no initialization op is needed.
    init_op = lambda: None
    learning_rate_fn = lambda: None
    return train_step, early_stopping, global_step, init_op, learning_rate_fn


def run_train_ops(train_step, early_stopping, global_step, init_op,
                  train_feed, trainval_feed, val_feed, test_feed, metrics_to_use,
                  model, learning_rate, num_epochs, early_stopping_criterion,
                  alternating_optimization_interval, lr_decay_factor, lr_decay_steps,
                  report_interval, _run, run_no, _log, traces):
    # Reset training progress
    init_op()
    early_stopping.reset()
    start_iteration = int(global_step.numpy())
    _log.info("Started training for %s epochs." % num_epochs)

    for epoch in range(start_iteration, num_epochs):
        # Here, we assume that train_feed, trainval_feed, val_feed, and test_feed are already prepared tensors
        # and that model.optimize uses the learning_rate internally.
        if isinstance(train_step, list):
            loss_to_optimize = int(epoch / alternating_optimization_interval) % len(train_step)
            loss_value = train_step[loss_to_optimize]()
        else:
            loss_value = train_step()
        train_loss = loss_value.numpy() if hasattr(loss_value, 'numpy') else loss_value

        # Compute validation loss and metrics directly
        val_loss = model.loss.numpy() if hasattr(model.loss, 'numpy') else model.loss
        val_metrics = compute_metrics(model, val_feed, metrics_to_use)

        if (epoch + 1) % report_interval == 0:
            train_metrics = compute_metrics(model, trainval_feed, metrics_to_use)
            metrics_string = build_metrics_string(train_loss, val_loss, train_metrics, val_metrics)
            _log.debug(f"After {int(global_step.numpy())} epochs:\n" + metrics_string)
            _run.log_scalar(f"train.loss-{run_no}", train_loss, epoch + 1)
            _run.log_scalar(f"val.loss-{run_no}", val_loss, epoch + 1)
            for name, value in train_metrics.items():
                _run.log_scalar(f"train.{name}-{run_no}", value, epoch + 1)
            for name, value in val_metrics.items():
                _run.log_scalar(f"val.{name}-{run_no}", value, epoch + 1)

        if early_stopping.should_stop(epoch, val_loss, val_metrics['accuracy']):
            _log.debug(f"Early stopping by {early_stopping_criterion} criterion after {int(global_step.numpy())} epochs.")
            break

        if lr_decay_factor > 0.0 and lr_decay_steps:
            if lr_decay_factor >= 1.0:
                raise ValueError(f"A learning rate decay factor of {lr_decay_factor} will not decay the learning rate.")
            if epoch + 1 in lr_decay_steps:
                learning_rate *= lr_decay_factor
                _log.debug(f"Decaying learning rate to {learning_rate}.")

        global_step.assign_add(1)

    early_stopping.after_stopping_ops()

    final_train_loss = model.loss.numpy() if hasattr(model.loss, 'numpy') else model.loss
    final_val_loss = model.loss.numpy() if hasattr(model.loss, 'numpy') else model.loss
    final_train_metrics = compute_metrics(model, trainval_feed, metrics_to_use)
    final_val_metrics = compute_metrics(model, val_feed, metrics_to_use)
    final_metrics_string = build_metrics_string(final_train_loss, final_val_loss,
                                               final_train_metrics, final_val_metrics)
    _log.debug(f"---\nTraining finished after {int(global_step.numpy())} epochs. Final values:\n" + final_metrics_string)

    _log.info("Evaluating on test set.")
    final_test_metrics = compute_metrics(model, test_feed, metrics_to_use)
    _log.debug("\n".join(f"Test {name} = {value:.4f}" for name, value in final_test_metrics.items()))

    traces['train.loss'].append(final_train_loss)
    traces['val.loss'].append(final_val_loss)
    for name, value in final_train_metrics.items():
        traces[f'train.{name}'].append(value)
    for name, value in final_val_metrics.items():
        traces[f'val.{name}'].append(value)
    for name, value in final_test_metrics.items():
        traces[f'test.{name}'].append(value)

    return final_test_metrics


def compute_metrics(model, feed, metrics_to_use):
    # Compute predictions and ground truth directly; assume feed is compatible with model.predict
    predictions = model.predict(feed)
    ground_truth = model.targets
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    if hasattr(ground_truth, 'numpy'):
        ground_truth = ground_truth.numpy()
    predictions = np.argmax(predictions, axis=1)
    ground_truth = np.argmax(ground_truth, axis=1)
    return {name: getattr(gnnbench.metrics, name)(ground_truth, predictions) for name in metrics_to_use}


def build_metrics_string(train_loss, val_loss, train_metrics, val_metrics):
    train_part = f"    - " + "; ".join(
        [f"train loss = {train_loss:.4f}"] + [f"train {name} = {value:.4f}" for name, value in train_metrics.items()]
    )
    val_part = f"    - " + "; ".join(
        [f"val loss = {val_loss:.4f}"] + [f"val {name} = {value:.4f}" for name, value in val_metrics.items()]
    )
    return train_part + "\n" + val_part + "\n"
