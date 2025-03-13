import os
import json
import logging
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver

import gnnbench.models
from gnnbench.data.make_dataset import get_dataset_and_split_planetoid, get_dataset, get_train_val_test_split, \
    get_split_feed_dicts
from gnnbench.train import build_train_ops, run_train_ops


def select_model(model_name):
    if "GraphSAGE" in model_name:
        model_name = "GraphSAGE"
    if "LabelProp" in model_name:
        model_name = "LabelProp"
    module = getattr(gnnbench.models, model_name.lower())
    return module.MODEL_INGREDIENT, module.build_model


def run_single_split(experiment_name, model_name, dataset, num_training_runs, dataset_source, data_path, metrics,
                     split_no, seed, train_config, model_config, db_host, db_port, target_db_name,
                     gpu_id=None, log_verbose=True):

    # load the builder methods for the selected model
    model_ingredient, build_model = select_model(model_name)

    # create the experiment...
    ex = get_experiment(experiment_name,
                        db_host=db_host,
                        db_port=db_port,
                        db_name=target_db_name,
                        ingredients=[model_ingredient],
                        log_verbose=log_verbose)

    # ...and add the configs to it
    ex.add_config(train_config)
    model_ingredient.add_config(model_config)
    # also add model configuration to experiment since it may contain updated settings such as learning rate
    ex.add_config(model_config)

    # some updates to the experiment config
    ex.add_config({'dataset': dataset,
                   'num_training_runs': num_training_runs,
                   'dataset_source': dataset_source,
                   'data_path': data_path,
                   'metrics': metrics,
                   'seed': seed,
                   'split_no': split_no,
                   'device_id': gpu_id
                   })

    @ex.capture
    def build_dataset(dataset, dataset_source, data_path, standardize_graph, split,
                      _log):
        if dataset_source == 'planetoid' or dataset_source == 'planetoid_random':
            return get_dataset_and_split_planetoid(dataset, data_path, _log)[:3]
        else:
            # dims:
            # graph_adj: num_nodes x num_nodes
            # node_features: num_nodes x num_features
            # labels: num_nodes x num_classes
            return get_dataset(dataset, data_path, standardize_graph, _log, split['train_examples_per_class'],
                               split['val_examples_per_class'])

    @ex.capture
    def build_split(random_state, labels, dataset, dataset_source, data_path, split, _log):
        if dataset_source == 'planetoid':
            return get_dataset_and_split_planetoid(dataset, data_path, _log)[3:]
        else:
            return get_train_val_test_split(random_state, labels, **split)

    @ex.capture
    def _build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                     train_feed, trainval_feed, val_feed, test_feed,
                     weight_decay, normalize_features):
        return build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                           train_feed, trainval_feed, val_feed, test_feed,
                           weight_decay, normalize_features)

    @ex.main
    def run_experiment_single_split(split_no,
                                    learning_rate, num_epochs, early_stopping_tolerance, early_stopping_criterion,
                                    improvement_tolerance, metrics,
                                    alternating_optimization_interval, lr_decay_factor, lr_decay_steps, report_interval,
                                    num_training_runs, _run, _log, _seed, device_id=None):
        dataset_slug = build_dataset()
        random_state = np.random.RandomState(_seed)
        split_indices = build_split(random_state, dataset_slug[-1])

        tf_seeds = random_state.randint(0, 1000000, num_training_runs)

        traces = defaultdict(list)
        test_metrics_for_all_inits = []
        _log.info(f"Starting {num_training_runs} training runs for split {split_no}.\n\n")

        # GPU configuration for TF2: set visible devices if device_id is provided
        if device_id is not None:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Assuming device_id is an index (as string) of the GPU
                    tf.config.set_visible_devices(gpus[int(device_id)], 'GPU')
            except Exception as e:
                _log.error(e)
        
        # In TF2, eager execution is enabled by default, so we don't need to create a graph or session
        split_slug = get_split_feed_dicts(*split_indices)
        model = _build_model(*dataset_slug, *split_slug)

        # Build training operations using TF2 style; assume build_train_ops returns a train_step callable,
        # early_stopping, global_step, an initialization function (init_op) and a learning rate function
        train_step, early_stopping, global_step, init_op, learning_rate = build_train_ops(
            model,
            early_stopping_tolerance, early_stopping_criterion, improvement_tolerance, learning_rate, _log
        )

        # Initialize variables if necessary (init_op should be a callable that performs initialization)
        init_op()

        for run_no in range(num_training_runs):
            _log.info(f"Starting run {run_no} for split {split_no}...\n---")
            tf.random.set_seed(tf_seeds[run_no])
            
            final_test_metrics = run_train_ops(
                train_step, early_stopping, global_step, init_op,
                *split_slug[-4:], metrics,
                model, learning_rate, num_epochs, early_stopping_criterion,
                alternating_optimization_interval, lr_decay_factor,
                lr_decay_steps,
                report_interval, _run, run_no, _log, traces
            )
            test_metrics_for_all_inits.append(final_test_metrics)
        
        _log.info("---\n")

        test_metrics_collected = defaultdict(list)
        for test_metrics in test_metrics_for_all_inits:
            for name, value in test_metrics.items():
                test_metrics_collected[name].append(value)
        for name, values in test_metrics_collected.items():
            _log.debug(f"Mean test set {name} over {num_training_runs} runs for split {split_no}: "
                       f"{float(np.mean(values)):.4f}, stddev: {float(np.std(values)):.4f}")
        return traces

    print(f"Running {experiment_name}...\n---")
    ex.run()
    print(f"Finished {experiment_name}.\n---")


def get_experiment(name, db_host, db_port, db_name, ingredients=None, log_verbose=True):

    if ingredients is None:
        ex = Experiment(name)
    else:
        ex = Experiment(name, ingredients=ingredients)

    ex.observers.append(MongoObserver.create(
        url=f"mongodb://{db_host}:{db_port}",
        db_name=db_name)
    )
    ex.logger = _get_logger(log_verbose)
    return ex


def _get_logger(verbose):
    logging.basicConfig(filename=None, level=logging.DEBUG if verbose else logging.WARN,
                        format='%(asctime)s::%(name)s::%(levelname)s - %(message)s')
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    return logging.getLogger()


# only used internally by spawn_worker.py
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # for being called from the multi-gpu script to run a single split on a single GPU
        _gpu_id = sys.argv[1]
        _log_verbose = int(sys.argv[2]) == 1
        _config = json.loads(sys.argv[3])
        _config['log_verbose'] = _log_verbose
        _config['gpu_id'] = _gpu_id

        run_single_split(**_config)
    else:
        exit(1)
