import numpy as np
import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor, scatter_add_tensor, dropout_supporting_sparse_tensors


def aggregate_mean(transformed_features, graph_adj, degrees, name):
    with tf.name_scope(name):
        output = tf.sparse.sparse_dense_matmul(graph_adj, transformed_features)
        return output / degrees


def aggregate_maxpool(model, features, agg_transform_size, adj_with_self_loops_indices, num_features, name):
    with tf.name_scope(name):
        fc_weights = model.add_weight(name=f"{name}-fc_weights",
                                      shape=[num_features, agg_transform_size],
                                      initializer=tf.keras.initializers.GlorotUniform())
        if isinstance(features, tf.SparseTensor):
            transformed_features = tf.sparse.sparse_dense_matmul(features, fc_weights)
        else:
            transformed_features = tf.matmul(features, fc_weights)
        transformed_features = tf.nn.relu(transformed_features)
        neighbours_features = tf.gather(transformed_features, adj_with_self_loops_indices)
        output = tf.reduce_max(neighbours_features, axis=1)
        return output


def aggregate_meanpool(model, features, agg_transform_size, adj_with_self_loops_indices, degrees, num_nodes, num_features, name):
    with tf.name_scope(name):
        self_indices, neighbours_indices = adj_with_self_loops_indices
        fc_weights = model.add_weight(name=f"{name}-fc_weights",
                                      shape=[num_features, agg_transform_size],
                                      initializer=tf.keras.initializers.GlorotUniform())
        if isinstance(features, tf.SparseTensor):
            transformed_features = tf.sparse.sparse_dense_matmul(features, fc_weights)
        else:
            transformed_features = tf.matmul(features, fc_weights)
        transformed_features = tf.nn.relu(transformed_features)
        edge_features = tf.gather(transformed_features, neighbours_indices)
        output = scatter_add_tensor(edge_features, self_indices, out_shape=[num_nodes, agg_transform_size])
        output = output / degrees
        return output


def sage_layer(model, features, output_dim, graph_adj, adj_with_self_loops_indices, degrees,
               aggregator, agg_transform_size, activation_fn,
               weight_decay, dropout_prob, name):
    with tf.name_scope(name):
        num_nodes = int(features.shape[0])
        num_features = int(features.shape[1])
        features = tf.cond(
            tf.cast(dropout_prob, tf.bool),
            lambda: dropout_supporting_sparse_tensors(features, 1 - dropout_prob),
            lambda: features
        )
        if aggregator == 'mean' or aggregator == 'gcn':
            agg_weights = model.add_weight(name=f"{name}-agg-weights",
                                          shape=[num_features, output_dim],
                                          initializer=tf.keras.initializers.GlorotUniform(),
                                          regularizer=tf.keras.regularizers.l2(weight_decay))
            if isinstance(features, tf.SparseTensor):
                transformed_features = tf.sparse.sparse_dense_matmul(features, agg_weights)
            else:
                transformed_features = tf.matmul(features, agg_weights)
            agg_features = aggregate_mean(transformed_features, graph_adj, degrees, f"{name}-aggregator")
        elif aggregator == 'meanpool' or aggregator == 'maxpool':
            if aggregator == 'meanpool':
                aggregated = aggregate_meanpool(model, features, agg_transform_size, adj_with_self_loops_indices,
                                                degrees, num_nodes, num_features, f"{name}-aggregator")
            elif aggregator == 'maxpool':
                aggregated = aggregate_maxpool(model, features, agg_transform_size, adj_with_self_loops_indices,
                                               num_features, f"{name}-aggregator")
            agg_weights = model.add_weight(name=f"{name}-agg-weights",
                                          shape=[int(aggregated.shape[1]), output_dim],
                                          initializer=tf.keras.initializers.GlorotUniform(),
                                          regularizer=tf.keras.regularizers.l2(weight_decay))
            agg_features = tf.matmul(aggregated, agg_weights)
        else:
            raise ValueError('Undefined aggregator.')
        if aggregator == 'gcn':
            output = agg_features
        else:
            skip_conn_weights = model.add_weight(name=f"{name}-skip_conn-weights",
                                                shape=[num_features, output_dim],
                                                initializer=tf.keras.initializers.GlorotUniform(),
                                                regularizer=tf.keras.regularizers.l2(weight_decay))
            if isinstance(features, tf.SparseTensor):
                skip_features = tf.sparse.sparse_dense_matmul(features, skip_conn_weights)
            else:
                skip_features = tf.matmul(features, skip_conn_weights)
            output = agg_features + skip_features
        bias = model.add_weight(name=f"{name}-bias", shape=[output_dim], initializer='zeros')
        output = tf.nn.bias_add(output, bias)
        if aggregator != 'gcn':
            normalizer = tf.norm(output, ord=2, axis=1, keepdims=True)
            normalizer = tf.clip_by_value(normalizer, clip_value_min=1.0, clip_value_max=np.inf)
            output = output / normalizer
        if activation_fn is not None:
            output = activation_fn(output)
        return output


class GraphSAGE(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, aggregator, agg_transform_size, dropout_prob, weight_decay,
                 normalize_features):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        self.aggregator = aggregator
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.agg_transform_size = agg_transform_size
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = sage_layer(self,
                               features=x,
                               output_dim=self.hidden_size,
                               graph_adj=self.graph_adj,
                               adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                               degrees=self.degrees,
                               aggregator=self.aggregator,
                               agg_transform_size=self.agg_transform_size,
                               activation_fn=tf.nn.relu,
                               weight_decay=self.weight_decay if i == 0 else 0.0,
                               dropout_prob=self.dropout_prob,
                               name=f"sage_layer{i}")
            output = sage_layer(self,
                                features=x,
                                output_dim=self.targets.shape[1],
                                graph_adj=self.graph_adj,
                                adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                degrees=self.degrees,
                                aggregator=self.aggregator,
                                agg_transform_size=self.agg_transform_size,
                                activation_fn=None,
                                weight_decay=0.0,
                                dropout_prob=self.dropout_prob,
                                name=f"sage_layer{self.num_layers - 1}")
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        adj_with_self_loops = add_self_loops(graph_adj)
        self.adj_dense_shape = adj_with_self_loops.shape
        adj_with_self_loops_tensor = to_sparse_tensor(adj_with_self_loops)
        adj_with_self_loops_coo = adj_with_self_loops.tocoo()
        self.degrees = np.array(adj_with_self_loops.sum(axis=1)).flatten().astype(np.float32)
        if self.aggregator == 'maxpool':
            neighbours_matrix = np.zeros((self.num_nodes, int(np.max(self.degrees))), dtype=np.int32)
            insert_index = 0
            self_node_old = 0
            for i, self_node in enumerate(adj_with_self_loops_coo.row):
                if self_node != self_node_old:
                    insert_index = 0
                neighbours_matrix[self_node, insert_index] = adj_with_self_loops_coo.col[i]
                insert_index += 1
                self_node_old = self_node
            self.adj_with_self_loops_indices = neighbours_matrix
        else:
            self.adj_with_self_loops_indices = np.array([adj_with_self_loops_coo.row, adj_with_self_loops_coo.col])
        return adj_with_self_loops_tensor


MODEL_INGREDIENT = Ingredient('model')

@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, aggregator, agg_transform_size, dropout_prob):
    return GraphSAGE(node_features, graph_adj, labels, dataset_indices_placeholder,
                     num_layers=num_layers, hidden_size=hidden_size,
                     dropout_prob=dropout_prob,
                     weight_decay=weight_decay,
                     aggregator=aggregator,
                     agg_transform_size=agg_transform_size,
                     normalize_features=normalize_features)