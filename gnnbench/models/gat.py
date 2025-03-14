"""Code in this file is inspired by Velickovic et al. - Graph Attention Networks
and Master's Thesis of Johannes Klicpera (TUM, KDD)."""

import numpy as np
import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor

ATTENTION_WEIGHTS = "attention_weights"
FILTER_WEIGHTS = "filter_weights"

def attention_mechanism(model, features, graph_adj, adj_with_self_loops_indices,
                        coefficient_dropout_prob, weight_decay, name):
    input_dim = int(features.shape[1])
    a_i = model.add_weight(name=f"{name}-att_i", shape=[input_dim, 1],
                           initializer=tf.keras.initializers.GlorotUniform(),
                           regularizer=tf.keras.regularizers.l2(weight_decay))
    a_j = model.add_weight(name=f"{name}-att_j", shape=[input_dim, 1],
                           initializer=tf.keras.initializers.GlorotUniform(),
                           regularizer=tf.keras.regularizers.l2(weight_decay))
    if not hasattr(model, 'attention_weights'):
        model.attention_weights = []
    model.attention_weights.extend([a_i, a_j])
    att_i = tf.matmul(features, a_i)
    bias_i = model.add_weight(name=f"{name}-bias_i", shape=[1], initializer='zeros')
    att_i = tf.nn.bias_add(att_i, bias_i)
    att_j = tf.matmul(features, a_j)
    bias_j = model.add_weight(name=f"{name}-bias_j", shape=[1], initializer='zeros')
    att_j = tf.nn.bias_add(att_j, bias_j)
    att_i_gather = tf.gather(att_i, adj_with_self_loops_indices[0], axis=0)
    att_j_gather = tf.gather(att_j, adj_with_self_loops_indices[1], axis=0)
    attention_weights_of_edges = att_i_gather + att_j_gather
    attention_weights_of_edges = tf.squeeze(attention_weights_of_edges, axis=-1)
    attention_weight_matrix = tf.SparseTensor(
        indices=graph_adj.indices,
        values=tf.nn.leaky_relu(attention_weights_of_edges, alpha=0.2),
        dense_shape=graph_adj.dense_shape
    )
    attention_coefficients = tf.sparse.softmax(attention_weight_matrix)
    if coefficient_dropout_prob:
        attention_coefficients = dropout_supporting_sparse_tensors(
            attention_coefficients, 1.0 - coefficient_dropout_prob)
    return attention_coefficients

def attention_head(model, inputs, output_dim, graph_adj, adj_with_self_loops_indices,
                   activation_fn, input_dropout_prob, coefficient_dropout_prob,
                   weight_decay, name):
    with tf.name_scope(name):
        input_dim = int(inputs.shape[1])
        if input_dropout_prob:
            inputs = dropout_supporting_sparse_tensors(inputs, 1.0 - input_dropout_prob)
        linear_transform_weights = model.add_weight(
            name=f"{name}-linear_transform_weights",
            shape=[input_dim, output_dim],
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        if not hasattr(model, 'filter_weights'):
            model.filter_weights = []
        model.filter_weights.append(linear_transform_weights)
        if isinstance(inputs, tf.SparseTensor):
            transformed_features = tf.sparse.sparse_dense_matmul(inputs, linear_transform_weights)
        else:
            transformed_features = tf.matmul(inputs, linear_transform_weights)
        attention_coefficients = attention_mechanism(model, transformed_features, graph_adj,
                                                     adj_with_self_loops_indices,
                                                     coefficient_dropout_prob,
                                                     weight_decay, name)
        if input_dropout_prob:
            transformed_features = dropout_supporting_sparse_tensors(
                transformed_features, 1.0 - input_dropout_prob)
        output = tf.sparse.sparse_dense_matmul(attention_coefficients, transformed_features)
        bias = model.add_weight(name=f"{name}-bias", shape=[output_dim], initializer='zeros')
        output = tf.nn.bias_add(output, bias)
        if activation_fn is not None:
            output = activation_fn(output)
        return output

def attention_layer(model, inputs, output_dim, num_heads, graph_adj,
                    adj_with_self_loops_indices, activation_fn, use_averaging,
                    input_dropout_prob, coefficient_dropout_prob, weight_decay, name):
    with tf.name_scope(name):
        head_results = []
        for i in range(num_heads):
            head = attention_head(model, inputs, output_dim, graph_adj,
                                  adj_with_self_loops_indices, activation_fn,
                                  input_dropout_prob, coefficient_dropout_prob,
                                  weight_decay, name=f"{name}-head{i}")
            head_results.append(head)
        if use_averaging:
            return tf.add_n(head_results) / num_heads
        else:
            return tf.concat(head_results, axis=1)

class GAT(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, num_heads, input_dropout_prob,
                 coefficient_dropout_prob, weight_decay, normalize_features, alt_opt):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_dropout_prob = input_dropout_prob
        self.coefficient_dropout_prob = coefficient_dropout_prob
        self.weight_decay = weight_decay
        self.alt_opt = alt_opt
        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = attention_layer(self, x, output_dim=self.hidden_size,
                                    num_heads=self.num_heads[i],
                                    graph_adj=self.graph_adj,
                                    adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                    activation_fn=tf.nn.elu,
                                    use_averaging=False,
                                    input_dropout_prob=self.input_dropout_prob,
                                    coefficient_dropout_prob=self.coefficient_dropout_prob,
                                    weight_decay=self.weight_decay,
                                    name=f"attention_layer{i}")
            output = attention_layer(self, x, output_dim=self.targets.shape[1],
                                     num_heads=self.num_heads[self.num_layers - 1],
                                     graph_adj=self.graph_adj,
                                     adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                                     activation_fn=None,
                                     use_averaging=True,
                                     input_dropout_prob=self.input_dropout_prob,
                                     coefficient_dropout_prob=self.coefficient_dropout_prob,
                                     weight_decay=self.weight_decay,
                                     name=f"attention_layer{self.num_layers - 1}")
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
        self.adj_with_self_loops_indices = np.array([adj_with_self_loops_coo.row, adj_with_self_loops_coo.col])
        return adj_with_self_loops_tensor

    def optimize(self, learning_rate, global_step):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.alt_opt:
            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    loss_value = self._loss()
                grads = tape.gradient(loss_value, self.attention_weights)
                optimizer.apply_gradients(zip(grads, self.attention_weights))
                with tf.GradientTape() as tape2:
                    loss_value2 = self._loss()
                grads2 = tape2.gradient(loss_value2, self.filter_weights)
                optimizer.apply_gradients(zip(grads2, self.filter_weights))
                global_step.assign_add(1)
                return loss_value
            return train_step
        else:
            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    loss_value = self._loss()
                grads = tape.gradient(loss_value, self.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.trainable_variables))
                global_step.assign_add(1)
                return loss_value
            return train_step

MODEL_INGREDIENT = Ingredient('model')

@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features, num_layers, hidden_size, num_heads,
                input_dropout_prob, coefficient_dropout_prob, alt_opt):
    return GAT(node_features, graph_adj, labels, dataset_indices_placeholder,
               num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads,
               input_dropout_prob=input_dropout_prob, coefficient_dropout_prob=coefficient_dropout_prob,
               weight_decay=weight_decay, normalize_features=normalize_features, alt_opt=alt_opt)