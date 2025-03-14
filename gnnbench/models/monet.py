import numpy as np
import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, add_self_loops
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor, dropout_supporting_sparse_tensors, scatter_add_tensor

GAUSSIAN_WEIGHTS = "gaussian_weights"
FILTER_WEIGHTS = "filter_weights"

def weighting_function(model, transformed_coordinates, graph_adj, adj_with_self_loops_indices, num_kernels, kernel_no):
    with tf.name_scope(f"kernel{kernel_no}"):
        r = int(transformed_coordinates.shape[1])
        num_nodes = int(graph_adj.dense_shape[0])
        mean_init_values = get_mean_inits(num_kernels)
        init_value = [mean_init_values[kernel_no]] * r
        mu = model.add_weight(name=f"kernel{kernel_no}-mu", shape=[r],
                              initializer=tf.constant_initializer(init_value), trainable=True)
        if not hasattr(model, 'gaussian_weights'):
            model.gaussian_weights = []
        model.gaussian_weights.append(mu)
        sigma = model.add_weight(name=f"kernel{kernel_no}-sigma", shape=[r],
                                 initializer=tf.ones_initializer(), trainable=True)
        model.gaussian_weights.append(sigma)
        gaussian_weights = tf.exp(-0.5 * tf.square(transformed_coordinates - mu) / (1e-14 + tf.square(sigma)))
        gaussian_weights = tf.reduce_prod(gaussian_weights, axis=1)
        self_node_indices, _ = adj_with_self_loops_indices
        gaussian_weight_means = scatter_add_tensor(gaussian_weights, self_node_indices, out_shape=[num_nodes])
        gaussian_weight_means = tf.gather(gaussian_weight_means, self_node_indices)
        gaussian_weights = gaussian_weights / (1e-14 + gaussian_weight_means)
        gaussian_weight_matrix = tf.SparseTensor(
            indices=graph_adj.indices,
            values=gaussian_weights,
            dense_shape=graph_adj.dense_shape
        )
        return gaussian_weight_matrix

def gaussian_kernel(model, inputs, output_dim, transformed_coordinates, graph_adj, adj_with_self_loops_indices,
                    num_kernels, kernel_no, weight_decay, name):
    with tf.name_scope(name):
        input_dim = int(inputs.shape[1])
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
        gaussian_weights = weighting_function(model, transformed_coordinates, graph_adj, adj_with_self_loops_indices,
                                              num_kernels, kernel_no)
        output = tf.sparse.sparse_dense_matmul(gaussian_weights, transformed_features)
        return output

def gmm_layer(model, inputs, output_dim, transformed_coordinates, num_kernels, graph_adj,
              adj_with_self_loops_indices, activation_fn, dropout_prob, weight_decay, name):
    with tf.name_scope(name):
        if dropout_prob:
            inputs = dropout_supporting_sparse_tensors(inputs, 1.0 - dropout_prob)
        kernel_results = []
        for kernel_no in range(num_kernels):
            kernel_results.append(gaussian_kernel(model, inputs=inputs, output_dim=output_dim,
                                                  transformed_coordinates=transformed_coordinates,
                                                  graph_adj=graph_adj,
                                                  adj_with_self_loops_indices=adj_with_self_loops_indices,
                                                  num_kernels=num_kernels,
                                                  kernel_no=kernel_no,
                                                  weight_decay=weight_decay,
                                                  name=f"{name}-kernel{kernel_no}"))
        output = tf.add_n(kernel_results)
        return activation_fn(output) if activation_fn else output

class MoNet(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, num_kernels, r, dropout_prob, weight_decay,
                 normalize_features, alt_opt):
        self.num_nodes = targets.shape[0]
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_kernels = num_kernels
        self.r = r
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self.alt_opt = alt_opt
        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            coordinate_transform_weights = self.add_weight(
                name="coordinate_transform_weights",
                shape=[2, self.r],
                initializer=tf.keras.initializers.GlorotUniform(),
                trainable=True
            )
            if not hasattr(self, 'gaussian_weights'):
                self.gaussian_weights = []
            self.gaussian_weights.append(coordinate_transform_weights)
            coordinate_transform_bias = self.add_weight(
                name="coordinate_transform_bias",
                shape=[self.r],
                initializer='zeros',
                trainable=True
            )
            transformed_coordinates = tf.matmul(self.coordinates, coordinate_transform_weights)
            transformed_coordinates = tf.nn.bias_add(transformed_coordinates, coordinate_transform_bias)
            transformed_coordinates = tf.nn.tanh(transformed_coordinates)
            x = self.features
            for i in range(0, self.num_layers - 1):
                x = gmm_layer(self, inputs=x,
                              output_dim=self.hidden_size,
                              transformed_coordinates=transformed_coordinates,
                              num_kernels=self.num_kernels,
                              graph_adj=self.graph_adj,
                              adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                              activation_fn=tf.nn.relu,
                              dropout_prob=self.dropout_prob,
                              weight_decay=self.weight_decay if i == 0 else 0.0,
                              name=f"gmm_layer{i}")
            output = gmm_layer(self, inputs=x,
                               output_dim=self.targets.shape[1],
                               transformed_coordinates=transformed_coordinates,
                               num_kernels=self.num_kernels,
                               graph_adj=self.graph_adj,
                               adj_with_self_loops_indices=self.adj_with_self_loops_indices,
                               activation_fn=None,
                               dropout_prob=self.dropout_prob,
                               weight_decay=0.0,
                               name=f"gmm_layer{self.num_layers - 1}")
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return tf.convert_to_tensor(features.todense(), dtype=tf.float32)

    def _preprocess_adj(self, graph_adj):
        adj_with_self_loops = add_self_loops(graph_adj)
        self.adj_dense_shape = adj_with_self_loops.shape
        adj_with_self_loops_tensor = to_sparse_tensor(adj_with_self_loops)
        adj_with_self_loops_coo = adj_with_self_loops.tocoo()
        self.adj_with_self_loops_indices = (adj_with_self_loops_coo.row, adj_with_self_loops_coo.col)
        self.coordinates = self._generate_coordinates(adj_with_self_loops, self.adj_with_self_loops_indices)
        return adj_with_self_loops_tensor

    @staticmethod
    def _generate_coordinates(adj_with_self_loops, adj_with_self_loops_indices):
        degrees = adj_with_self_loops.sum(axis=1).astype(np.float32)
        inv_degrees = 1.0 / np.sqrt(degrees)
        start_nodes, end_nodes = adj_with_self_loops_indices
        start_node_degrees = inv_degrees[list(start_nodes)]
        end_node_degrees = inv_degrees[list(end_nodes)]
        return np.hstack([start_node_degrees, end_node_degrees])

    def optimize(self, learning_rate, global_step):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if self.alt_opt:
            @tf.function
            def train_step():
                with tf.GradientTape() as tape:
                    loss_value = self._loss()
                grads = tape.gradient(loss_value, self.gaussian_weights)
                optimizer.apply_gradients(zip(grads, self.gaussian_weights))
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

def get_mean_inits(num_kernels):
    return np.arange(-1.0 + (1.0 / (2 * num_kernels)), 1.0, 2.0 / num_kernels)

MODEL_INGREDIENT = Ingredient('model')

@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, num_kernels, r, dropout_prob, alt_opt):
    return MoNet(node_features, graph_adj, labels, dataset_indices_placeholder,
                 num_layers=num_layers, hidden_size=hidden_size,
                 num_kernels=num_kernels,
                 r=r,
                 dropout_prob=dropout_prob,
                 weight_decay=weight_decay,
                 normalize_features=normalize_features,
                 alt_opt=alt_opt)