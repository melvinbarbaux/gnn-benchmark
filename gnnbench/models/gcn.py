import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize, renormalize_adj
from gnnbench.models.base_model import GNNModel
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor

def graph_convolution(inputs, sparse_renormalized_laplacian, weights, input_is_sparse=False):
    if input_is_sparse:
        output = tf.sparse.sparse_dense_matmul(inputs, weights)
    else:
        output = tf.matmul(inputs, weights)
    return tf.sparse.sparse_dense_matmul(sparse_renormalized_laplacian, output)

def graph_convolution_layer(model, output_dim, inputs, sparse_renormalized_laplacian,
                            activation_fn, dropout_prob, weight_decay, name, input_is_sparse=False):
    with tf.name_scope(name):
        input_dim = int(inputs.shape[1])
        weights = model.add_weight(name=f"{name}-weights", shape=[input_dim, output_dim],
                                    initializer=tf.keras.initializers.GlorotUniform(),
                                    regularizer=tf.keras.regularizers.l2(weight_decay))
        bias = model.add_weight(name=f"{name}-bias", shape=[output_dim],
                                initializer='zeros')
        if dropout_prob:
            inputs = dropout_supporting_sparse_tensors(inputs, 1.0 - dropout_prob)
        convolved = graph_convolution(inputs, sparse_renormalized_laplacian, weights, input_is_sparse)
        output = convolved + bias
        if activation_fn is not None:
            output = activation_fn(output)
        return output

class GCN(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider,
                 num_layers, hidden_size, dropout_prob, weight_decay, normalize_features):
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.weight_decay = weight_decay
        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            x = self.features
            for i in range(self.num_layers - 1):
                x = graph_convolution_layer(self,
                                            output_dim=self.hidden_size,
                                            inputs=x,
                                            sparse_renormalized_laplacian=self.graph_adj,
                                            activation_fn=tf.nn.relu,
                                            dropout_prob=self.dropout_prob,
                                            weight_decay=(self.weight_decay if i == 0 else 0.0),
                                            name=f"gc{i}",
                                            input_is_sparse=(i == 0))
            output = graph_convolution_layer(self,
                                             output_dim=self.targets.shape[1],
                                             inputs=x,
                                             sparse_renormalized_laplacian=self.graph_adj,
                                             activation_fn=None,
                                             dropout_prob=self.dropout_prob,
                                             weight_decay=0.0,
                                             name=f"gc{self.num_layers - 1}",
                                             input_is_sparse=False)
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        return to_sparse_tensor(renormalize_adj(graph_adj))

MODEL_INGREDIENT = Ingredient('model')

@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, dropout_prob):
    return GCN(node_features, graph_adj, labels, dataset_indices_placeholder,
               num_layers=num_layers, hidden_size=hidden_size,
               dropout_prob=dropout_prob,
               weight_decay=weight_decay,
               normalize_features=normalize_features)