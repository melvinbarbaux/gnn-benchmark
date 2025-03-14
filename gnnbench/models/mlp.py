import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize
from gnnbench.models.base_model import GNNModel
from gnnbench.util import dropout_supporting_sparse_tensors, to_sparse_tensor


def fully_connected_layer(model, inputs, output_dim, activation_fn, dropout_prob, weight_decay, name):
    with tf.name_scope(name):
        input_dim = int(inputs.shape[1])
        weights = model.add_weight(
            name=f"{name}-weights",
            shape=[input_dim, output_dim],
            initializer=tf.keras.initializers.GlorotUniform(),
            regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        if dropout_prob:
            inputs = dropout_supporting_sparse_tensors(inputs, 1 - dropout_prob)
        if isinstance(inputs, tf.SparseTensor):
            output = tf.sparse.sparse_dense_matmul(inputs, weights)
        else:
            output = tf.matmul(inputs, weights)
        bias = model.add_weight(name=f"{name}-bias", shape=[output_dim], initializer='zeros')
        output = tf.nn.bias_add(output, bias)
        return activation_fn(output) if activation_fn is not None else output


class MLP(GNNModel):
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
            for i in range(0, self.num_layers - 1):
                x = fully_connected_layer(
                    self,
                    inputs=x,
                    output_dim=self.hidden_size,
                    activation_fn=tf.nn.relu,
                    dropout_prob=self.dropout_prob,
                    weight_decay=self.weight_decay,
                    name=f"fc{i}"
                )
            output = fully_connected_layer(
                self,
                inputs=x,
                output_dim=self.targets.shape[1],
                activation_fn=None,
                dropout_prob=self.dropout_prob,
                weight_decay=self.weight_decay,
                name=f"fc{self.num_layers - 1}"
            )
        with tf.name_scope('extract_relevant_nodes'):
            return tf.gather(output, self.nodes_to_consider)

    def _preprocess_features(self, features):
        if self.normalize_features:
            features = row_normalize(features)
        return to_sparse_tensor(features)

    def _preprocess_adj(self, graph_adj):
        return to_sparse_tensor(graph_adj)


MODEL_INGREDIENT = Ingredient('model')


@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                weight_decay, normalize_features,
                num_layers, hidden_size, dropout_prob):
    return MLP(
        node_features, graph_adj, labels, dataset_indices_placeholder,
        num_layers=num_layers, hidden_size=hidden_size,
        dropout_prob=dropout_prob,
        weight_decay=weight_decay,
        normalize_features=normalize_features
    )