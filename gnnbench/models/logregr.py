import tensorflow as tf
from sacred import Ingredient

from gnnbench.data.preprocess import row_normalize
from gnnbench.models.base_model import GNNModel
from gnnbench.util import to_sparse_tensor


class LogisticRegression(GNNModel):
    def __init__(self, features, graph_adj, targets, nodes_to_consider, weight_decay, normalize_features):
        self.normalize_features = normalize_features
        with tf.name_scope('extract_relevant_nodes'):
            targets = tf.gather(targets, nodes_to_consider)
        super().__init__(features, graph_adj, targets)
        self.nodes_to_consider = nodes_to_consider
        self.weight_decay = weight_decay
        self._build_model_graphs()

    def _inference(self):
        with tf.name_scope('inference'):
            weights = self.add_weight(
                name="weights",
                shape=[int(self.features.get_shape()[1]), self.targets.shape[1]],
                initializer=tf.keras.initializers.GlorotUniform(),
                regularizer=tf.keras.regularizers.l2(self.weight_decay)
            )
            output = tf.sparse.sparse_dense_matmul(self.features, weights)
            bias = self.add_weight(
                name="bias",
                shape=[self.targets.shape[1]],
                initializer='zeros'
            )
            output = tf.nn.bias_add(output, bias)
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
                weight_decay, normalize_features):
    return LogisticRegression(node_features, graph_adj, labels, dataset_indices_placeholder,
                              weight_decay=weight_decay,
                              normalize_features=normalize_features)