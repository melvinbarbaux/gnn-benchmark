import tensorflow as tf

__all__ = [
    'GNNModel',
]


class GNNModel(tf.keras.Model):
    """Base class for all Graph Neural Network (GNN) models."""

    def __init__(self, features, graph_adj, targets):
        """Create a model.

        Parameters
        ----------
        graph_adj : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        features : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr]
            Attribute matrix in CSR or numpy format.
        targets : np.ndarray, shape [num_nodes, num_classes]
            One-hot matrix of node labels.
        """
        super(GNNModel, self).__init__()
        self.targets = targets
        self.graph_adj = self._preprocess_adj(graph_adj)
        self.features = self._preprocess_features(features)
        self._build_model_graphs()

    def _inference(self):
        """
        Builds the inference graph of the model.

        Returns
        -------
        logits : tf.Tensor, shape [num_nodes, num_classes]
            The logits produced by the model (before softmax).
        """
        raise NotImplementedError

    def call(self, inputs=None, training=False):
        """
        Overrides tf.keras.Model call. Here, we simply return the inference.
        """
        return self._inference()

    def _predict(self):
        """
        Computes predictions on the targets given in the constructor.

        Returns
        -------
        predictions : tf.Tensor, shape [num_nodes, num_classes]
            Softmax probabilities for each node and each class.
        """
        return tf.nn.softmax(self.inference)

    def _loss(self):
        """
        Computes the cross-entropy plus regularization loss on the targets.

        Returns
        -------
        loss : tf.Tensor, shape [], dtype tf.float32
            The loss tensor.
        """
        output = self.inference
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.targets)
        )
        regularization_losses = self.losses
        if not regularization_losses:
            return loss
        return loss + tf.add_n(regularization_losses)

    def _build_model_graphs(self):
        """
        Builds the inference, prediction and loss parts of the model.
        """
        self.inference = self._inference()
        self.predict = self._predict()
        self.loss = self._loss()

    def _preprocess_features(self, features):
        """
        Preprocess the features. Even if no preprocessing is needed

        Returns
        -------
        features_tensor : tf.Tensor or tf.SparseTensor
            The preprocessed features.
        """
        raise NotImplementedError

    def _preprocess_adj(self, graph_adj):
        """
        Preprocess the adjacency matrix. Even if no preprocessing is needed, conversion en tf.SparseTensor
        peut être effectuée ici.

        Returns
        -------
        graph_adj_tensor : tf.Tensor or tf.SparseTensor
            The preprocessed adjacency matrix.
        """
        raise NotImplementedError

    def optimize(self, learning_rate, global_step):
        """
        Defines a training step function for the model using TensorFlow 2 idioms.

        Parameters
        ----------
        learning_rate : float
            The learning rate for the optimizer.
        global_step : tf.Variable
            A variable tracking the global training step.

        Returns
        -------
        train_step : function
            A tf.function performing one training step and returning the loss.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss_value = self._loss()
            gradients = tape.gradient(loss_value, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            global_step.assign_add(1)
            return loss_value

        return train_step