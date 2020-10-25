import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, List, Callable
from pipeline import Pipeline, Model, config
import flappy_bird_dataset

class GunnerFarnebackRewardModel(Model):

      def __init__(self, X: tf.Tensor, y: tf.Tensor = None) -> None:
          self._X = X
          self._y = y
          self._evaluation_ops_train = None
          self._evaluation_ops_test = None
          self._build_graph()

      @staticmethod
      def shape_X() -> Tuple:
          return (64, 64,)

      @staticmethod
      def shape_y() -> Tuple:
          return (3,)

      @staticmethod
      def BahdanauAttention(units: int) -> Callable:
          W1 = layers.Dense(units=units)
          W2 = layers.Dense(units=units)
          V = layers.Dense(1)
          def _op(features: tf.Tensor, hidden: tf.Tensor) -> List:
              hidden_with_time = tf.expand_dims(hidden, axis=1)
              attention_hidden = tf.nn.tanh(W1(features) + W2(hidden_with_time))
              score = V(attention_hidden)
              attention_weights = tf.nn.softmax(score, axis=1)
              context_vector = tf.reduce_sum(attention_weights*features, axis=1)
              return context_vector, attention_weights
          return _op

      @staticmethod
      def Encoder(embedding_dim: int) -> Callable:
          def _op(tensor: tf.Tensor) -> tf.Tensor:
              embedding_out = layers.Dense(units=embedding_dim)(tensor)
              return tf.nn.relu(embedding_out)
          return _op

      @staticmethod
      def Decoder(embedding_dim: int, units: int) -> Callable:
          attn = GunnerFarnebackRewardModel.BahdanauAttention(units)
          lstm = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
          def _op(x: tf.Tensor, features: tf.Tensor, hidden: tf.Tensor, cell: tf.Tensor) -> List:
              context_vector, attention_weights = attn(features, hidden)
              x = tf.concat([context_vector, x], axis=-1)
              return lstm(tf.expand_dims(x, axis=1), initial_state=[hidden, cell])
          return _op

      @staticmethod
      def Attention(blocks: int, units: int, embedding_dim: int, batch_size: int) -> Callable:
          decoder = GunnerFarnebackRewardModel.Decoder(embedding_dim, units)
          def _op(features: tf.Tensor) -> tf.Tensor:
              hidden_state = tf.zeros((batch_size, units))
              cell_state = tf.zeros((batch_size, units))
              for _ in range(blocks):
                  _, hidden_state, cell_state = decoder(hidden_state, features, hidden_state, cell_state)
              return hidden_state
          return _op

      def _build_graph(self) -> None:
          conv_0 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(self._X)
          conv_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.leaky_relu)(conv_0)
          conv_1_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(conv_1)
          conv_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.leaky_relu)(conv_1_1)
          conv_2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(conv_2)
          conv_squeezed = tf.reshape(conv_2_1, [-1, -1, 128])
          attn = GunnerFarnebackRewardModel.Attention(4, 256, 256, tf.shape(self._X)[0])(conv_squeezed)
          #dense_1 = layers.Dense(units=1024, kernel_initializer=tf.initializers.glorot_normal(),
          #                       activation=tf.nn.leaky_relu)(layers.Flatten()(conv_2_1))
          #dense_2 = layers.Dense(units=512, kernel_initializer=tf.initializers.glorot_normal(),
          #                       activation=tf.nn.leaky_relu)(dense_1)
          #dense_3 = layers.Dense(units=512, kernel_initializer=tf.initializers.glorot_normal(), 
          #                       activation=tf.nn.leaky_relu)(dense_2)
          y_logits = layers.Dense(units=3)(attn)
          self._y_hat = layers.Activation(tf.nn.softmax, name="y_hat")(y_logits)
          if self._y is None:
             return
          self._loss = tf.losses.softmax_cross_entropy(logits=y_logits, onehot_labels=self._y)
          optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
          gradients = optimizer.compute_gradients(self._loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "local"))
          for idx, (grad, var) in enumerate(gradients):
              if grad is not None:
                 gradients[idx] = (tf.clip_by_norm(grad, 10), var)
          self._grad = optimizer.apply_gradients(gradients)

      @property
      def evaluation_ops_train(self) -> List[tf.Tensor]:
          return self._evaluation_ops_train

      @evaluation_ops_train.setter
      def evaluation_ops_train(self, evaluation_ops: List[tf.Tensor]) -> None:
          self._evaluation_ops_train = evaluation_ops

      @property
      def evaluation_ops_test(self) -> List[tf.Tensor]:
          return self._evaluation_ops_test

      @evaluation_ops_test.setter
      def evaluation_ops_test(self, evaluation_ops: List[tf.Tensor]) -> None:
          self._evaluation_ops_test = evaluation_ops

def main():
    pipeline = Pipeline(GunnerFarnebackRewardModel, batch_size=32, n_epoch=1000,
                        config=config.SAVE_WEIGHTS+config.LOAD_WEIGHTS+config.LOSS_EVENT+config.HAMMING_LOSS_EVENT,
                        evaluation_metrics=dict(TRAIN=["MacroPrecision", "MacroRecall", "MacroF1Score", "HammingLoss"],
                                                TEST=["MacroPrecision", "MacroRecall", "MacroF1Score", "HammingLoss"]))
    X_train, X_test, y_train, y_test = flappy_bird_dataset.load_flow(resolution=(64, 64), datapoints_per_class=2500)
    pipeline.fit(X_train, X_test, y_train, y_test)
