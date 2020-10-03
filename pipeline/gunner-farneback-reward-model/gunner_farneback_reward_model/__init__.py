import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
     import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple, List
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
          return ()

      def _build_graph(self) -> None:
          conv_0 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(self._X)
          conv_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.leaky_relu)(conv_0)
          conv_1_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(conv_1)
          conv_2 = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.leaky_relu)(conv_1_1)
          conv_2_1 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(conv_2)
          dense_1 = layers.Dense(units=1024, kernel_initializer=tf.initializers.glorot_normal(),
                                 activation=tf.nn.leaky_relu)(layers.Flatten()(conv_2_1))
          dense_2 = layers.Dense(units=512, kernel_initializer=tf.initializers.glorot_normal(),
                                 activation=tf.nn.leaky_relu)(dense_1)
          dense_3 = layers.Dense(units=512, kernel_initializer=tf.initializers.glorot_normal(), 
                                 activation=tf.nn.leaky_relu)(dense_2)
          y_logits = layers.Dense(units=3)(dense_3)
          self._y_hat = layers.Activation(tf.nn.softmax)(y_logits)
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
