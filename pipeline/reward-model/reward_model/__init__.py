import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
     import tensorflow.compat.v1.keras.layers as layers
from typing import Tuple
from pipeline import Pipeline, Model
import flappy_bird_dataset

class RewardModel(Model):

      def __init__(self, X: tf.Tensor, y: tf.Tensor) -> None:
          self._X = X
          self._y = y
          self._evaluation_ops = None
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
          y_logits = layers.Dense(units=3)
          self._y_hat = layers.Activation(tf.nn.softmax)(y_logits)
          self._loss = tf.losses.softmax_cross_entropy(logits=y_logits, one_hot_labels=self._y)
          optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
          self._grad = optimizer.minimize(self._loss)

def main():
    pipeline = Pipeline(RewardModel, batch_size=32, n_epoch=1000, evaluation_metrics=["macro_precision", "macro_recall",
                                                                                      "macro_f1_score", "hamming_loss"])
    X_train, X_test, y_train, y_test = flappy_bird_dataset.load_flow(resolution=(64, 64))
    pipeline.fit(X_train, X_test, y_train, y_test)
