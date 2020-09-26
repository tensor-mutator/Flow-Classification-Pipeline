import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
from .model import Model
from .metrics import 

class Pipeline:

      def __init__(self, model: Model, batch_size: int = 32) -> None:
          self._model = model
          self._batch_size = batch_size
          self._X_placeholder = tf.placeholder(shape=[None] + list(model.shape_X()), dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=[None] + list(model.shape_y()), dtype=tf.int32)
          self._iterator = self._generate_iterator()
          self._check_loss()
          self._check_evaluation_metrics()

      def _generate_iterator(self) -> tf.data.Iterator:
          dataset = tf.data.Dataset.from_tensor_slices((self._x_placeholder, self._y_placeholder))
          dataset = dataset.shuffle(tf.shape(self._x_placeholder)[0]).batch(self._batch_size)
          return dataset.make_initializable_iterator()

      def run(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
