import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from .model import Model
from .exceptions import *
from .metrics import MicroPrecision, MicroRecall, MacroPrecision, MicroF1Score, MacroRecall, MacroF1Score, HammingLoss
from .losses import bp_mll

GREEN = "\033[32m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
DEFAULT = "\033[0m"

class Pipeline:

      LOSSES: Dict = {"bp_mll": bp_mll, "huber_loss": tf.losses.huber_loss, "log_loss": tf.losses.log_loss,
                      "mse": tf.losses.mean_squared_error, "softmax_cross_entropy": tf.losses.softmax_cross_entropy,
                      "sigmoid_cross_entropy": tf.losses.sigmoid_cross_entropy}

      OPTIMIZERS: Dict = {"adagrad": tf.train.AdagradOptimizer, "adadelta": tf.train.AdadeltaOptimizer,
                          "rmsprop": tf.train.RMSPropOptimizer, "adam": tf.train.AdamOptimizer}

      EVALUATION_METRICS: Dict = {"micro_precision": MicroPrecision, "micro_recall": MicroRecall, "micro_f1_score": MicroF1Score,
                                  "macro_precision": MacroPrecision, "macro_recall": MacroRecall, "macro_f1_score": MacroF1Score,
                                  "hamming_loss": HammingLoss}

      def __init__(self, model: Model, batch_size: int, n_epoch: int, loss: str = None,
                   optimizer: str = None, lr: float = None, evaluation_metrics: List = None) -> None:
          self._batch_size = batch_size
          self._n_epoch = n_epoch
          self._loss = loss
          self._optimizer = optimizer
          self._lr = lr if lr else 1e-4
          self._evaluation_metrics = evaluation_metrics
          self._X_placeholder = tf.placeholder(shape=[None] + list(model.shape_X()) + [3], dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=[None] + list(model.shape_y()) + [3], dtype=tf.int32)
          self._iterator = self._generate_iterator()
          self._model = self._get_model(model)
          self._check_loss()
          self._check_evaluation_metrics()

      def _generate_iterator(self) -> tf.data.Iterator:
          dataset = tf.data.Dataset.from_tensor_slices((self._X_placeholder, self._y_placeholder))
          dataset = dataset.shuffle(tf.cast(tf.shape(self._X_placeholder)[0], tf.int64)).batch(self._batch_size)
          return dataset.make_initializable_iterator()

      def _get_model(self, model: Model) -> Model:
          X_, y_ = self._iterator.get_next()
          return model(X_, y_)

      def _check_loss(self) -> None:
          if not self._model.grad:
             loss = Pipeline.LOSSES.get(self._loss, None)
             if not loss:
                raise InvalidLossError("Invalid loss: {}".format(self._loss))
             self._model.loss = loss(self._model.y, self._model.y_hat)
             optimizer = Pipeline.OPTIMIZERS.get(self._optimizer, None)
             if not optimizer:
                raise InvalidOptimizerError("Invalid optimizer: {}".format(self._optimizer))
             self._model.grad = optimizer(learning_rate=self._lr).minimize(self._model.loss)

      def _check_evaluation_metrics(self) -> None:
          if not self._model.evaluation_ops:
             metric_ops = list()
             for metric in self._evaluation_metrics:
                 if not Pipeline.EVALUATION_METRICS.get(metric, None):
                    raise InvalidMetricError("Invalid evaluation metric: {}".format(metric))
                 metric_ops.append(Pipeline.EVALUATION_METRICS[metric](self._model.y, self._model.y_hat))
             self._model.evaluation_ops = metric_ops

      def fit(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
          def run_(session, total_loss, total_accuracy) -> List:
              _, loss = session.run([self._model.grad, self._model.loss])
              accuracy_scores = session.run(self._model.evaluation_ops)
              total_loss += loss
              total_accuracy = list(map(lambda x, y: x+y, accuracy_scores, total_accuracy))
              return total_loss, total_accuracy
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          self._session = session = tf.Session(config=config)
          with session.graph.as_default():
               session.run(tf.global_variables_initializer())
               for epoch in range(self._n_epoch):
                   train_loss, train_accuracy = 0, [0 for _ in range(len(self._evaluation_metrics))]
                   test_loss, test_accuracy = 0, [0 for _ in range(len(self._evaluation_metrics))]
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_train,
                                                                      self._y_placeholder: y_train})
                   with tqdm(total=len(y_train)) as progress:
                        try:
                           while True:
                                 train_loss, train_accuracy = run_(session, train_loss, train_accuracy)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_test,
                                                                      self._y_placeholder: y_test})
                   with tqdm(total=len(y_test)) as progress:
                        try:
                           while True:
                                 test_loss, test_accuracy = run_(session, test_loss, test_accuracy)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   print(f"\nepoch: {CYAN}{epoch+1}{DEFAULT}")
                   print(f"\ttraining set:")
                   print(f"\t\tloss: {GREEN}{train_loss/len(y_train)}{DEFAULT}")
                   for metric, accuracy in zip(self._evaluation_metrics, train_accuracy):
                       print(f"\t\t\t{metric}: {GREEN}{accuracy/len(y_train)}{DEFAULT}")
                   print(f"\ttest set:")
                   print(f"\t\tloss: {MAGENTA}{test_loss/len(y_test)}{DEFAULT}")
                   for metric, accuracy in zip(self._evaluation_metrics, test_accuracy):
                       print(f"\t\t\t{metric}: {MAGENTA}{accuracy/len(y_test)}{DEFAULT}")

      def __del__(self) -> None:
          self._session.close()
