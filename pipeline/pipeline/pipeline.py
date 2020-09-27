import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
import tqdm
from typing import Dict
from .model import Model
from .metrics import MicroPrecision, MicroRecall, MacroPrecision, MicroF1Score, MacroRecall, MacroF1Score, HammingLoss
from .losses import bp_mll

class Pipeline:

      class InvalidLossError(Exception):

            def __init__(self, msg: str) -> None:
                super(Pipeline.InvalidLossError, self).__init__(msg)

      class InvalidOptimizerError(Exception):

            def __init__(self, msg: str) -> None:
                super(Pipeline.InvalidOptimizerError, self).__init__(msg)

      class InvalidMetricError(Exception):

            def __init__(self, msg: str) -> None:
                super(Pipeline.InvalidMetricError, self).__init__(msg)

      LOSSES: Dict = {"bp_mll": bp_mll, "huber_loss": tf.losses.huber_loss, "log_loss": tf.losses.log_loss,
                      "mse": tf.losses.mean_squared_error, "softmax_cross_entropy": tf.losses.softmax_cross_entropy,
                      "sigmoid_cross_entropy": tf.losses.sigmoid_cross_entropy}

      OPTIMIZERS: Dict = {"adagrad": tf.train.AdagradOptimizer, "adadelta": tf.train.AdadeltaOptimizer,
                          "rmsprop": tf.train.RMSPropOptimizer, "adam": tf.train.AdamOptimizer}

      EVALUATION_METRICS: Dict = {"micro_precision": MicroPrecision, "micro_recall": MicroRecall, "micro_f1_score": MicroF1Score,
                                  "macro_precision": MacroPrecision, "macro_recall": MacroRecall, "macro_f1_score": MacroF1Score,
                                  "hamming_loss": HammingLoss}

      def __init__(self, model: Model, batch_size: int, n_epoch: int, loss: str = None,
                   optimizer: str = None, lr: float = None) -> None:
          self._batch_size = batch_size
          self._n_epoch = epoch
          self._loss = loss
          self._optimizer = optimizer
          self._lr = lr
          self._X_placeholder = tf.placeholder(shape=[None] + list(model.shape_X()), dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=[None] + list(model.shape_y()), dtype=tf.int32)
          self._iterator = self._generate_iterator()
          self._model = self._get_model(model)
          self._check_loss()
          self._check_evaluation_metrics()

      def _generate_iterator(self) -> tf.data.Iterator:
          dataset = tf.data.Dataset.from_tensor_slices((self._X_placeholder, self._y_placeholder))
          dataset = dataset.shuffle(tf.shape(self._X_placeholder)[0]).batch(self._batch_size)
          return dataset.make_initializable_iterator()

      def _get_model(self, model: Model) -> Model:
          X_, y_ = self._iterator.get_next()
          return model(X_, y_)

      def _check_loss(self) -> None:
          if not self._model.grad:
             loss = Pipeline.LOSSES.get(self._loss, None)
             if not loss:
                raise Pipeline.InvalidLossError("Invalid loss: {}".format(self._loss))
             self._model.loss = loss(self._model.y, self._model.y_hat)
             optimizer = Pipeline.OPTIMIZERS.get(self._optimizer, None)
             if not optimizer:
                raise Pipeline.InvalidOptimizerError("Invalid optimizer: {}".format(self._optimizer))
             self._model.grad = optimizer(learning_rate=self._lr).minimize(self._model.loss)

      def _check_evaluation_metrics(self) -> None:
          if not self._model.evaluation_ops:
             metric_ops = list()
             for metric in self._model.evaluation_metrics:
                 if not Pipeline.EVALUATION_METRICS.get(metric, None):
                    raise Pipeline.InvalidMetricError("Invalid evaluation metric: {}".format(metric))
                 metric_ops.append(Pipeline.EVALUATION_METRICS[metric](self._model.y, self._model.y_hat))
             self._model.evaluation_ops = metric_ops

      def run(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
          def run_(session, total_loss, total_accuracy) -> List:
              _, loss = session.run([self._model.grad, self._model.loss])
              accuracy_scores = session.run(self._model.evaluation_ops)
              total_loss += loss
              total_accuracy = list(map(lambda x, y: x+y, accuracy_scores, total_accuracy))
              total_loss, total_accuracy
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          session = tf.Session(config=config)
          with session.graph.as_default():
               session.run(tf.global_variables_initializer())
               for epoch in range(self._n_epoch):
                   train_loss, train_accuracy = 0, [0 for _ in range(len(self._model.evaluation_metrics))]
                   test_loss, test_accuracy = 0, [0 for _ in range(len(self._model.evaluation_metrics))]
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_train,
                                                                      self._y_placeholder: y_train})
                   with tdqm(total=len(y_train)) as progress:
                        try:
                           while True:
                                 train_loss, train_accuracy = run_(session, train_loss, train_accuracy)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           continue
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_test,
                                                                      self._y_placeholder: y_test})
                   with tdqm(total=len(y_test)) as progress:
                        try:
                           while True:
                                 test_loss, test_accuracy = run_(session, test_loss, test_accuracy)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           continue
                   print(f"\nepoch: \033[36m{epoch+1}\033[0m")
                   print(f"\ttraining set:")
                   print(f"\t\tloss: \033[32m{train_loss/len(y_train)}\033[0m")
                   for metric, accuracy in zip(self._model.evaluation_metrics, train_accuracy):
                       print(f"\t\t\t{metric}: \033[32m{accuracy/len(y_train)}\033[0m")
                   print(f"\ttest set:")
                   print(f"\t\tloss: \033[35m{test_loss/len(y_test)}\033[0m")
                   for metric, accuracy in zip(self._model.evaluation_metrics, test_accuracy):
                       print(f"\t\t\t{metric}: \033[32m{accuracy/len(y_test)}\033[0m")

      def __del__(self) -> None:
          self._session.close()
