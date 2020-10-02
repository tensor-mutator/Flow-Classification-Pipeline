import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from .model import Model
from .exceptions import *
from .metrics import (MicroPrecision, MicroRecall, MacroPrecision, MicroF1Score, MacroRecall,
                      MacroF1Score, HammingLoss, TP, FP, TN, FN)
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

      EVALUATION_METRICS: Dict = {"MicroPrecision": MicroPrecision, "MicroRecall": MicroRecall, "MicroF1Score": MicroF1Score,
                                  "MacroPrecision": MacroPrecision, "MacroRecall": MacroRecall, "MacroF1Score": MacroF1Score,
                                  "HammingLoss": HammingLoss, "MacroTP": TP, "MacroFP": FP, "MacroTN": TN, "MacroFN": FN,
                                  "MicroTP": lambda y, y_hat: TP(y, y_hat, type="micro"), "MicroFP": lambda y, y_hat: FP(y, y_hat, type="micro"),
                                  "MicroTN": lambda y, y_hat: TN(y, y_hat, type="micro"), "MicroFN": lambda y, y_hat: FN(y, y_hat, type="micro"),
                                 }

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
          dataset = dataset.shuffle(tf.cast(tf.shape(self._X_placeholder)[0], tf.int64)).batch(self._batch_size).prefetch(1)
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
          def generate_evaluation_ops(metrics):
              ops = list()
              for metric in metrics:
                  if not Pipeline.EVALUATION_METRICS.get(metric, None):
                     raise InvalidMetricError("Invalid evaluation metric: {}".format(metric))
                  ops.append(Pipeline.EVALUATION_METRICS[metric](self._model.y, self._model.y_hat))
              return ops
          if not self._model.evaluation_ops_train and not self._model_evaluation_ops_test:
             if self._evaluation_metrics:
                ops = generate_avaluation_ops(self._evaluation_metrics.get("TRAIN", []))
                self._model.evaluation_ops_train = ops
                test_ops = list(set.difference(set(self._evaluation_metrics.get("TEST", [])), set(self._evaluation_metrics.get("TRAIN", []))))
                ops = generate_avaluation_ops(test_ops)

      def fit(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
          def run_(session, total_loss, total_accuracy, train=True) -> List:
              if train:
                 _, loss, accuracy_scores = session.run([self._model.grad, self._model.loss, self._model.evaluation_ops_train])
              else:
                 loss, accuracy_scores = session.run([self._model.loss, self._model.evaluation_ops_test])
              total_loss += loss
              total_accuracy = list(map(lambda x, y: x+y, accuracy_scores, total_accuracy))
              return total_loss, total_accuracy
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          n_batches_train = np.ceil(np.size(y_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(y_test, axis=0)/self._batch_size)
          self._session = session = tf.Session(config=config)
          with session.graph.as_default():
               session.run(tf.global_variables_initializer())
               for epoch in range(self._n_epoch):
                   train_loss, train_accuracy = 0, [0 for _ in range(len(self._evaluation_metrics.get("TRAIN", [])))]
                   test_loss, test_accuracy = 0, [0 for _ in range(len(self._evaluation_metrics.get("TEST", [])))]
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_train,
                                                                      self._y_placeholder: y_train})
                   #with tqdm(total=len(y_train)) as progress:
                   try:
                      while True:
                            train_loss, train_accuracy = run_(session, train_loss, train_accuracy)
                                 #progress.update(self._batch_size)
                   except tf.errors.OutOfRangeError:
                      ...
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_test,
                                                                      self._y_placeholder: y_test})
                   #with tqdm(total=len(y_test)) as progress:
                   try:
                      while True:
                            test_loss, test_accuracy = run_(session, test_loss, test_accuracy, train=False)
                                 #progress.update(self._batch_size)
                   except tf.errors.OutOfRangeError:
                      ...
                   print(f"\nEPOCH: {CYAN}{epoch+1}{DEFAULT}")
                   print(f"\tTraining set:")
                   print(f"\t\tLoss: {GREEN}{train_loss/len(y_train)}{DEFAULT}")
                   for metric, accuracy in zip(self._evaluation_metrics.get("TRAIN", []), train_accuracy):
                       print(f"\t\t{metric}: {GREEN}{accuracy/n_batches_train}{DEFAULT}")
                   print(f"\tTest set:")
                   print(f"\t\tLoss: {MAGENTA}{test_loss/len(y_test)}{DEFAULT}")
                   for metric, accuracy in zip(self._evaluation_metrics.get("TEST", []), test_accuracy):
                       print(f"\t\t{metric}: {MAGENTA}{accuracy/n_batches_test}{DEFAULT}")

      def __del__(self) -> None:
          self._session.close()
