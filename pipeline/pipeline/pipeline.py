import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
from contextlib import contextmanager
import numpy as np
import os
from tqdm import tqdm
from glob import glob
from typing import Dict, List, Generator, Any
from .model import Model
from .exceptions import *
from .metrics import (MicroPrecision, MicroRecall, MacroPrecision, MicroF1Score, MacroRecall,
                      MacroF1Score, HammingLoss, TP, FP, TN, FN)
from .losses import bp_mll
from .config import config

GREEN = "\033[32m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
DEFAULT = "\033[0m"
WIPE = "\033[2K"
UP = "\033[2A"

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
                   optimizer: str = None, lr: float = None, evaluation_metrics: List = None,
                   config: bin = config.DEFAULT) -> None:
          self._batch_size = batch_size
          self._n_epoch = n_epoch
          self._loss = loss
          self._optimizer = optimizer
          self._lr = lr if lr else 1e-4
          self._evaluation_metrics = evaluation_metrics
          self._config = config
          self._X_placeholder = tf.placeholder(shape=[None] + list(model.shape_X()) + [3], dtype=tf.float32)
          self._y_placeholder = tf.placeholder(shape=[None] + list(model.shape_y()) + [3], dtype=tf.int32)
          self._iterator = self._generate_iterator()
          self._model = self._get_model(model)
          self._check_loss()
          self._check_evaluation_metrics()
          self._session = self._get_session()
          self._model_name = model.__name__
          self._generate_checkpoint_directory()

      @contextmanager
      def _fit_context(self) -> Generator:
          self._load_weights()
          train_writer, test_writer = self._generate_summary_writer()
          yield self._session, train_writer, test_writer
          self._save_weights()
          if train_writer and test_writer:
             train_writer.close()
             test_writer.close()

      def _load_weights(self) -> None:
          if self._config & config.LOAD_WEIGHTS:
             self._saver = tf.train.Saver(max_to_keep=5)
             if glob(os.path.join(self._model_name, "{}.ckpt.*".format(self._model_name))):
                ckpt = tf.train.get_checkpoint_state(self._model_name)
                self._saver.restore(self._session, ckpt.model_checkpoint_path)

      def _generate_summary_writer(self) -> Any:
          summary_cond = config.LOSS_EVENT+config.HAMMING_LOSS_EVENT+config.MACRO_PRECISION_EVENT+config.MACRO_RECALL_EVENT
          summary_cond += config.MACRO_F1_SCORE_EVENT+config.MICRO_PRECISION_EVENT+config.MICRO_RECALL_EVENT+config.MICRO_F1_SCORE_EVENT
          summary_cond += config.MACRO_TP_EVENT+config.MACRO_FP_EVENT+config.MACRO_TN_EVENT+config.MACRO_FN_EVENT
          summary_cond += config.MICRO_TP_EVENT+config.MICRO_FP_EVENT+config.MICRO_TN_EVENT+config.MICRO_FN_EVENT
          if self._config & summary_cond:
             train_writer = tf.summary.FileWriter(os.path.join(self._model_name, "{} TRAIN EVENT".format(self._model_name)), self._session.graph)
             test_writer = tf.summary.FileWriter(os.path.join(self._model_name, "{} TEST EVENT".format(self._model_name)), self._session.graph)
             return train_writer, test_writer
          return None, None

      def _save_weights(self) -> None:
          if self._config & config.SAVE_WEIGHTS:
             if getattr(self, "_saver", None) is None:
                self._saver = tf.train.Saver(max_to_keep=5)
             self._saver.save(self._session, os.path.join(self._model_name, "{}.ckpt".format(self._model_name)))

      def _generate_checkpoint_directory(self) -> None:
          checkpoint_directory_cond = config.SAVE_WEIGHTS+config.LOSS_EVENT+config.HAMMING_LOSS_EVENT+config.MACRO_PRECISION_EVENT+config.MACRO_RECALL_EVENT
          checkpoint_directory_cond += config.MACRO_F1_SCORE_EVENT+config.MICRO_PRECISION_EVENT+config.MICRO_RECALL_EVENT+config.MICRO_F1_SCORE_EVENT
          checkpoint_directory_cond += config.MACRO_TP_EVENT+config.MACRO_FP_EVENT+config.MACRO_TN_EVENT+config.MACRO_FN_EVENT
          checkpoint_directory_cond += config.MICRO_TP_EVENT+config.MICRO_FP_EVENT+config.MICRO_TN_EVENT+config.MICRO_FN_EVENT
          if self._config & checkpoint_directory_cond:
             if not os.path.exists(self._model_name):
                os.mkdir(self._model_name)

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
          if not self._model.evaluation_ops_train and not self._model.evaluation_ops_test:
             if self._evaluation_metrics:
                train_ops = generate_evaluation_ops(self._evaluation_metrics.get("TRAIN", []))
                self._model.evaluation_ops_train = train_ops
                test_ops = generate_evaluation_ops(self._evaluation_metrics.get("TEST", []))
                self._model.evaluation_ops_test = test_ops

      def _get_session(self) -> tf.Session:
          config = tf.ConfigProto()
          config.gpu_options.allow_growth = True
          return tf.Session(config=config)

      def _save_summary(self, writer: tf.summary.FileWriter, epoch: int, loss: float, metrics: zip) -> None:
          summary = tf.Summary()
          if self._config & config.LOSS_EVENT:
             summary.value.add(tag="{} Performance/Epoch - Loss".format(self._model_name), simple_value=loss)
          for metric, score in metrics:
              if metric == "HammingLoss" and self._config & config.HAMMING_LOSS_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - HammingLoss".format(self._model_name), simple_value=score)
              if metric == "MicroPrecision" and self._config & config.MICRO_PRECISION_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroPrecision".format(self._model_name), simple_value=score)
              if metric == "MicroRecall" and self._config & config.MICRO_RECALL_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroRecall".format(self._model_name), simple_value=score)
              if metric == "MicroF1Score" and self._config & config.MICRO_F1_SCORE_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroF1Score".format(self._model_name), simple_value=score)
              if metric == "MacroPrecision" and self._config & config.MACRO_PRECISION_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroPrecision".format(self._model_name), simple_value=score)
              if metric == "MacroRecall" and self._config & config.MACRO_RECALL_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroRecall".format(self._model_name), simple_value=score)
              if metric == "MacroF1Score" and self._config & config.MACRO_F1_SCORE_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroF1Score".format(self._model_name), simple_value=score)
              if metric == "MacroTP" and self._config & config.MACRO_TP_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroTP".format(self._model_name), simple_value=score)
              if metric == "MacroFP" and self._config & config.MACRO_FP_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroFP".format(self._model_name), simple_value=score)
              if metric == "MacroTN" and self._config & config.MACRO_TN_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroTN".format(self._model_name), simple_value=score)
              if metric == "MacroFN" and self._config & config.MACRO_FN_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MacroFN".format(self._model_name), simple_value=score)
              if metric == "MicroTP" and self._config & config.MICRO_TP_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroTP".format(self._model_name), simple_value=score)
              if metric == "MicroFP" and self._config & config.MICRO_FP_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroFP".format(self._model_name), simple_value=score)
              if metric == "MicroTN" and self._config & config.MICRO_TN_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroTN".format(self._model_name), simple_value=score)
              if metric == "MicroFN" and self._config & config.MICRO_FN_EVENT:
                 summary.value.add(tag="{} Performance/Epoch - MicroFN".format(self._model_name), simple_value=score)
          if writer:
             writer.add_summary(summary, epoch)

      def _fit(self, X_train: np.ndarray, X_test: np.ndarray,
               y_train: np.ndarray, y_test: np.ndarray, session: tf.Session,
               train_writer: tf.summary.FileWriter, test_writer: tf.summary.FileWriter) -> None:
          def run_(session, total_loss, total_accuracy, train=True) -> List:
              if train:
                 _, loss, accuracy_scores = session.run([self._model.grad, self._model.loss, self._model.evaluation_ops_train])
              else:
                 loss, accuracy_scores = session.run([self._model.loss, self._model.evaluation_ops_test])
              total_loss += loss
              total_accuracy = list(map(lambda x, y: x+y, accuracy_scores, total_accuracy))
              return total_loss, total_accuracy
          n_batches_train = np.ceil(np.size(y_train, axis=0)/self._batch_size)
          n_batches_test = np.ceil(np.size(y_test, axis=0)/self._batch_size)
          with session.graph.as_default():
               session.run(tf.global_variables_initializer())
               for epoch in range(self._n_epoch):
                   train_loss, train_score = 0, [0 for _ in range(len(self._evaluation_metrics.get("TRAIN", [])))]
                   test_loss, test_score = 0, [0 for _ in range(len(self._evaluation_metrics.get("TEST", [])))]
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_train,
                                                                      self._y_placeholder: y_train})
                   with tqdm(total=len(y_train)) as progress:
                        try:
                           while True:
                                 train_loss, train_score = run_(session, train_loss, train_score)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   session.run(self._iterator.initializer, feed_dict={self._X_placeholder: X_test,
                                                                      self._y_placeholder: y_test})
                   with tqdm(total=len(y_test)) as progress:
                        try:
                           while True:
                                 test_loss, test_score = run_(session, test_loss, test_score, train=False)
                                 progress.update(self._batch_size)
                        except tf.errors.OutOfRangeError:
                           ...
                   print(f"{UP}\r{WIPE}\n{WIPE}EPOCH: {CYAN}{epoch+1}{DEFAULT}")
                   print(f"\n\tTraining set:")
                   print(f"\t\tLoss: {GREEN}{train_loss/n_batches_train}{DEFAULT}")
                   for metric, score in zip(self._evaluation_metrics.get("TRAIN", []), train_score):
                       print(f"\t\t{metric}: {GREEN}{score/n_batches_train}{DEFAULT}")
                   self._save_summary(train_writer, epoch=epoch+1, loss=train_loss/n_batches_train,
                                      metrics=zip(self._evaluation_metrics.get("TRAIN", []), train_score))
                   print(f"\n\tTest set:")
                   print(f"\t\tLoss: {MAGENTA}{test_loss/n_batches_test}{DEFAULT}")
                   for metric, score in zip(self._evaluation_metrics.get("TEST", []), test_score):
                       print(f"\t\t{metric}: {MAGENTA}{score/n_batches_test}{DEFAULT}")
                   self._save_summary(test_writer, epoch=epoch+1, loss=test_loss/n_batches_test,
                                      metrics=zip(self._evaluation_metrics.get("TEST", []), test_score))

      def fit(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray) -> None:
          with self._fit_context() as [session, train_writer, test_writer]:
               self._fit(X_train, X_test, y_train, y_test, session, train_writer, test_writer)

      def __del__(self) -> None:
          self._session.close()
