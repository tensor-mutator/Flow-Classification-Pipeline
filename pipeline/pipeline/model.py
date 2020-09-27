from abc import ABCMeta, abstractmethod
from typing import Any, Tuple
import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf

class Model(metaclass=ABCMeta):

      @abstractmethod
      def __init__(self, X: tf.Tensor, y: tf.Tensor) -> None:
          ...

      @property
      def loss(self) -> str:
          return self._loss

      @loss.setter
      def loss(self, loss: float) -> None:
          self._loss = loss

      @property
      def grad(self) -> tf.Tensor:
          return self._grad

      @grad.setter
      def grad(self, grad_op: tf.Tensor) -> None:
          self._grad = grad_op

      @property
      def y_hat(self) -> tf.Tensor:
          return self._y_hat

      @property
      def y(self) -> tf.Tensor:
          return self._y

      @property
      @abstractmethod
      def evaluation_ops(self) -> List[tf.Tensor]:
          return self._evaluation_ops

      @evaluation_ops.setter
      def evaluation_ops(self, evaluation_ops: List[tf.Tensor]) -> None:
          self._evaluation_ops = evaluation_ops

      @property
      @abstractmethod
      def evaluation_metrics(self) -> List[str]:
          ...

      @staticmethod
      @abstractmethod
      def shape_X() -> Tuple:
          ...

      @staticmethod
      @abstractmethod
      def shape_y() -> Tuple:
          ...
