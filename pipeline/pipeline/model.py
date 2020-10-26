from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, List
import tensorflow.compat.v1 as tf

class Model(metaclass=ABCMeta):

      @abstractmethod
      def __init__(self, X: tf.Tensor, y: tf.Tensor = None) -> None:
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
      def X(self) -> tf.Tensor:
          return self._X

      @property
      @abstractmethod
      def evaluation_ops_train(self) -> List[tf.Tensor]:
          ...

      @evaluation_ops_train.setter
      @abstractmethod
      def evaluation_ops_test(self, evaluation_ops: List[tf.Tensor]) -> None:
          ...

      @property
      @abstractmethod
      def evaluation_ops_test(self) -> List[tf.Tensor]:
          ...

      @evaluation_ops_test.setter
      @abstractmethod
      def evaluation_ops_test(self, evaluation_ops: List[tf.Tensor]) -> None:
          ...

      @staticmethod
      @abstractmethod
      def shape_X() -> Tuple:
          ...

      @staticmethod
      @abstractmethod
      def shape_y() -> Tuple:
          ...
