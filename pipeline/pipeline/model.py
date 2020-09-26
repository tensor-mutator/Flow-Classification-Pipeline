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
      def loss(self) -> Any:
          return self._loss

      @property
      def grad(self) -> Any:
          return self._grad

      @property
      def evaluation_ops(self) -> List[tf.Tensor]:
          return self._evaluation_ops

      @property
      def evaluation_metrics(self) -> List[str]:
          return self._evaluation_metrics

      @staticmethod
      @abstractmethod
      def shape_X() -> Tuple:
          return self._shape_X

      @staticmethod
      @abstractmethod
      def shape_y() -> Tuple:
          return self._shape_y
