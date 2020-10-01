import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
from pipeline import Pipeline, Model
import flappy_bird_dataset

class LucasKanadeModel(Model):

      def __init__(self, X: tf.Tensor, y: tf.Tensor) -> None:
          self._X = X
          self._y = y
          self._build_graph()

      def _build_graph(self) -> None:
          

def main():
    pipeline = Pipeline(LucasKanadeModel, batch_size=32, n_epoch=1000, evaluation_metrics=["macro_precision", "macro_recall",
                                                                                           "macro_f1_score", "hamming_loss"])
    X_train, X_test, y_train, y_test = flappy_bird_dataset.load_flow()
    pipeline.fit(X_train, X_test, y_train, y_test)
