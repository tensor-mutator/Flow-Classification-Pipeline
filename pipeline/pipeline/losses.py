import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf

__all__ = ["bp_mll"]

def bp_mll(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError
