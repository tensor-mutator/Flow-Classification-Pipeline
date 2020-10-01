import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf

__all__ = ["MicroPrecision", "MicroRecall", "MicroF1Score", "MacroPrecision", "MacroRecall", "MacroF1Score",
           "HammingLoss"]

def TP(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), y, tf.zeros_like(y_hat)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), y, tf.zeros_like(y_hat)))

def FP(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), tf.cast(tf.equal(y, 0), tf.float32),
                                     tf.zeros_like(y_hat)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), tf.cast(tf.equal(y, 0), tf.float32),
                                     tf.zeros_like(y_hat)))

def TN(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.equal(y_hat, 0), tf.cast(tf.equal(y, 0), tf.float32),
                                     tf.zeros_like(y_hat)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.equal(y_hat, 0), tf.cast(tf.equal(y, 0), tf.float32),
                                     tf.zeros_like(y_hat)))

def FN(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.equal(y_hat, 0), y, tf.float32),
                                     tf.zeros_like(y_hat)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.equal(y_hat, 0), y, tf.float32),
                                     tf.zeros_like(y_hat)))

def MicroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    TP_plus_FP = tf.reduce_sum(y_hat)
    TP = tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), y, tf.zeros_like(y_hat)))
    return TP/TP_plus_FP

def MicroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    TP_plus_FN = tf.reduce_sum(y)
    TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_hat, tf.zeros_like(y)))
    return TP/TP_plus_FN

def MicroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    MicroPR = MicroPrecision(y, y_hat)
    MicroRC = MicroRecall(y, y_hat)
    return 2*MicroPR*MicroRC/(MicroPR+MicroRC)

def MacroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    TP_plus_FP = tf.reduce_sum(y_hat, axis=0)
    TP = tf.reduce_sum(tf.where(tf.cast(y_hat, tf.bool), y, tf.zeros_like(y_hat)), axis=0)
    return tf.reduce_mean(TP/TP_plus_FP)

def MacroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    TP_plus_FN = tf.reduce_sum(y, axis=0)
    TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_hat, tf.zeros_like(y)), axis=0)
    return tf.reduce_mean(TP/TP_plus_FN)

def MacroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    MacroPR = MacroPrecision(y, y_hat)
    MacroRC = MacroRecall(y, y_hat)
    return 2*MacroPR*MacroRC/(MacroPR+MacroRC)

def HammingLoss(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError
    
