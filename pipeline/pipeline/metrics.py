import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf

__all__ = ["MicroPrecision", "MicroRecall", "MicroF1Score", "MacroPrecision", "MacroRecall", "MacroF1Score",
           "HammingLoss", "TP", "FP", "TN", "FN"]

def TP(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)))

def FP(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), tf.cast(tf.equal(y, 0), tf.int32), tf.zeros_like(y_cap)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), tf.equal(y, 0), tf.zeros_like(y_cap)))

def TN(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.equal(y_cap, 0), tf.cast(tf.equal(y, 0), tf.int32), tf.zeros_like(y_cap)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.equal(y_cap, 0), tf.equal(y, 0), tf.zeros_like(y_cap)))

def FN(y: tf.Tensor, y_hat: tf.Tensor, type: str = "Macro") -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    if type == "Macro":
       return tf.reduce_sum(tf.where(tf.equal(y_cap, 0), y, tf.zeros_like(y_cap)), axis=0)
    else:
       return tf.reduce_sum(tf.where(tf.equal(y_cap, 0), y, tf.zeros_like(y_cap)))

def MicroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    TP_plus_FP = tf.reduce_sum(y_cap)
    TP = tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)))
    return TP/TP_plus_FP

def MicroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    TP_plus_FN = tf.reduce_sum(y)
    TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_cap, tf.zeros_like(y)))
    return TP/TP_plus_FN

def MicroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    MicroPR = MicroPrecision(y, y_hat)
    MicroRC = MicroRecall(y, y_hat)
    return 2*MicroPR*MicroRC/(MicroPR+MicroRC)

def MacroPrecision(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    TP_plus_FP = tf.reduce_sum(y_cap, axis=0)
    TP = tf.reduce_sum(tf.where(tf.cast(y_cap, tf.bool), y, tf.zeros_like(y_cap)), axis=0)
    return tf.reduce_mean(TP/TP_plus_FP)

def MacroRecall(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    TP_plus_FN = tf.reduce_sum(y, axis=0)
    TP = tf.reduce_sum(tf.where(tf.cast(y, tf.bool), y_cap, tf.zeros_like(y)), axis=0)
    return tf.reduce_mean(TP/TP_plus_FN)

def MacroF1Score(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    MacroPR = MacroPrecision(y, y_hat)
    MacroRC = MacroRecall(y, y_hat)
    return 2*MacroPR*MacroRC/(MacroPR+MacroRC)

def HammingLoss(y: tf.Tensor, y_hat: tf.Tensor) -> tf.Tensor:
    y_pred = tf.argmax(y_hat, axis=-1)
    y_cap = tf.one_hot(indices=y_pred, depth=tf.shape(y)[-1], dtype=tf.int32)
    return tf.reduce_sum(tf.cast(tf.not_equal(y, y_cap), tf.int32))/tf.reduce_sum(tf.ones_like(y_cap))
