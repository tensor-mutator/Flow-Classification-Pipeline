__version__ = '0.1.0'

import warnings
with warnings.catch_warnings():  
     warnings.filterwarnings("ignore", category=FutureWarning)
     import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.WARN)
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_eager_execution()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from .model import Model
from .pipeline import Pipeline
from .metrics import (MicroPrecision, MicroRecall, MicroF1Score, MacroPrecision, MacroRecall, MacroF1Score, 
                      HammingLoss, TP, FP, TN, FN)
from .losses import bp_mll
from .config import config

__all__ = ["Model", "Pipeline", "MicroPrecision", "MicroRecall", "MicroF1Score", "MacroPrecision",
           "MacroRecall", "MacroF1Score", "HammingLoss", "TP", "FP", "TN", "FN", "bp_mll", "config"]
