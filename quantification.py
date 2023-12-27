
import tensorflow as tf, keras.backend as K
from keras import losses
from utils import MyWheightType

def quant_loss(weight_type = MyWheightType.C1Q1):  
  def count_CC_loss(y_true, y_pred):
    class_index = K.argmax(y_pred, axis = 1)
    y, idx, CC = tf.unique_with_counts(class_index)

  ##---------------------------------------------------#
  ## y and CC must have all possible positions contained in y_pred
  ##---------------------------------------------------#
  ##Possible indices
    true_index = tf.range(y_pred.shape[1], dtype = y.dtype)
    y_aux = tf.concat([y, true_index], axis = -1)
  ##class_index values in y's head and other values in y's tail
    y, _ = tf.unique(y_aux)

  ##Possible indices have count = 0
    CC_aux = tf.zeros(y_pred.shape[1], dtype = CC.dtype)
    CC_aux = tf.concat([CC, CC_aux], axis = -1)
  ##CC initial values in CC's head and CC's tail has just count = 0 (padding)
    CC = tf.slice(CC_aux, [0], [y_pred.shape[1]])
  #---------------------------------------------------#

  ##Puts CC in the same order of y
    CC = K.gather(CC, tf.contrib.framework.argsort(y))
    CC = K.cast(CC, K.floatx())
  ##Count mean
    CC = CC / K.cast(K.shape(idx), K.floatx())
  #===================================================#

  ##Real count
    y_true_aux = K.cast(y_true, K.floatx())
    y_true_aux = K.mean(y_true, axis = 0)

    ERR = K.abs(y_true_aux - CC)
  ##Each sample has the counting error of its class
    ERR = tf.gather(ERR, class_index)

  ##Weighting cases
    lambda1 = lambda2 = 1.0

    if weight_type == MyWheightType.C2Q1:
      lambda2 = 0.5
    elif weight_type == MyWheightType.C1Q2:
      lambda1 = 0.5    

    return lambda1 * losses.categorical_crossentropy(y_true, y_pred) + lambda2 * ERR

  return count_CC_loss
