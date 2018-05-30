from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

def _dense_ww_dtd(layer, R, parameter1, parameter2):
  print('_dense_ww_dtd')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  Z = K.square(layer.kernel)
  Zs = K.sum(Z, axis=0, keepdims=True)
  return K.sum((Z/Zs) * K.expand_dims(R, axis=1), axis=2)

def _dense_z_dtd(layer, R, parameter1, parameter2):
  print('_dense_z_dtd')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  Z = layer.kernel * K.expand_dims(layer.input, axis=-1)
  Zs = K.sum(Z, axis=1, keepdims=True)
  Zs += 1e-12 * K.switch(
    K.greater_equal(Zs, 0),
    K.ones_like(Zs, dtype=K.floatx()),
    K.ones_like(Zs, dtype=K.floatx()) * -1.)
  return K.sum((Z/Zs) * K.expand_dims(R, axis=1), axis=2)

def _dense_zplus_dtd(layer, R, parameter1, parameter2):
  print('_dense_zplus_dtd')

  return _dense_alphabeta_dtd(layer, R, 0., None)

def _dense_zBeta_dtd(layer, R, lower, upper):
  print('_dense_zBeta_dtd')

  raise NotImplementedError('zBeta is not implemented')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

def _dense_alphabeta_dtd(layer, R, beta, parameter2):
  print('_dense_alphabeta_dtd')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  alpha = 1 + beta

  Z = layer.kernel * K.expand_dims(layer.input, axis=-1)

  if not alpha == 0:
    Zp = K.maximum(Z, 0)
    Zsp = K.sum(Zp, axis=1, keepdims=True)
    Zsp += 1e-12 * K.switch(
      K.greater_equal(Zsp, 0),
      K.ones_like(Zsp, dtype=K.floatx()),
      K.ones_like(Zsp, dtype=K.floatx()) * -1.)
    Ralpha = alpha * K.sum((Zp/Zsp) * K.expand_dims(R, axis=1), axis=2)
  else:
    Ralpha = 0

  if not beta == 0:
    Zn = K.minimum(Z, 0)
    Zsn = K.sum(Zn, axis=1, keepdims=True)
    Zsn += 1e-12 * K.switch(
      K.greater_equal(Zsn, 0),
      K.ones_like(Zsn, dtype=K.floatx()),
      K.ones_like(Zsn, dtype=K.floatx()) * -1.)
    Rbeta = beta * K.sum((Zn/Zsn) * K.expand_dims(R, axis=1), axis=2)
  else:
    Rbeta = 0

  return Ralpha - Rbeta
