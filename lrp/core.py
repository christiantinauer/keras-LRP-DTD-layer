from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

def _input_lrp(layer, R, parameter):
  print('_input_lrp')

  return R


def _dropout_lrp(layer, R, parameter):
  print('_dropout_lrp')

  return R


def _flatten_lrp(layer, R, parameter):
  print('_flatten_lrp')

  return K.reshape(R, (-1,) + layer.input_shape[1:])


def _activation_lrp(layer, R, parameter):
  print('_activation_lrp')

  # https://github.com/sebastian-lapuschkin/lrp_toolbox/issues/5
  # https://github.com/sebastian-lapuschkin/lrp_toolbox/commit/71932f3306ec58534e6e9fb796a66e478aa7e70c
  # if layer.activation.__name__ == 'softmax':
  #   return layer.input * R

  return R


def _dense_simple_lrp(layer, R, parameter):
  print('_dense_simple_lrp')

  return _dense_epsilon_lrp(layer, R, 1e-12)

def _dense_epsilon_lrp(layer, R, epsilon):
  print('_dense_epsilon_lrp')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  Z = layer.kernel * K.expand_dims(layer.input, axis=-1)
  Zs = K.sum(Z, axis=1, keepdims=True)
  if layer.use_bias:
    Zs += layer.bias
  Zs += epsilon * K.switch(
    K.greater_equal(Zs, 0),
    K.ones_like(Zs, dtype=K.floatx()),
    K.ones_like(Zs, dtype=K.floatx()) * -1.)
  return K.sum((Z/Zs) * K.expand_dims(R, axis=1), axis=2)

def _dense_ww_lrp(layer, R, parameter):
  print('_dense_ww_lrp')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  Z = K.square(layer.kernel)
  Zs = K.sum(Z, axis=0, keepdims=True)
  return K.sum((Z/Zs) * K.expand_dims(R, axis=1), axis=2)

def _dense_flat_lrp(layer, R, parameter):
  print('_dense_flat_lrp')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  Z = K.ones_like(layer.kernel, dtype=K.floatx())
  Zs = K.sum(Z, axis=0, keepdims=True)
  return K.sum((Z/Zs) * K.expand_dims(R, axis=1), axis=2)

def _dense_alphabeta_lrp(layer, R, beta):
  print('_dense_alphabeta_lrp')

  if layer.activation.__name__ == 'softmax':
    raise NotImplementedError('Please use softmax only on Activation layer '
                              'and not directly on Dense layer')

  alpha = 1 + beta

  Z = layer.kernel * K.expand_dims(layer.input, axis=-1)

  if not alpha == 0:
    Zp = K.maximum(Z, 0)
    Zsp = K.sum(Zp, axis=1, keepdims=True)
    if layer.use_bias:
      Zsp += K.maximum(layer.bias, 0)
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
    if layer.use_bias:
      Zsn += K.minimum(layer.bias, 0)
    Zsn += 1e-12 * K.switch(
      K.greater_equal(Zsn, 0),
      K.ones_like(Zsn, dtype=K.floatx()),
      K.ones_like(Zsn, dtype=K.floatx()) * -1.)
    Rbeta = beta * K.sum((Zn/Zsn) * K.expand_dims(R, axis=1), axis=2)
  else:
    Rbeta = 0

  return Ralpha - Rbeta
