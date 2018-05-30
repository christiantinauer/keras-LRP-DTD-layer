from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

def _ww_dtd(layer, R, parameter1, parameter2):
  print('_convolutional3d_ww_dtd')

  Z = K.square(layer.kernel)
  Zs = K.sum(Z, axis=[0, 1, 2, 3])
  return K.conv3d_transpose(R, Z / Zs, K.shape(layer.input), strides=layer.strides,
                            padding=layer.padding, data_format=layer.data_format)

def _z_dtd(layer, R, parameter1, parameter2):
  print('_convolutional3d_z_dtd')

  X = layer.input + 1e-12
  Z = K.conv3d(X, layer.kernel, strides=layer.strides, padding=layer.padding,
               data_format=layer.data_format)
  S = R / Z
  C = K.conv3d_transpose(S, layer.kernel, K.shape(layer.input), strides=layer.strides,
                         padding=layer.padding, data_format=layer.data_format)
  return X * C

def _zplus_dtd(layer, R, parameter1, parameter2):
  print('_convolutional3d_zplus_dtd')

  return _alphabeta_dtd(layer, R, .0, None)

def _zBeta_dtd(layer, R, lower, upper):
  print('_convolutional3d_zBeta_dtd')

  raise NotImplementedError('zBeta is not implemented')

def _alphabeta_dtd(layer, R, beta, parameter2):
  print('_convolutional3d_alphabeta_dtd')

  alpha = 1 + beta

  X = layer.input + 1e-12

  if not alpha == 0:
    Wp = K.maximum(layer.kernel, 1e-12)
    Zp = K.conv3d(X, Wp, strides=layer.strides, padding=layer.padding,
                  data_format=layer.data_format)
    Salpha = alpha * (R / Zp)
    Calpha = K.conv3d_transpose(Salpha, Wp, K.shape(layer.input), strides=layer.strides,
                                padding=layer.padding, data_format=layer.data_format)
  else:
    Calpha = 0

  if not beta == 0:
    Wn = K.minimum(layer.kernel, -1e-12)
    Zn = K.conv3d(X, Wn, strides=layer.strides, padding=layer.padding,
                  data_format=layer.data_format)
    Sbeta = -beta * (R / Zn)
    Cbeta = K.conv3d_transpose(Sbeta, Wn, K.shape(layer.input), strides=layer.strides,
                               padding=layer.padding, data_format=layer.data_format)
  else:
    Cbeta = 0

  return X * (Calpha + Cbeta)
