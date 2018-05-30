from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from tensorflow.python.ops import gen_nn_ops

from ..lrp import maxpooling3d

def _ww_dtd(layer, R, parameter1, parameter2):
  print('_maxpooling3d_ww_dtd')

  # return _z_dtd(layer, R, parameter1, parameter2)
  return maxpooling3d._ww_lrp(layer, R, parameter1)

def _z_dtd(layer, R, parameter1, parameter2):
  print('_maxpooling3d_z_dtd')

  Z = layer.output + 1e-12
  S = R / Z
  C = gen_nn_ops.max_pool3d_grad(layer.input, Z, S, ksize=(1,) + layer.pool_size + (1,),
                                 strides=(1,) + layer.strides + (1,),
                                 padding=K.tensorflow_backend._preprocess_padding(layer.padding))
  return layer.input * C

def _zplus_dtd(layer, R, parameter1, parameter2):
  print('_maxpooling3d_zplus_dtd')

  return _z_dtd(layer, R, parameter1, parameter2)

def _zBeta_dtd(layer, R, lower, upper):
  print('_maxpooling3d_zBeta_dtd')

  return _z_dtd(layer, R, lower, upper)

def _alphabeta_dtd(layer, R, beta, parameter2):
  print('_maxpooling3d_alphabeta_dtd')

  return _z_dtd(layer, R, beta, parameter2)
