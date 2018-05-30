from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

from . import patches

def _simple_lrp(layer, R, parameter):
  '''
  LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
  '''
  print('_conv3d_simple_lrp')

  return _epsilon_lrp(layer, R, 1e-12)

def _epsilon_lrp(layer, R, epsilon):
  '''
  LRP according to Eq(58) in DOI: 10.1371/journal.pone.0130140
  '''
  print('_conv3d_epsilon_lrp')

  volume_patches = patches.extract_volume_patches(layer.input,
                                                  layer.kernel_size,
                                                  layer.strides,
                                                  layer.padding)
  Z = layer.kernel * K.expand_dims(volume_patches, -1)
  Zs = K.reshape(layer.output,
                 (-1,
                  layer.output_shape[1], layer.output_shape[2], layer.output_shape[3],
                  1, 1, 1,
                  1, layer.output_shape[4]))
  Zs += epsilon * K.switch(
    K.greater_equal(Zs, 0),
    K.ones_like(Zs, dtype=K.floatx()),
    K.ones_like(Zs, dtype=K.floatx()) * -1.)
  result = K.sum((Z/Zs) * K.reshape(R,
                                    (-1,
                                     layer.output_shape[1],
                                     layer.output_shape[2],
                                     layer.output_shape[3],
                                     1, 1, 1,
                                     1, layer.output_shape[4])),
                 axis=8)
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.kernel_size,
                                         layer.strides,
                                         layer.padding)

def _ww_lrp(layer, R, parameter): 
  '''
  LRP according to Eq(12) in https://arxiv.org/pdf/1512.02479v1.pdf
  '''
  print('_conv3d_ww_lrp')

  Z = K.square(layer.kernel)
  Zs = K.sum(Z, axis=[0, 1, 2, 3], keepdims=True)
  result = K.sum((Z/Zs) * K.reshape(R,
                                    (-1,
                                     layer.output_shape[1],
                                     layer.output_shape[2],
                                     layer.output_shape[3],
                                     1, 1, 1,
                                     1, layer.output_shape[4])),
                 axis=8)
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.kernel_size,
                                         layer.strides,
                                         layer.padding)

def _flat_lrp(layer, R, parameter):
  '''
  Distribute relevance for each output evenly to the output neurons' receptive fields.
  '''
  print('_conv3d_flat_lrp')

  Z = K.ones_like(layer.kernel)
  Zs = K.sum(Z, axis=[0, 1, 2, 3], keepdims=True)
  result = K.sum((Z/Zs) * K.reshape(R,
                                    (-1,
                                     layer.output_shape[1],
                                     layer.output_shape[2],
                                     layer.output_shape[3],
                                     1, 1, 1,
                                     1, layer.output_shape[4])),
                 axis=8)
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.kernel_size,
                                         layer.strides,
                                         layer.padding)

def _alphabeta_lrp(layer, R, beta):
  '''
  LRP according to Eq(60) in DOI: 10.1371/journal.pone.0130140
  '''
  print('_conv3d_alphabeta_lrp')

  alpha = 1 + beta
  volume_patches = patches.extract_volume_patches(layer.input,
                                                  layer.kernel_size,
                                                  layer.strides,
                                                  layer.padding)
  Z = layer.kernel * K.expand_dims(volume_patches, -1)

  if not alpha == 0:
    Zp = K.maximum(Z, 0)
    Zsp = K.sum(Zp, axis=[4, 5, 6, 7], keepdims=True)
    if layer.use_bias:
      Zsp += K.maximum(layer.bias, 0)
    Zsp += 1e-12 * K.switch(
      K.greater_equal(Zsp, 0),
      K.ones_like(Zsp, dtype=K.floatx()),
      K.ones_like(Zsp, dtype=K.floatx()) * -1.)
    Ralpha = alpha * K.sum((Zp/Zsp) * K.reshape(R,
                                                (-1,
                                                 layer.output_shape[1],
                                                 layer.output_shape[2],
                                                 layer.output_shape[3],
                                                 1, 1, 1,
                                                 1, layer.output_shape[4])),
                           axis=8)
  else:
    Ralpha = 0

  if not beta == 0:
    Zn = K.minimum(Z, 0)
    Zsn = K.sum(Zn, axis=[4, 5, 6, 7], keepdims=True)
    if layer.use_bias:
      Zsn += K.minimum(layer.bias, 0)
    Zsn += 1e-12 * K.switch(
      K.greater_equal(Zsn, 0),
      K.ones_like(Zsn, dtype=K.floatx()),
      K.ones_like(Zsn, dtype=K.floatx()) * -1.)
    Rbeta = beta * K.sum((Zn/Zsn) * K.reshape(R,
                                              (-1,
                                               layer.output_shape[1],
                                               layer.output_shape[2],
                                               layer.output_shape[3],
                                               1, 1, 1,
                                               1, layer.output_shape[4])),
                         axis=8)
  else:
    Rbeta = 0

  result = Ralpha - Rbeta
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.kernel_size,
                                         layer.strides,
                                         layer.padding)
