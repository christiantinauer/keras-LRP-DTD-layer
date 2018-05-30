from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K

from . import patches

def _simple_lrp(layer, R, parameter):
  '''
  LRP according to Eq(56) in DOI: 10.1371/journal.pone.0130140
  '''
  print('_maxpooling3d_simple_lrp')

  volume_patches = patches.extract_volume_patches(layer.input,
                                                  layer.pool_size,
                                                  layer.strides,
                                                  layer.padding)
  Z = K.equal(K.reshape(layer.output,
                        (-1,
                         layer.output_shape[1], layer.output_shape[2], layer.output_shape[3],
                         1, 1, 1,
                         layer.output_shape[4])),
              volume_patches)
  Z = K.switch(Z, K.ones_like(Z, dtype=K.floatx()), K.zeros_like(Z, dtype=K.floatx()))
  Zs = K.sum(Z, axis=[4, 5, 6], keepdims=True)
  Zs += 1e-12 * K.switch(
    K.greater_equal(Zs, 0),
    K.ones_like(Zs, dtype=K.floatx()),
    K.ones_like(Zs, dtype=K.floatx()) * -1.)
  result = (Z/Zs) * K.reshape(R,
                              (-1,
                               layer.output_shape[1],
                               layer.output_shape[2],
                               layer.output_shape[3],
                               1, 1, 1,
                               layer.output_shape[4]))
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.pool_size,
                                         layer.strides,
                                         layer.padding)

def _epsilon_lrp(layer, R, epsilon):
  '''
  Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
  '''
  return _simple_lrp(layer, R, epsilon)

def _ww_lrp(layer, R, parameter): 
  '''
  There are no weights to use. default to _flat_lrp
  '''
  return _flat_lrp(layer, R, parameter)

def _flat_lrp(layer, R, parameter):
  '''
  Distribute relevance for each output evenly to the output neurons' receptive fields.
  '''
  print('_maxpooling3d_flat_lrp')

  Z = K.ones((layer.pool_size[0], layer.pool_size[1], layer.pool_size[2], 1), dtype=K.floatx())
  Zs = K.sum(Z, axis=[0, 1, 2], keepdims=True)
  result = (Z/Zs) * K.reshape(R,
                              (-1,
                               layer.output_shape[1],
                               layer.output_shape[2],
                               layer.output_shape[3],
                               1, 1, 1,
                               layer.output_shape[4]))
  return patches.restitch_volume_patches(result,
                                         layer.input_shape,
                                         layer.pool_size,
                                         layer.strides,
                                         layer.padding)

def _alphabeta_lrp(layer, R, beta):
  '''
  Since there is only one (or several equally strong) dominant activations, default to _simple_lrp
  '''
  return _simple_lrp(layer, R, beta)
