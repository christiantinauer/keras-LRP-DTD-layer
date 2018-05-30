from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer

from . import core
# from . import maxpooling
from . import maxpooling3d
# from . import convolutional
from . import convolutional3d

LRP_IMPLEMENTATIONS = {
  'Input': {
    'simple': core._input_lrp,
    'flat': core._input_lrp,
    'ww': core._input_lrp,
    'epsilon': core._input_lrp,
    'alphabeta': core._input_lrp,
  },
  'InputLayer': {
    'simple': core._input_lrp,
    'flat': core._input_lrp,
    'ww': core._input_lrp,
    'epsilon': core._input_lrp,
    'alphabeta': core._input_lrp,
  },
  'Dropout': {
    'simple': core._dropout_lrp,
    'flat': core._dropout_lrp,
    'ww': core._dropout_lrp,
    'epsilon': core._dropout_lrp,
    'alphabeta': core._dropout_lrp,
  },
  'Flatten': {
    'simple': core._flatten_lrp,
    'flat': core._flatten_lrp,
    'ww': core._flatten_lrp,
    'epsilon': core._flatten_lrp,
    'alphabeta': core._flatten_lrp,
  },
  'Activation': {
    'simple': core._activation_lrp,
    'flat': core._activation_lrp,
    'ww': core._activation_lrp,
    'epsilon': core._activation_lrp,
    'alphabeta': core._activation_lrp,
  },
  'Softmax': {
    'simple': core._activation_lrp,
    'flat': core._activation_lrp,
    'ww': core._activation_lrp,
    'epsilon': core._activation_lrp,
    'alphabeta': core._activation_lrp,
  },
  'Dense': {
    'simple': core._dense_simple_lrp,
    'flat': core._dense_flat_lrp,
    'ww': core._dense_ww_lrp,
    'epsilon': core._dense_epsilon_lrp,
    'alphabeta': core._dense_alphabeta_lrp,
  },
  # 'MaxPooling2D': {
  #   'simple': maxpooling._maxpooling_simple_lrp_from_gradients,
  #   # 'simple': maxpooling._maxpooling2d_simple_lrp_from_sample,
  #   'flat': maxpooling._maxpooling2d_flat_lrp,
  #   'ww': maxpooling._maxpooling2d_flat_lrp,
  #   'epsilon': maxpooling._maxpooling_simple_lrp_from_gradients,
  #   'alphabeta': maxpooling._maxpooling_simple_lrp_from_gradients
  # },
  'MaxPooling3D': {
    'simple': maxpooling3d._simple_lrp,
    'flat': maxpooling3d._flat_lrp,
    'ww': maxpooling3d._ww_lrp,
    'epsilon': maxpooling3d._epsilon_lrp,
    'alphabeta': maxpooling3d._alphabeta_lrp,
  },
  # 'Conv2D': {
  #   'simple': convolutional._convolution2d_simple_lrp,
  #   # 'simple': convolutional._convolution2d_simple_lrp_from_sample,
  #   'flat': convolutional._convolution2d_flat_lrp,
  #   'ww': convolutional._convolution2d_ww_lrp,
  #   'epsilon': convolutional._convolution2d_epsilon_lrp,
  #   'alphabeta': convolutional._convolution2d_alphabeta_lrp
  # },
  'Conv3D': {
    'simple': convolutional3d._simple_lrp,
    'flat': convolutional3d._flat_lrp,
    'ww': convolutional3d._ww_lrp,
    'epsilon': convolutional3d._epsilon_lrp,
    'alphabeta': convolutional3d._alphabeta_lrp,
  }
}

class LRP(Layer):
  def __init__(self, variant='simple', parameter=None, **kwargs):
    if variant not in {'simple', 'flat', 'ww', 'epsilon', 'alphabeta'}:
      raise ValueError('Invalid variant for Relevance:', variant)

    if (variant == 'epsilon' or variant == 'alphabeta') and parameter is None:
      raise ValueError('Variant ', variant, ' needs parameter to be set.')

    self.variant = variant
    self.parameter = parameter
    super(LRP, self).__init__(**kwargs)

  def build(self, input_shape):
    self.built = True

  def compute_output_shape(self, input_shape):
    if self._relevance_output_shape == None:
      raise Exception('Relevance output shape is not set. Was layer already called?')

    return self._relevance_output_shape

  def call(self, inputs):
    input_tensor = inputs
    R = input_tensor
    previous_layer = None

    while True:
      layer, _, _ = input_tensor._keras_history
      if layer == previous_layer:
        break

      previous_layer = layer
      input_tensor = layer.input

      R = LRP_IMPLEMENTATIONS[layer.__class__.__name__][self.variant](
        layer, R, self.parameter
      )
      layer.relevance = R

    self._relevance_output_shape = input_tensor._keras_shape
    return R

  def get_config(self):
    config = {
      'variant': self.variant,
      'parameter': self.parameter
    }

    base_config = super(LRP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_LRP_of_input_from_model(model, variant='simple', parameter=None):
  return LRP(variant=variant, parameter=parameter)(model.output)
