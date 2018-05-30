from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Layer

from ..lrp import core as lrpcore
from . import core
from . import maxpooling3d
from . import convolutional3d

DTD_IMPLEMENTATIONS = {
  'Input': {
    'ww': lrpcore._input_lrp,
    'z': lrpcore._input_lrp,
    'zplus': lrpcore._input_lrp,
    'zBeta': lrpcore._input_lrp,
    'alphabeta': lrpcore._input_lrp
  },
  'InputLayer': {
    'ww': lrpcore._input_lrp,
    'z': lrpcore._input_lrp,
    'zplus': lrpcore._input_lrp,
    'zBeta': lrpcore._input_lrp,
    'alphabeta': lrpcore._input_lrp
  },
  'Dropout': {
    'ww': lrpcore._dropout_lrp,
    'z': lrpcore._dropout_lrp,
    'zplus': lrpcore._dropout_lrp,
    'zBeta': lrpcore._dropout_lrp,
    'alphabeta': lrpcore._dropout_lrp
  },
  'Flatten': {
    'ww': lrpcore._flatten_lrp,
    'z': lrpcore._flatten_lrp,
    'zplus': lrpcore._flatten_lrp,
    'zBeta': lrpcore._flatten_lrp,
    'alphabeta': lrpcore._flatten_lrp
  },
  'Activation': {
    'ww': lrpcore._activation_lrp,
    'z': lrpcore._activation_lrp,
    'zplus': lrpcore._activation_lrp,
    'zBeta': lrpcore._activation_lrp,
    'alphabeta': lrpcore._activation_lrp
  },
  'Softmax': {
    'ww': lrpcore._activation_lrp,
    'z': lrpcore._activation_lrp,
    'zplus': lrpcore._activation_lrp,
    'zBeta': lrpcore._activation_lrp,
    'alphabeta': lrpcore._activation_lrp
  },
  'Dense': {
    'ww': core._dense_ww_dtd,
    'z': core._dense_z_dtd,
    'zplus': core._dense_zplus_dtd,
    'zBeta': core._dense_zBeta_dtd,
    'alphabeta': core._dense_alphabeta_dtd
  },
  'MaxPooling3D': {
    'ww': maxpooling3d._ww_dtd,
    'z': maxpooling3d._z_dtd,
    'zplus': maxpooling3d._zplus_dtd,
    'zBeta': maxpooling3d._zBeta_dtd,
    'alphabeta': maxpooling3d._alphabeta_dtd
  },
  'Conv3D': {
    'ww': convolutional3d._ww_dtd,
    'z': convolutional3d._z_dtd,
    'zplus': convolutional3d._zplus_dtd,
    'zBeta': convolutional3d._zBeta_dtd,
    'alphabeta': convolutional3d._alphabeta_dtd
  }
}

class DTD(Layer):
  def __init__(self, variant='simple', parameter1=None, parameter2=None, **kwargs):
    if variant not in {'ww', 'z', 'zplus', 'zBeta', 'alphabeta'}:
      raise ValueError('Invalid variant for Relevance:', variant)

    if (variant == 'zBeta' or variant == 'alphabeta') and parameter1 is None:
      raise ValueError('Variant ', variant, ' needs parameter1 to be set.')

    if (variant == 'zBeta') and parameter2 is None:
      raise ValueError('Variant ', variant, ' needs parameter2 to be set.')

    self.variant = variant
    self.parameter1 = parameter1
    self.parameter2 = parameter2
    super(DTD, self).__init__(**kwargs)

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

      R = DTD_IMPLEMENTATIONS[layer.__class__.__name__][self.variant](
        layer, R, self.parameter1, self.parameter2
      )
      layer.relevance = R

    self._relevance_output_shape = input_tensor._keras_shape
    return R

  def get_config(self):
    config = {
      'variant': self.variant,
      'parameter1': self.parameter1,
      'parameter2': self.parameter2
    }

    base_config = super(DTD, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def get_DTD_of_input_from_model(model, variant='z', parameter1=None, parameter2=None):
  return DTD(variant=variant, parameter1=parameter1, parameter2=parameter2)(model.output)
