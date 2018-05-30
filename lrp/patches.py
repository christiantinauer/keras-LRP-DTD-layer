from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import ceil

from keras import backend as K

import tensorflow as tf
import tensorflow_volumepatches as tf_vp

def extract_volume_patches(volumes, patch_size, strides, padding):
  patches = tf_vp.extract_volume_patches(volumes,
                                         ksizes=[1, patch_size[0], patch_size[1], patch_size[2], 1],
                                         strides=[1, strides[0], strides[1], strides[2], 1],
                                         padding=K.tensorflow_backend._preprocess_padding(padding))
  patches_shape = patches.get_shape().as_list()
  return K.reshape(patches,
                   (-1,
                    patches_shape[1], patches_shape[2], patches_shape[3],
                    patch_size[0], patch_size[1], patch_size[2],
                    int(patches_shape[4] / (patch_size[0] * patch_size[1] * patch_size[2]))))

def restitch_volume_patches(patches, input_shape, patch_size, strides, padding):
  patches_shape = patches.get_shape().as_list()
  patches = K.reshape(patches,
                      (-1,
                       patches_shape[1], patches_shape[2], patches_shape[3],
                       patches_shape[4] * patches_shape[5] * patches_shape[6] * patches_shape[7]))
  return tf_vp.integrate_volume_patches(patches,
                                        (1,) + input_shape[1:],
                                        ksizes=[1, patch_size[0], patch_size[1], patch_size[2], 1],
                                        strides=[1, strides[0], strides[1], strides[2], 1],
                                        padding=K.tensorflow_backend._preprocess_padding(padding))
