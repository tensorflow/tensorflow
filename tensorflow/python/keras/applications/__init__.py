# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras Applications are canned architectures with pre-trained weights."""
# pylint: disable=g-import-not-at-top
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras_applications

from tensorflow.python.keras import backend
from tensorflow.python.keras import engine
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import utils
from tensorflow.python.util import tf_inspect

# `get_submodules_from_kwargs` has been introduced in 1.0.5, but we would
# like to be able to handle prior versions. Note that prior to 1.0.5,
# `keras_applications` did not expose a `__version__` attribute.
if not hasattr(keras_applications, 'get_submodules_from_kwargs'):

  if 'engine' in tf_inspect.getfullargspec(
      keras_applications.set_keras_submodules)[0]:
    keras_applications.set_keras_submodules(
        backend=backend,
        layers=layers,
        models=models,
        utils=utils,
        engine=engine)
  else:
    keras_applications.set_keras_submodules(
        backend=backend,
        layers=layers,
        models=models,
        utils=utils)


def keras_modules_injection(base_fun):
  """Decorator injecting tf.keras replacements for Keras modules.

  Arguments:
      base_fun: Application function to decorate (e.g. `MobileNet`).

  Returns:
      Decorated function that injects keyword argument for the tf.keras
      modules required by the Applications.
  """

  def wrapper(*args, **kwargs):
    if hasattr(keras_applications, 'get_submodules_from_kwargs'):
      kwargs['backend'] = backend
      if 'layers' not in kwargs:
        kwargs['layers'] = layers
      kwargs['models'] = models
      kwargs['utils'] = utils
    return base_fun(*args, **kwargs)
  return wrapper


from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.xception import Xception

del absolute_import
del division
del print_function


"""
`keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)`

MobileNet model, with weights pre-trained on ImageNet.

Note that this model only supports the data format `'channels_last'` (height, width, channels).

The default input size for this model is 224x224.

Arguments

* input_shape: optional shape tuple, only to be specified if `include_top` is `False` (otherwise the input shape has to be `(224, 224, 3)` (with `'channels_last'` data format) or `(3, 224, 224`) (with `'channels_first'` data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. `(200, 200, 3)` would be one valid value.
* alpha: controls the width of the network.
** If `alpha` < 1.0, proportionally decreases the number of filters in each layer.
** If `alpha` > 1.0, proportionally increases the number of filters in each layer.
** If `alpha` = 1, default number of filters from the paper are used at each layer.
* depth_multiplier: depth multiplier for depthwise convolution (also called the resolution multiplier)
* dropout: dropout rate
* include_top: whether to include the fully-connected layer at the top of the network.
* weights: `None` (random initialization) or  `'imagenet'` (ImageNet weights)
* input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
* pooling: Optional pooling mode for feature extraction when `include_top` is `False`.
** `None` means that the output of the model will be the 4D tensor output of the last convolutional layer.
** `'avg'` means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
** `'max'` means that global max pooling will be applied.
** classes: optional number of classes to classify images into, only to be specified if `include_top` is `True`, and if no `weights` argument is specified.

Returns

A Keras `Model` instance.

"""
