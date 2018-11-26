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
