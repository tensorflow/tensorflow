# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Bring in all of the public TensorFlow interface into this module."""

import logging as _logging
import os as _os
import sys as _sys
import typing as _typing

from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import LazyLoader as _LazyLoader

# pylint: disable=g-bad-import-order

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]
try:
  from tensorboard.summary._tf import summary
  _current_module.__path__ = (
      [_module_util.get_parent_dir(summary)] + _current_module.__path__)
  setattr(_current_module, "summary", summary)
except ImportError:
  _logging.warning(
      "Limited tf.compat.v2.summary API due to missing TensorBoard "
      "installation.")

# Lazy-load estimator.
_estimator_module = "tensorflow_estimator.python.estimator.api._v2.estimator"
estimator = _LazyLoader("estimator", globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "estimator", estimator)

_keras_module = "keras.api._v2.keras"
_keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", _keras)

# We would like the following to work for fully enabling 2.0 in a 1.0 install:
#
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
#
# This make this one symbol available directly.
from tensorflow.python.compat.v2_compat import enable_v2_behavior  # pylint: disable=g-import-not-at-top
setattr(_current_module, "enable_v2_behavior", enable_v2_behavior)

# Add module aliases
if hasattr(_current_module, 'keras'):
  # It is possible that keras is a lazily loaded module, which might break when
  # actually trying to import it. Have a Try-Catch to make sure it doesn't break
  # when it doing some very initial loading, like tf.compat.v2, etc.
  try:
    _keras_package = "keras.api._v2.keras."
    _losses = _LazyLoader("losses", globals(), _keras_package + "losses")
    _metrics = _LazyLoader("metrics", globals(), _keras_package + "metrics")
    _optimizers = _LazyLoader(
        "optimizers", globals(), _keras_package + "optimizers")
    _initializers = _LazyLoader(
        "initializers", globals(), _keras_package + "initializers")
    setattr(_current_module, "losses", _losses)
    setattr(_current_module, "metrics", _metrics)
    setattr(_current_module, "optimizers", _optimizers)
    setattr(_current_module, "initializers", _initializers)
  except ImportError:
    pass

# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v2 import estimator as estimator
  from keras.api._v2 import keras
  from keras.api._v2.keras import losses
  from keras.api._v2.keras import metrics
  from keras.api._v2.keras import optimizers
  from keras.api._v2.keras import initializers
# pylint: enable=g-import-not-at-top
