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

# Lazy-load estimator.
_estimator_module = "tensorflow_estimator.python.estimator.api._v1.estimator"
estimator = _LazyLoader("estimator", globals(), _estimator_module)
_module_dir = _module_util.get_parent_dir_for_name(_estimator_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "estimator", estimator)

_keras_module = "keras.api._v1.keras"
keras = _LazyLoader("keras", globals(), _keras_module)
_module_dir = _module_util.get_parent_dir_for_name(_keras_module)
if _module_dir:
  _current_module.__path__ = [_module_dir] + _current_module.__path__
setattr(_current_module, "keras", keras)

# Explicitly import lazy-loaded modules to support autocompletion.
# pylint: disable=g-import-not-at-top
if _typing.TYPE_CHECKING:
  from tensorflow_estimator.python.estimator.api._v1 import estimator
# pylint: enable=g-import-not-at-top


from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)

# Add module aliases from Keras to TF.
# Some tf endpoints actually lives under Keras.
if hasattr(_current_module, "keras"):
  # It is possible that keras is a lazily loaded module, which might break when
  # actually trying to import it. Have a Try-Catch to make sure it doesn't break
  # when it doing some very initial loading, like tf.compat.v2, etc.
  try:
    _layer_package = "keras.api._v1.keras.__internal__.legacy.layers"
    layers = _LazyLoader("layers", globals(), _layer_package)
    _module_dir = _module_util.get_parent_dir_for_name(_layer_package)
    if _module_dir:
      _current_module.__path__ = [_module_dir] + _current_module.__path__
    setattr(_current_module, "layers", layers)

    _legacy_rnn_package = "keras.api._v1.keras.__internal__.legacy.rnn_cell"
    _rnn_cell = _LazyLoader("legacy_rnn", globals(), _legacy_rnn_package)
    _module_dir = _module_util.get_parent_dir_for_name(_legacy_rnn_package)
    if _module_dir:
      _current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__
    _current_module.nn.rnn_cell = _rnn_cell
  except ImportError:
    pass
