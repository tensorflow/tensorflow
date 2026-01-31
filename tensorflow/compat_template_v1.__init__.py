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

# pylint: disable=g-bad-import-order,g-import-not-at-top,protected-access

import os as _os
import sys as _sys

from tensorflow.python.tools import module_util as _module_util
from tensorflow.python.util.lazy_loader import KerasLazyLoader as _KerasLazyLoader

# API IMPORTS PLACEHOLDER

# WRAPPER_PLACEHOLDER

# Hook external TensorFlow modules.
_current_module = _sys.modules[__name__]

# Lazy load Keras v1
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals(), mode="v1"))
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v1.keras")
else:
  _module_dir = _module_util.get_parent_dir_for_name("keras.api._v1.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__


from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
_current_module.app.flags = flags  # pylint: disable=undefined-variable
setattr(_current_module, "flags", flags)

# Add module aliases from Keras to TF.
# Some tf endpoints actually lives under Keras.
_current_module.layers = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.layers",
    name="layers",
    mode="v1")
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name(
      "tf_keras.api._v1.keras.__internal__.legacy.layers")
else:
  _module_dir = _module_util.get_parent_dir_for_name(
      "keras.api._v1.keras.__internal__.legacy.layers")
_current_module.__path__ = [_module_dir] + _current_module.__path__

_current_module.nn.rnn_cell = _KerasLazyLoader(
    globals(),
    submodule="__internal__.legacy.rnn_cell",
    name="rnn_cell",
    mode="v1")
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name(
      "tf_keras.api._v1.keras.__internal__.legacy.rnn_cell")
else:
  _module_dir = _module_util.get_parent_dir_for_name(
      "keras.api._v1.keras.__internal__.legacy.rnn_cell")
_current_module.nn.__path__ = [_module_dir] + _current_module.nn.__path__
