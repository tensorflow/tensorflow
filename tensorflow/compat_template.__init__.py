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

# Lazy load Keras v2
_tf_uses_legacy_keras = (
    _os.environ.get("TF_USE_LEGACY_KERAS", None) in ("true", "True", "1"))
setattr(_current_module, "keras", _KerasLazyLoader(globals(), mode="v2"))
if _tf_uses_legacy_keras:
  _module_dir = _module_util.get_parent_dir_for_name("tf_keras.api._v2.keras")
else:
  _module_dir = _module_util.get_parent_dir_for_name("keras.api._v2.keras")
_current_module.__path__ = [_module_dir] + _current_module.__path__


# We would like the following to work for fully enabling 2.0 in a 1.0 install:
#
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
#
# This make this one symbol available directly.
from tensorflow.python.compat.v2_compat import enable_v2_behavior  # pylint: disable=g-import-not-at-top
setattr(_current_module, "enable_v2_behavior", enable_v2_behavior)

# Add module aliases
_losses = _KerasLazyLoader(
    globals(), submodule="losses", name="losses", mode="v2")
_metrics = _KerasLazyLoader(
    globals(), submodule="metrics", name="metrics", mode="v2")
_optimizers = _KerasLazyLoader(
    globals(), submodule="optimizers", name="optimizers", mode="v2")
_initializers = _KerasLazyLoader(
    globals(), submodule="initializers", name="initializers", mode="v2")
setattr(_current_module, "losses", _losses)
setattr(_current_module, "metrics", _metrics)
setattr(_current_module, "optimizers", _optimizers)
setattr(_current_module, "initializers", _initializers)
