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
"""Import core names of TensorFlow.

Programs that want to build TensorFlow Ops and Graphs without having to import
the constructors and utilities individually can import this file:


import tensorflow as tf
"""

import ctypes
import importlib
import sys
import traceback

# We aim to keep this file minimal and ideally remove completely.
# If you are adding a new file with @tf_export decorators,
# import it in modules_with_exports.py instead.

# go/tf-wildcard-import
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top

from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow

# pylint: enable=wildcard-import

# from tensorflow.python import keras
# from tensorflow.python.layers import layers
from tensorflow.python.saved_model import saved_model
from tensorflow.python.tpu import api

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

from tensorflow.python.compat import v2_compat

# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
    '__cxx11_abi_flag__',
    '__monolithic_build__',
])

# Expose symbols minus dunders, unless they are allowlisted above.
# This is necessary to export our dunders.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]

# TODO(b/296442875): remove this when we remove the tf.distribution package.
# This import is needed for tf.compat.v1.distributions.
from tensorflow.python.ops.distributions import distributions
