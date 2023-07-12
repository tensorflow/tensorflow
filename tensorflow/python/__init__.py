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
from tensorflow.python.eager import context

# pylint: enable=wildcard-import

# from tensorflow.python import keras
from tensorflow.python.feature_column import feature_column_lib as feature_column
# from tensorflow.python.layers import layers
from tensorflow.python.module import module
from tensorflow.python.profiler import profiler
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import saved_model
from tensorflow.python.summary import summary
from tensorflow.python.tpu import api
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat

# Import the names from python/training.py as train.Name.
from tensorflow.python.training import training as train
from tensorflow.python.training import quantize_training as _quantize_training

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test

from tensorflow.python.compat import v2_compat

from tensorflow.python.util.all_util import make_all
from tensorflow.python.util.tf_export import tf_export

# Eager execution
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.remote import connect_to_remote_host
from tensorflow.python.eager.def_function import function

# Check whether TF2_BEHAVIOR is turned on.
from tensorflow.python.eager import monitoring as _monitoring
from tensorflow.python import tf2 as _tf2
_tf2_gauge = _monitoring.BoolGauge(
    '/tensorflow/api/tf2_enable', 'Environment variable TF2_BEHAVIOR is set".')
_tf2_gauge.get_cell().set(_tf2.enabled())

# TensorFlow Debugger (tfdbg).
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.ops import gen_debug_ops

# DLPack
from tensorflow.python.dlpack.dlpack import from_dlpack
from tensorflow.python.dlpack.dlpack import to_dlpack

# XLA JIT compiler APIs.
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla

# MLIR APIs.
from tensorflow.python.compiler.mlir import mlir

# Update dispatch decorator docstrings to contain lists of registered APIs.
# (This should come after any imports that register APIs.)
from tensorflow.python.util import dispatch
dispatch.update_docstrings_with_api_lists()

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
