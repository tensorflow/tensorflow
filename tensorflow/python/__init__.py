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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""

import ctypes
import importlib
import inspect
import sys
import traceback

# go/tf-wildcard-import
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top

# pywrap_tensorflow is a SWIG generated python library that dynamically loads
# _pywrap_tensorflow.so. The default mode for loading keeps all the symbol
# private and not visible to other libraries that may be loaded. Setting
# the mode to RTLD_GLOBAL to make the symbols visible, so libraries such
# as the ones implementing custom ops can have access to tensorflow
# framework's symbols.
# one catch is that numpy *must* be imported before the call to
# setdlopenflags(), or there is a risk that later c modules will segfault
# when importing numpy (gh-2034).
import numpy as np
_default_dlopen_flags = sys.getdlopenflags()
sys.setdlopenflags(_default_dlopen_flags | ctypes.RTLD_GLOBAL)
from tensorflow.python import pywrap_tensorflow
sys.setdlopenflags(_default_dlopen_flags)

try:
  from tensorflow.core.framework.graph_pb2 import *
except ImportError:
  msg = """%s\n\nError importing tensorflow.  Unless you are using bazel,
you should not try to import tensorflow from its source directory;
please exit the tensorflow source tree, and relaunch your python interpreter
from there.""" % traceback.format_exc()
  raise ImportError(msg)

from tensorflow.core.framework.node_def_pb2 import *
from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.attr_value_pb2 import *
from tensorflow.core.protobuf.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Framework
from tensorflow.python.framework.framework_lib import *
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# Bring in subpackages.
from tensorflow.python.ops import nn
from tensorflow.python.ops import image_ops as image
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat
from tensorflow.python.summary import summary

# Import the names from python/training.py as train.Name.
from tensorflow.python.training import training as train

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test

from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.all_util import make_all

# Import modules whose docstrings contribute, for use by remove_undocumented
# below.
from tensorflow.python.client import client_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import framework_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import histogram_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops

# Symbols whitelisted for export without documentation.
# TODO(cwhipkey): review these and move to contrib, expose through
# documentation, or remove.
_allowed_symbols = [
    'AttrValue',
    'ConfigProto',
    'DeviceSpec',
    'Event',
    'GPUOptions',
    'GRAPH_DEF_VERSION',
    'GRAPH_DEF_VERSION_MIN_CONSUMER',
    'GRAPH_DEF_VERSION_MIN_PRODUCER',
    'GraphDef',
    'GraphOptions',
    'HistogramProto',
    'LogMessage',
    'NameAttrList',
    'NodeDef',
    'OptimizerOptions',
    'RunOptions',
    'RunMetadata',
    'SessionLog',
    'Summary',
    'initialize_all_tables',
]

# The following symbols are kept for compatibility. It is our plan
# to remove them in the future.
_allowed_symbols.extend([
    'arg_max',
    'arg_min',
    'create_partitioned_variables',
    'deserialize_many_sparse',
    'lin_space',
    'list_diff',  # Use tf.listdiff instead.
    'parse_single_sequence_example',
    'serialize_many_sparse',
    'serialize_sparse',
    'sparse_matmul',   ## use tf.matmul instead.
])

# This is needed temporarily because we import it explicitly.
_allowed_symbols.extend([
    'platform',  ## This is included by the tf.learn main template.
    'pywrap_tensorflow',
])

# Dtypes exported by framework/dtypes.py.
# TODO(cwhipkey): expose these through documentation.
_allowed_symbols.extend([
    'QUANTIZED_DTYPES',
    'bfloat16',
    'bfloat16_ref',
    'bool',
    'bool_ref',
    'complex64',
    'complex64_ref',
    'complex128',
    'complex128_ref',
    'double',
    'double_ref',
    'half',
    'half_ref',
    'float16',
    'float16_ref',
    'float32',
    'float32_ref',
    'float64',
    'float64_ref',
    'int16',
    'int16_ref',
    'int32',
    'int32_ref',
    'int64',
    'int64_ref',
    'int8',
    'int8_ref',
    'qint16',
    'qint16_ref',
    'qint32',
    'qint32_ref',
    'qint8',
    'qint8_ref',
    'quint16',
    'quint16_ref',
    'quint8',
    'quint8_ref',
    'string',
    'string_ref',
    'uint16',
    'uint16_ref',
    'uint8',
    'uint8_ref',
])

# Export modules and constants.
_allowed_symbols.extend([
    'app',
    'compat',
    'errors',
    'flags',
    'gfile',
    'image',
    'logging',
    'newaxis',
    'nn',
    'python_io',
    'resource_loader',
    'summary',
    'sysconfig',
    'test',
    'train',
    'user_ops',
])

# Variables framework.versions:
_allowed_symbols.extend([
    'VERSION',
    'GIT_VERSION',
    'COMPILER_VERSION',
])

# Remove all extra symbols that don't have a docstring or are not explicitly
# referenced in the whitelist.
remove_undocumented(__name__, _allowed_symbols,
                    [framework_lib, array_ops, client_lib, check_ops,
                     compat, constant_op, control_flow_ops, functional_ops,
                     histogram_ops, io_ops, math_ops, nn, script_ops,
                     session_ops, sparse_ops, state_ops, string_ops,
                     summary, tensor_array_ops, train])

# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
])

# Expose symbols minus dunders, unless they are whitelisted above.
# This is necessary to export our dunders.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]
