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
import sys
import traceback

# TODO(drpng): write up instructions for editing this file in a doc and point to
# the doc instead.
# If you want to edit this file to expose modules in public tensorflow API, you
# need to follow these steps:
# 1. Consult with tensorflow team and get approval for adding a new API to the
#    public interface.
# 2. Document the module in the gen_docs_combined.py.
# 3. Import the module in the main tensorflow namespace by adding an import
#    statement in this file.
# 4. Sanitize the entry point by making sure that your module does not expose
#    transitively imported modules used for implementation, such as os, sys.

# go/tf-wildcard-import
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top

import numpy as np

from tensorflow.python import pywrap_tensorflow

# Protocol buffers
from tensorflow.core.framework.graph_pb2 import *
from tensorflow.core.framework.node_def_pb2 import *
from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.attr_value_pb2 import *
from tensorflow.core.protobuf.meta_graph_pb2 import TensorInfo
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.core.protobuf.config_pb2 import *
from tensorflow.core.protobuf.tensorflow_server_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Framework
from tensorflow.python.framework.framework_lib import *
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# pylint: enable=wildcard-import

# Bring in subpackages.
from tensorflow.python.estimator import estimator_lib as estimator
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.layers import layers
from tensorflow.python.ops import bitwise_ops as bitwise
from tensorflow.python.ops import image_ops as image
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import spectral_ops as spectral
from tensorflow.python.ops.losses import losses
from tensorflow.python.profiler import profiler
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat
from tensorflow.python.saved_model import saved_model
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
from tensorflow.python.ops import confusion_matrix as confusion_matrix_m
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
    'ClusterDef',
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
    'MetaGraphDef',
    'NameAttrList',
    'NodeDef',
    'OptimizerOptions',
    'RunOptions',
    'RunMetadata',
    'SessionLog',
    'Summary',
    'TensorInfo',  # Used for tf.saved_model functionality.
]

# The following symbols are kept for compatibility. It is our plan
# to remove them in the future.
_allowed_symbols.extend([
    'arg_max',
    'arg_min',
    'mul',  # use tf.multiply instead.
    'neg',  # use tf.negative instead.
    'sub',  # use tf.subtract instead.
    'create_partitioned_variables',
    'deserialize_many_sparse',
    'lin_space',
    'list_diff',  # Use tf.listdiff instead.
    'listdiff',  # Use tf.listdiff instead.
    'parse_single_sequence_example',
    'serialize_many_sparse',
    'serialize_sparse',
    'sparse_matmul',  ## use tf.matmul instead.
])

# This is needed temporarily because we import it explicitly.
_allowed_symbols.extend([
    'pywrap_tensorflow',
])

# Dtypes exported by framework/dtypes.py.
# TODO(cwhipkey): expose these through documentation.
_allowed_symbols.extend([
    'QUANTIZED_DTYPES',
    'bfloat16',
    'bool',
    'complex64',
    'complex128',
    'double',
    'half',
    'float16',
    'float32',
    'float64',
    'int16',
    'int32',
    'int64',
    'int8',
    'qint16',
    'qint32',
    'qint8',
    'quint16',
    'quint8',
    'string',
    'uint16',
    'uint8',
    'resource',
])

# Export modules and constants.
_allowed_symbols.extend([
    'app',
    'bitwise',
    'compat',
    'errors',
    'estimator',
    'feature_column',
    'flags',
    'gfile',
    'graph_util',
    'image',
    'logging',
    'losses',
    'metrics',
    'newaxis',
    'nn',
    'python_io',
    'resource_loader',
    'saved_model',
    'sets',
    'spectral',
    'summary',
    'sysconfig',
    'test',
    'train',
    'user_ops',
    'layers',
    'profiler',
])

# Variables framework.versions:
_allowed_symbols.extend([
    'VERSION',
    'GIT_VERSION',
    'COMPILER_VERSION',
])

# Remove all extra symbols that don't have a docstring or are not explicitly
# referenced in the whitelist.
remove_undocumented(__name__, _allowed_symbols, [
    framework_lib, array_ops, check_ops, client_lib, compat, constant_op,
    control_flow_ops, confusion_matrix_m, functional_ops, histogram_ops, io_ops,
    losses, math_ops, metrics, nn, resource_loader, sets, script_ops,
    session_ops, sparse_ops, state_ops, string_ops, summary, tensor_array_ops,
    train, layers, profiler
])

# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
])

# Expose symbols minus dunders, unless they are whitelisted above.
# This is necessary to export our dunders.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]
