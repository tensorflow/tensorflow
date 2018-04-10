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
from tensorflow.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors
from tensorflow.python.framework import graph_util

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# Namespaces
from tensorflow.python.ops import initializers_ns as initializers

# pylint: enable=wildcard-import

# Bring in subpackages.
from tensorflow.python import data
from tensorflow.python import keras
from tensorflow.python.estimator import estimator_lib as estimator
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.layers import layers
from tensorflow.python.ops import bitwise_ops as bitwise
from tensorflow.python.ops import image_ops as image
from tensorflow.python.ops import manip_ops as manip
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import spectral_ops as spectral
from tensorflow.python.ops.distributions import distributions
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.losses import losses
from tensorflow.python.profiler import profiler
from tensorflow.python.saved_model import saved_model
from tensorflow.python.summary import summary
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat

# Import boosted trees ops to make sure the ops are registered (but unused).
from tensorflow.python.ops import gen_boosted_trees_ops as _gen_boosted_trees_ops

# Import cudnn rnn ops to make sure their ops are registered.
from tensorflow.python.ops import gen_cudnn_rnn_ops as _


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
from tensorflow.python.util.tf_export import tf_export

# Import modules whose docstrings contribute, for use by remove_undocumented
# below.
from tensorflow.python.client import client_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import framework_lib
from tensorflow.python.framework import subscribe
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

# Eager execution
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.framework.ops import enable_eager_execution

# Necessary for the symbols in this module to be taken into account by
# the namespace management system (API decorators).
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell

# Required due to `rnn` and `rnn_cell` not being imported in `nn` directly
# (due to a circular dependency issue: rnn depends on layers).
nn.dynamic_rnn = rnn.dynamic_rnn
nn.static_rnn = rnn.static_rnn
nn.raw_rnn = rnn.raw_rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn
nn.rnn_cell = rnn_cell

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
    'SummaryMetadata',
    'TensorInfo',  # Used for tf.saved_model functionality.
]

# Export protos
# pylint: disable=undefined-variable
tf_export('AttrValue')(AttrValue)
tf_export('ConfigProto')(ConfigProto)
tf_export('Event', 'summary.Event')(Event)
tf_export('GPUOptions')(GPUOptions)
tf_export('GraphDef')(GraphDef)
tf_export('GraphOptions')(GraphOptions)
tf_export('HistogramProto')(HistogramProto)
tf_export('LogMessage')(LogMessage)
tf_export('MetaGraphDef')(MetaGraphDef)
tf_export('NameAttrList')(NameAttrList)
tf_export('NodeDef')(NodeDef)
tf_export('OptimizerOptions')(OptimizerOptions)
tf_export('RunMetadata')(RunMetadata)
tf_export('RunOptions')(RunOptions)
tf_export('SessionLog', 'summary.SessionLog')(SessionLog)
tf_export('Summary', 'summary.Summary')(Summary)
tf_export('summary.SummaryDescription')(SummaryDescription)
tf_export('SummaryMetadata')(SummaryMetadata)
tf_export('summary.TaggedRunMetadata')(TaggedRunMetadata)
tf_export('TensorInfo')(TensorInfo)
# pylint: enable=undefined-variable


# The following symbols are kept for compatibility. It is our plan
# to remove them in the future.
_allowed_symbols.extend([
    'arg_max',
    'arg_min',
    'create_partitioned_variables',
    'deserialize_many_sparse',
    'lin_space',
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
    'uint64',
    'uint32',
    'uint16',
    'uint8',
    'resource',
    'variant',
])

# Export modules and constants.
_allowed_symbols.extend([
    'app',
    'bitwise',
    'compat',
    'data',
    'distributions',
    'errors',
    'estimator',
    'feature_column',
    'flags',
    'gfile',
    'graph_util',
    'image',
    'initializers',
    'keras',
    'layers',
    'linalg',
    'logging',
    'losses',
    'manip',
    'metrics',
    'newaxis',
    'nn',
    'profiler',
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
])

# Variables framework.versions:
_allowed_symbols.extend([
    'VERSION',
    'GIT_VERSION',
    'COMPILER_VERSION',
    'CXX11_ABI_FLAG',
    'MONOLITHIC_BUILD',
])

# Eager execution
_allowed_symbols.extend([
    'enable_eager_execution',
    'executing_eagerly',
])

# Remove all extra symbols that don't have a docstring or are not explicitly
# referenced in the whitelist.
remove_undocumented(__name__, _allowed_symbols, [
    framework_lib, array_ops, check_ops, client_lib, compat, constant_op,
    control_flow_ops, confusion_matrix_m, data, distributions,
    functional_ops, histogram_ops, io_ops, keras, layers,
    losses, math_ops, metrics, nn, profiler, resource_loader, sets, script_ops,
    session_ops, sparse_ops, state_ops, string_ops, summary, tensor_array_ops,
    train
])

# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
    '__cxx11_abi_flag__',
    '__monolithic_build__',
])

# Expose symbols minus dunders, unless they are whitelisted above.
# This is necessary to export our dunders.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]
