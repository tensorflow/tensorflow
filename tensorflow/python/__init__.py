# Copyright 2015 Google Inc. All Rights Reserved.
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

# pylint: disable=wildcard-import,g-bad-import-order
"""Import core names of TensorFlow.

Programs that want to build TensorFlow Ops and Graphs without having to import
the constructors and utilities individually can import this file:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

"""

import inspect
import traceback

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.core.framework.graph_pb2 import *
except ImportError:
  msg = """%s\n\nError importing tensorflow.  Unless you are using bazel,
you should not try to import tensorflow from its source directory;
please exit the tensorflow source tree, and relaunch your python interpreter
from there.""" % traceback.format_exc()
  raise ImportError(msg)

from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.attr_value_pb2 import *
from tensorflow.core.protobuf.config_pb2 import *
from tensorflow.core.util.event_pb2 import *
# Import things out of contrib
from tensorflow import contrib

# Framework
from tensorflow.python.framework.framework_lib import *
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# Bring in subpackages
from tensorflow.python.ops import nn
from tensorflow.python.ops import image_ops as image
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat

# Import the names from python/training.py as train.Name.
from tensorflow.python.training import training as train

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test

from tensorflow.python.util.all_util import make_all

# Import modules whose docstrings contribute, for use by make_all below.
from tensorflow.python.client import client_lib
from tensorflow.python.framework import framework_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops


# Don't export modules except for the few we really want
_whitelist = set([app, compat, contrib, errors, flags, gfile, image,
                  logging, nn, python_io, resource_loader, sysconfig, test,
                  train, user_ops])

# Export all symbols directly accessible from 'tf.' by drawing on the doc
# strings of other modules.
__all__ = make_all(__name__,
                   [framework_lib, array_ops, client_lib, constant_op,
                    control_flow_ops, io_ops, math_ops, nn, script_ops,
                    sparse_ops, state_ops, train])

# Symbols whitelisted for export without documentation.
# TODO(cwhipkey): review these and move to contrib, expose through
# documentation, or remove.
__all__.extend([
    'AttrValue',
    'ClusterDef',
    'ConfigProto',
    'Event',
    'GPUOptions',
    'GRAPH_DEF_VERSION',
    'GRAPH_DEF_VERSION_MIN_CONSUMER',
    'GRAPH_DEF_VERSION_MIN_PRODUCER',
    'GraphDef',
    'GraphOptions',
    'GrpcServer',
    'HistogramProto',
    'JobDef',
    'LogMessage',
    'NameAttrList',
    'NodeDef',
    'OptimizerOptions',
    'PaddingFIFOQueue',
    'RunOptions',
    'RunOutputs',
    'ServerDef',
    'SessionLog',
    'Summary',
    'arg_max',
    'arg_min',
    'assign',
    'assign_add',
    'assign_sub',
    'bitcast',
    'bytes',
    'compat',
    'create_partitioned_variables',
    'deserialize_many_sparse',
    'initialize_all_tables',
    'lin_space',
    'list_diff',
    'parse_single_sequence_example',
    'py_func',
    'scalar_mul',
    'serialize_many_sparse',
    'serialize_sparse',
    'shape_n',
    'sparse_matmul',
    'sparse_segment_mean_grad',
    'sparse_segment_sqrt_n_grad',
    'string_to_hash_bucket',
    'unique_with_counts',
    'user_ops',
])

# Dtypes exported by framework/dtypes.py.
# TODO(cwhipkey): expose these through documentation.
__all__.extend([
    'QUANTIZED_DTYPES',
    'bfloat16', 'bfloat16_ref',
    'bool', 'bool_ref',
    'complex64', 'complex64_ref',
    'double', 'double_ref',
    'float32', 'float32_ref',
    'float64', 'float64_ref',
    'int16', 'int16_ref',
    'int32', 'int32_ref',
    'int64', 'int64_ref',
    'int8', 'int8_ref',
    'qint16', 'qint16_ref',
    'qint32', 'qint32_ref',
    'qint8', 'qint8_ref',
    'quint16', 'quint16_ref',
    'quint8', 'quint8_ref',
    'string', 'string_ref',
    'uint16', 'uint16_ref',
    'uint8', 'uint8_ref',
])

# Export modules.
__all__.extend([
    'app',
    'contrib',
    'errors',
    'flags',
    'gfile',
    'image',
    'logging',
    'nn',
    'python_io',
    'resource_loader',
    'sysconfig',
    'test',
    'train',
])

__all__.append('__version__')
