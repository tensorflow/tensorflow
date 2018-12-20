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
from tensorflow.python import distribute
from tensorflow.python import keras
from tensorflow.python.feature_column import feature_column_lib as feature_column
from tensorflow.python.layers import layers
from tensorflow.python.ops import bitwise_ops as bitwise
from tensorflow.python.ops import image_ops as image
from tensorflow.python.ops import manip_ops as manip
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import ragged
from tensorflow.python.ops import sets
from tensorflow.python.ops.distributions import distributions
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.signal import signal
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

from tensorflow.python.util.all_util import make_all
from tensorflow.python.util.tf_export import tf_export

# Eager execution
from tensorflow.python.eager.context import executing_eagerly
from tensorflow.python.eager.remote import connect_to_remote_host
from tensorflow.python.eager.def_function import function
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
nn.static_state_saving_rnn = rnn.static_state_saving_rnn
nn.rnn_cell = rnn_cell

# Export protos
# pylint: disable=undefined-variable
tf_export(v1=['AttrValue'])(AttrValue)
tf_export(v1=['ConfigProto'])(ConfigProto)
tf_export('Event', 'summary.Event')(Event)
tf_export(v1=['GPUOptions'])(GPUOptions)
tf_export(v1=['GraphDef'])(GraphDef)
tf_export(v1=['GraphOptions'])(GraphOptions)
tf_export(v1=['HistogramProto'])(HistogramProto)
tf_export(v1=['LogMessage'])(LogMessage)
tf_export(v1=['MetaGraphDef'])(MetaGraphDef)
tf_export(v1=['NameAttrList'])(NameAttrList)
tf_export(v1=['NodeDef'])(NodeDef)
tf_export(v1=['OptimizerOptions'])(OptimizerOptions)
tf_export(v1=['RunMetadata'])(RunMetadata)
tf_export(v1=['RunOptions'])(RunOptions)
tf_export(v1=['SessionLog', 'summary.SessionLog'])(SessionLog)
tf_export('Summary', 'summary.Summary')(Summary)
tf_export('summary.SummaryDescription')(SummaryDescription)
tf_export('SummaryMetadata')(SummaryMetadata)
tf_export('summary.TaggedRunMetadata')(TaggedRunMetadata)
tf_export(v1=['TensorInfo'])(TensorInfo)
# pylint: enable=undefined-variable

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
