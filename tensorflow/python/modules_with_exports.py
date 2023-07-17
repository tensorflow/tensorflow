# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Imports modules that should be scanned during API generation.

This file should eventually contain everything we need to scan looking for
tf_export decorators.
"""
# go/tf-wildcard-import
# pylint: disable=wildcard-import,g-bad-import-order,g-import-not-at-top
# pylint: disable=unused-import,g-importing-member

# Protocol buffers
from tensorflow.core.framework.graph_pb2 import *
from tensorflow.core.framework.node_def_pb2 import *
from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.attr_value_pb2 import *
from tensorflow.core.protobuf.meta_graph_pb2 import TensorInfo
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.core.protobuf.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Compiler
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla
from tensorflow.python.compiler.mlir import mlir

# Data
from tensorflow.python import data

# TensorFlow Debugger (tfdbg).
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.ops import gen_debug_ops

# Distribute
from tensorflow.python import distribute

# DLPack
from tensorflow.python.dlpack.dlpack import from_dlpack
from tensorflow.python.dlpack.dlpack import to_dlpack

# Eager
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring as _monitoring
from tensorflow.python.eager import remote

# Check whether TF2_BEHAVIOR is turned on.
from tensorflow.python import tf2 as _tf2
_tf2_gauge = _monitoring.BoolGauge(
    '/tensorflow/api/tf2_enable', 'Environment variable TF2_BEHAVIOR is set".')
_tf2_gauge.get_cell().set(_tf2.enabled())

# Framework
from tensorflow.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.random_crop_ops import *
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import bitwise_ops as bitwise
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_audio_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.ops import gen_tpu_ops
from tensorflow.python.ops import gen_uniform_quant_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops as image
from tensorflow.python.ops import initializers_ns as initializers
from tensorflow.python.ops import manip_ops as manip
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn
from tensorflow.python.ops import ragged
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import sets
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.distributions import distributions
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.ops.linalg.sparse import sparse
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.ragged import ragged_ops
from tensorflow.python.ops.signal import signal
from tensorflow.python.ops.structured import structured_ops as _structured_ops

# Update the RaggedTensor package docs w/ a list of ops that support dispatch.
ragged.__doc__ += ragged_ops.ragged_dispatch.ragged_op_list()

# Required due to `rnn` and `rnn_cell` not being imported in `nn` directly
# (due to a circular dependency issue: rnn depends on layers).
nn.dynamic_rnn = rnn.dynamic_rnn
nn.static_rnn = rnn.static_rnn
nn.raw_rnn = rnn.raw_rnn
nn.bidirectional_dynamic_rnn = rnn.bidirectional_dynamic_rnn
nn.static_state_saving_rnn = rnn.static_state_saving_rnn
nn.rnn_cell = rnn_cell

# Function
from tensorflow.core.function.trace_type import *

from tensorflow.python.util.tf_export import tf_export

# _internal APIs
from tensorflow.python.distribute.combinations import generate
from tensorflow.python.distribute.experimental.rpc.rpc_ops import *
from tensorflow.python.distribute.merge_call_interim import *
from tensorflow.python.distribute.multi_process_runner import *
from tensorflow.python.distribute.multi_worker_test_base import *
from tensorflow.python.distribute.sharded_variable import *
from tensorflow.python.distribute.strategy_combinations import *
from tensorflow.python.framework.combinations import *
from tensorflow.python.framework.composite_tensor import *
from tensorflow.python.framework.test_combinations import *
from tensorflow.python.util.tf_decorator import make_decorator
from tensorflow.python.util.tf_decorator import unwrap

from tensorflow.python.distribute.parameter_server_strategy_v2 import *
from tensorflow.python.distribute.coordinator.cluster_coordinator import *
from tensorflow.python.distribute.failure_handling.failure_handling import *
from tensorflow.python.distribute.failure_handling.preemption_watcher import *

tf_export('__internal__.decorator.make_decorator', v1=[])(make_decorator)
tf_export('__internal__.decorator.unwrap', v1=[])(unwrap)

# Export protos
# pylint: disable=undefined-variable
tf_export(v1=['AttrValue'])(AttrValue)
tf_export(v1=['ConfigProto'])(ConfigProto)
tf_export(v1=['Event', 'summary.Event'])(Event)
tf_export(v1=['GPUOptions'])(GPUOptions)
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
tf_export(v1=['Summary', 'summary.Summary'])(Summary)
tf_export(v1=['summary.SummaryDescription'])(SummaryDescription)
tf_export(v1=['SummaryMetadata'])(SummaryMetadata)
tf_export(v1=['summary.TaggedRunMetadata'])(TaggedRunMetadata)
tf_export(v1=['TensorInfo'])(TensorInfo)
# pylint: enable=undefined-variable
