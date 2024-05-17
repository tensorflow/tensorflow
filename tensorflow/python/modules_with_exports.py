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

# Checkpoint Sharding
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.checkpoint.sharding import sharding_policies

# Compat
from tensorflow.python.compat import v2_compat

# Compiler
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla
from tensorflow.python.compiler.mlir import mlir

# Data
from tensorflow.python import data

# Distributions
from tensorflow.python.ops import distributions

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

# Feature Column
from tensorflow.python.feature_column import feature_column_lib as feature_column

# Framework
from tensorflow.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import config
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops

# Function
from tensorflow.core.function.trace_type import *

# IO
from tensorflow.python.lib.io import python_io

# Module
from tensorflow.python.module import module

# Ops
from tensorflow.python.ops.random_crop_ops import *
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import bitwise_ops as bitwise
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import gen_audio_ops
from tensorflow.python.ops import gen_boosted_trees_ops
from tensorflow.python.ops import gen_clustering_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
from tensorflow.python.ops import gen_filesystem_ops
from tensorflow.python.ops import gen_map_ops
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
from tensorflow.python.ops import tensor_getitem_override
from tensorflow.python.ops import while_v2
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

# Platform
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test

# Pywrap TF
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow

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

# Profiler
from tensorflow.python.profiler import profiler
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2
from tensorflow.python.profiler import trace

# Saved Model
from tensorflow.python.saved_model import saved_model

# Session
from tensorflow.python.client.client_lib import *

# Summary
from tensorflow.python.summary import summary
from tensorflow.python.summary import tb_summary

# TPU
from tensorflow.python.tpu import api

# Training
from tensorflow.python.training import training as train
from tensorflow.python.training import quantize_training as _quantize_training

# User Ops
from tensorflow.python.user_ops import user_ops

# Util
from tensorflow.python.util import compat
from tensorflow.python.util import all_util
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

from tensorflow.python.util import tf_decorator_export
from tensorflow.python import proto_exports

# Update dispatch decorator docstrings to contain lists of registered APIs.
# (This should come after any imports that register APIs.)
from tensorflow.python.util import dispatch
dispatch.update_docstrings_with_api_lists()

# Export protos
# pylint: disable=undefined-variable
# pylint: enable=undefined-variable
