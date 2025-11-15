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
from tensorflow.core.protobuf.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Checkpoint Sharding

# Compat

# Compiler

# Data

# Distributions

# TensorFlow Debugger (tfdbg).

# Distribute

# DLPack

# Eager
from tensorflow.python.eager import monitoring as _monitoring

# Check whether TF2_BEHAVIOR is turned on.
from tensorflow.python import tf2 as _tf2
_tf2_gauge = _monitoring.BoolGauge(
    '/tensorflow/api/tf2_enable', 'Environment variable TF2_BEHAVIOR is set".')
_tf2_gauge.get_cell().set(_tf2.enabled())

# Feature Column

# Framework
from tensorflow.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
from tensorflow.python.framework.versions import *

# Function
from tensorflow.core.function.trace_type import *

# IO

# Module

# Ops
from tensorflow.python.ops.random_crop_ops import *
from tensorflow.python.ops import nn
from tensorflow.python.ops import ragged
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.ragged import ragged_ops

# Platform

# Pywrap TF

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

# Saved Model

# Session
from tensorflow.python.client.client_lib import *

# Summary

# TPU

# Training

# User Ops

# Util

# _internal APIs
from tensorflow.python.distribute.experimental.rpc.rpc_ops import *
from tensorflow.python.distribute.merge_call_interim import *
from tensorflow.python.distribute.multi_process_runner import *
from tensorflow.python.distribute.multi_worker_test_base import *
from tensorflow.python.distribute.sharded_variable import *
from tensorflow.python.distribute.strategy_combinations import *
from tensorflow.python.framework.combinations import *
from tensorflow.python.framework.composite_tensor import *
from tensorflow.python.framework.test_combinations import *

from tensorflow.python.distribute.parameter_server_strategy_v2 import *
from tensorflow.python.distribute.coordinator.cluster_coordinator import *
from tensorflow.python.distribute.failure_handling.failure_handling import *
from tensorflow.python.distribute.failure_handling.preemption_watcher import *


# Update dispatch decorator docstrings to contain lists of registered APIs.
# (This should come after any imports that register APIs.)
from tensorflow.python.util import dispatch
dispatch.update_docstrings_with_api_lists()

# Export protos
# pylint: disable=undefined-variable
# pylint: enable=undefined-variable
