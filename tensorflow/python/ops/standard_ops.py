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

# pylint: disable=unused-import
"""Import names of Tensor Flow standard Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform as _platform
import sys as _sys

from tensorflow.python import autograph
from tensorflow.python.training.experimental import loss_scaling_gradient_tape

# pylint: disable=g-bad-import-order
# Imports the following modules so that @RegisterGradient get executed.
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import cudnn_rnn_grad
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.ops import manip_grad
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import random_grad
from tensorflow.python.ops import rnn_grad
from tensorflow.python.ops import sparse_grad
from tensorflow.python.ops import state_grad
from tensorflow.python.ops import tensor_array_grad


# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.array_ops import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.check_ops import *
from tensorflow.python.ops.clip_ops import *
from tensorflow.python.ops.special_math_ops import *
# TODO(vrv): Switch to import * once we're okay with exposing the module.
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.control_flow_ops import case
from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import no_op
from tensorflow.python.ops.control_flow_ops import tuple  # pylint: disable=redefined-builtin
# pylint: enable=redefined-builtin
from tensorflow.python.eager import wrap_function
from tensorflow.python.ops.control_flow_ops import while_loop
from tensorflow.python.ops.batch_ops import *
from tensorflow.python.ops.critical_section_ops import *
from tensorflow.python.ops.data_flow_ops import *
from tensorflow.python.ops.functional_ops import *
from tensorflow.python.ops.gradients import *
from tensorflow.python.ops.histogram_ops import *
from tensorflow.python.ops.init_ops import *
from tensorflow.python.ops.io_ops import *
from tensorflow.python.ops.linalg_ops import *
from tensorflow.python.ops.logging_ops import Print
from tensorflow.python.ops.logging_ops import get_summary_op
from tensorflow.python.ops.logging_ops import timestamp
from tensorflow.python.ops.lookup_ops import initialize_all_tables
from tensorflow.python.ops.lookup_ops import tables_initializer
from tensorflow.python.ops.manip_ops import *
from tensorflow.python.ops.math_ops import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.numerics import *
from tensorflow.python.ops.parsing_ops import *
from tensorflow.python.ops.partitioned_variables import *
from tensorflow.python.ops.proto_ops import *
from tensorflow.python.ops.ragged import ragged_dispatch as _ragged_dispatch
from tensorflow.python.ops.ragged import ragged_operators as _ragged_operators
from tensorflow.python.ops.random_ops import *
from tensorflow.python.ops.script_ops import py_func
from tensorflow.python.ops.session_ops import *
from tensorflow.python.ops.sort_ops import *
from tensorflow.python.ops.sparse_ops import *
from tensorflow.python.ops.state_ops import assign
from tensorflow.python.ops.state_ops import assign_add
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.state_ops import count_up_to
from tensorflow.python.ops.state_ops import scatter_add
from tensorflow.python.ops.state_ops import scatter_div
from tensorflow.python.ops.state_ops import scatter_mul
from tensorflow.python.ops.state_ops import scatter_sub
from tensorflow.python.ops.state_ops import scatter_min
from tensorflow.python.ops.state_ops import scatter_max
from tensorflow.python.ops.state_ops import scatter_update
from tensorflow.python.ops.state_ops import scatter_nd_add
from tensorflow.python.ops.state_ops import scatter_nd_sub
# TODO(simister): Re-enable once binary size increase due to scatter_nd
# ops is under control.
# from tensorflow.python.ops.state_ops import scatter_nd_mul
# from tensorflow.python.ops.state_ops import scatter_nd_div
from tensorflow.python.ops.state_ops import scatter_nd_update
from tensorflow.python.ops.stateless_random_ops import *
from tensorflow.python.ops.string_ops import *
from tensorflow.python.ops.template import *
from tensorflow.python.ops.tensor_array_ops import *
from tensorflow.python.ops.variable_scope import *  # pylint: disable=redefined-builtin
from tensorflow.python.ops.variables import *
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map

# pylint: disable=g-import-not-at-top
if _platform.system() == "Windows":
  from tensorflow.python.compiler.tensorrt import trt_convert_windows as trt
else:
  from tensorflow.python.compiler.tensorrt import trt_convert as trt
# pylint: enable=g-import-not-at-top

# pylint: enable=wildcard-import
# pylint: enable=g-bad-import-order


# These modules were imported to set up RaggedTensor operators and dispatchers:
del _ragged_dispatch, _ragged_operators
