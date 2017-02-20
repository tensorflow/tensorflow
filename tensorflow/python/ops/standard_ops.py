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

import sys as _sys

# Imports the following modules so that @RegisterGradient get executed.
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import sparse_grad
from tensorflow.python.ops import state_grad
from tensorflow.python.ops import tensor_array_grad
from tensorflow.python.util.all_util import remove_undocumented


# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.array_ops import *
from tensorflow.python.ops.check_ops import *
from tensorflow.python.ops.clip_ops import *
from tensorflow.python.ops.special_math_ops import *
# TODO(vrv): Switch to import * once we're okay with exposing the module.
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from tensorflow.python.ops.control_flow_ops import Assert
from tensorflow.python.ops.control_flow_ops import group
from tensorflow.python.ops.control_flow_ops import no_op
from tensorflow.python.ops.control_flow_ops import tuple
from tensorflow.python.ops.control_flow_ops import cond
from tensorflow.python.ops.control_flow_ops import case
from tensorflow.python.ops.control_flow_ops import while_loop
from tensorflow.python.ops.data_flow_ops import *
from tensorflow.python.ops.functional_ops import *
from tensorflow.python.ops.gradients import *
from tensorflow.python.ops.histogram_ops import *
from tensorflow.python.ops.init_ops import *
from tensorflow.python.ops.io_ops import *
from tensorflow.python.ops.linalg_ops import *
from tensorflow.python.ops.logging_ops import Print
from tensorflow.python.ops.logging_ops import get_summary_op
from tensorflow.python.ops.math_ops import *
from tensorflow.python.ops.numerics import *
from tensorflow.python.ops.parsing_ops import *
from tensorflow.python.ops.partitioned_variables import *
from tensorflow.python.ops.random_ops import *
from tensorflow.python.ops.script_ops import py_func
from tensorflow.python.ops.session_ops import *
from tensorflow.python.ops.sparse_ops import *
from tensorflow.python.ops.state_ops import assign
from tensorflow.python.ops.state_ops import assign_add
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.state_ops import count_up_to
from tensorflow.python.ops.state_ops import scatter_add
from tensorflow.python.ops.state_ops import scatter_div
from tensorflow.python.ops.state_ops import scatter_mul
from tensorflow.python.ops.state_ops import scatter_sub
from tensorflow.python.ops.state_ops import scatter_update
from tensorflow.python.ops.state_ops import scatter_nd_add
from tensorflow.python.ops.state_ops import scatter_nd_sub
# TODO(simister): Re-enable once binary size increase due to scatter_nd
# ops is under control.
# from tensorflow.python.ops.state_ops import scatter_nd_mul
# from tensorflow.python.ops.state_ops import scatter_nd_div
from tensorflow.python.ops.state_ops import scatter_nd_update
from tensorflow.python.ops.string_ops import *
from tensorflow.python.ops.template import *
from tensorflow.python.ops.tensor_array_ops import *
from tensorflow.python.ops.variable_scope import *
from tensorflow.python.ops.variables import *
# pylint: enable=wildcard-import

#### For use in remove_undocumented below:
from tensorflow.python.framework import constant_op as _constant_op
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import check_ops as _check_ops
from tensorflow.python.ops import clip_ops as _clip_ops
from tensorflow.python.ops import confusion_matrix as _confusion_matrix
from tensorflow.python.ops import control_flow_ops as _control_flow_ops
from tensorflow.python.ops import data_flow_ops as _data_flow_ops
from tensorflow.python.ops import functional_ops as _functional_ops
from tensorflow.python.ops import gradients as _gradients
from tensorflow.python.ops import histogram_ops as _histogram_ops
from tensorflow.python.ops import init_ops as _init_ops
from tensorflow.python.ops import io_ops as _io_ops
from tensorflow.python.ops import linalg_ops as _linalg_ops
from tensorflow.python.ops import logging_ops as _logging_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.ops import numerics as _numerics
from tensorflow.python.ops import parsing_ops as _parsing_ops
from tensorflow.python.ops import partitioned_variables as _partitioned_variables
from tensorflow.python.ops import random_ops as _random_ops
from tensorflow.python.ops import script_ops as _script_ops
from tensorflow.python.ops import session_ops as _session_ops
from tensorflow.python.ops import sparse_ops as _sparse_ops
from tensorflow.python.ops import special_math_ops as _special_math_ops
from tensorflow.python.ops import state_ops as _state_ops
from tensorflow.python.ops import string_ops as _string_ops
from tensorflow.python.ops import template as _template
from tensorflow.python.ops import tensor_array_ops as _tensor_array_ops
from tensorflow.python.ops import variable_scope as _variable_scope
from tensorflow.python.ops import variables as _variables


_allowed_symbols_math_ops = [
    # TODO(drpng): decide if we want to reference these in the documentation.
    "reduced_shape",
    "sparse_segment_mean_grad",
    "sparse_segment_sqrt_n_grad",

    # Legacy: will be removed.
    "arg_max",
    "arg_min",
    "lin_space",
    "sparse_matmul",  # Use tf.matmul.
    # Deprecated (see versions.h):
    "batch_fft",
    "batch_fft2d",
    "batch_fft3d",
    "batch_ifft",
    "batch_ifft2d",
    "batch_ifft3d",
    "mul",  # use tf.multiply instead.
    "neg",  # use tf.negative instead.
    "sub",  # use tf.subtract instead.

    # These are documented in nn.
    # We are are not importing nn because it would create a circular dependency.
    "sigmoid",
    "tanh",
]

_allowed_symbols_array_ops = [
    # TODO(drpng): make sure they are documented.
    # Scalars:
    "NEW_AXIS",
    "SHRINK_AXIS",
    "newaxis",

    # Documented in training.py.
    # I do not import train, to avoid circular dependencies.
    # TODO(drpng): this is defined in gen_array_ops, clearly not the right
    # place.
    "stop_gradient",

    # See gen_docs_combined for tf.copy documentation.
    "copy",

    ## TODO(drpng): make them inaccessible directly.
    ## TODO(drpng): Below, to-doc means that we need to find an appropriate
    ##  documentation section to reference.
    ## For re-exporting to tf.*:
    "constant",
    "edit_distance",  # to-doc
    # From gen_array_ops:
    "copy_host",  # to-doc
    "immutable_const",  # to-doc
    "invert_permutation",  # to-doc
    "quantize_and_dequantize",  # to-doc

    # TODO(drpng): legacy symbols to be removed.
    "list_diff",  # Use tf.listdiff instead.
    "batch_matrix_diag",
    "batch_matrix_band_part",
    "batch_matrix_diag_part",
    "batch_matrix_set_diag",
    "concat_v2",  # Use tf.concat instead.
]

_allowed_symbols_partitioned_variables = [
    "PartitionedVariable",   # Requires doc link.
    # Legacy.
    "create_partitioned_variables",
    "variable_axis_size_partitioner",
    "min_max_variable_partitioner",
    "fixed_size_partitioner",
]

_allowed_symbols_control_flow_ops = [
    # TODO(drpng): Find a place in the documentation to reference these or
    # remove.
    "control_trigger",
    "loop_cond",
    "merge",
    "switch",
]

_allowed_symbols_functional_ops = [
    "nest",  # Used by legacy code.
]

_allowed_symbols_gradients = [
    # Documented in training.py:
    # Not importing training.py to avoid complex graph dependencies.
    "AggregationMethod",
    "gradients",  # tf.gradients = gradients.gradients
    "hessians",
]

_allowed_symbols_clip_ops = [
    # Documented in training.py:
    # Not importing training.py to avoid complex graph dependencies.
    "clip_by_average_norm",
    "clip_by_global_norm",
    "clip_by_norm",
    "clip_by_value",
    "global_norm",
]

_allowed_symbols_image_ops = [
    # Documented in training.py.
    # We are not importing training.py to avoid complex dependencies.
    "audio_summary",
    "histogram_summary",
    "image_summary",
    "merge_all_summaries",
    "merge_summary",
    "scalar_summary",

    # TODO(drpng): link in training.py if it should be documented.
    "get_summary_op",
]

_allowed_symbols_variable_scope_ops = [
    "get_local_variable",  # Documented in framework package.
]

_allowed_symbols_misc = [
    "deserialize_many_sparse",
    "parse_single_sequence_example",
    "serialize_many_sparse",
    "serialize_sparse",
    "confusion_matrix",
]

_allowed_symbols = (_allowed_symbols_array_ops +
                    _allowed_symbols_clip_ops +
                    _allowed_symbols_control_flow_ops +
                    _allowed_symbols_functional_ops +
                    _allowed_symbols_image_ops +
                    _allowed_symbols_gradients +
                    _allowed_symbols_math_ops +
                    _allowed_symbols_variable_scope_ops +
                    _allowed_symbols_misc +
                    _allowed_symbols_partitioned_variables)

remove_undocumented(__name__, _allowed_symbols,
                    [_sys.modules[__name__],
                     _array_ops,
                     _check_ops,
                     _clip_ops,
                     _confusion_matrix,
                     _control_flow_ops,
                     _constant_op,
                     _data_flow_ops,
                     _functional_ops,
                     _gradients,
                     _histogram_ops,
                     _init_ops,
                     _io_ops,
                     _linalg_ops,
                     _logging_ops,
                     _math_ops,
                     _numerics,
                     _parsing_ops,
                     _partitioned_variables,
                     _random_ops,
                     _script_ops,
                     _session_ops,
                     _sparse_ops,
                     _special_math_ops,
                     _state_ops,
                     _string_ops,
                     _template,
                     _tensor_array_ops,
                     _variable_scope,
                     _variables,])
