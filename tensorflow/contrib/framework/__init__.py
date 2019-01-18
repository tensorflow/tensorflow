# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Framework utilities.

@@assert_same_float_dtype
@@assert_scalar
@@assert_scalar_int
@@convert_to_tensor_or_sparse_tensor
@@get_graph_from_inputs
@@is_numeric_tensor
@@is_non_decreasing
@@is_strictly_increasing
@@is_tensor
@@reduce_sum_n
@@remove_squeezable_dimensions
@@with_shape
@@with_same_shape

@@deprecated
@@deprecated_args
@@deprecated_arg_values

@@arg_scope
@@add_arg_scope
@@current_arg_scope
@@has_arg_scope
@@arg_scoped_arguments

@@prepend_name_scope
@@strip_name_scope

@@add_model_variable
@@assert_global_step
@@assert_or_get_global_step
@@assign_from_checkpoint
@@assign_from_checkpoint_fn
@@assign_from_values
@@assign_from_values_fn
@@create_global_step
@@filter_variables
@@fuse_op
@@get_global_step
@@get_or_create_global_step
@@get_local_variables
@@get_model_variables
@@get_name_scope
@@get_trainable_variables
@@get_unique_variable
@@get_variables_by_name
@@get_variables_by_suffix
@@get_variable_full_name
@@get_variables_to_restore
@@get_variables
@@global_variable
@@local_variable
@@model_variable
@@variable
@@VariableDeviceChooser
@@convolutional_delta_orthogonal
@@convolutional_orthogonal_1d
@@convolutional_orthogonal_2d
@@convolutional_orthogonal_3d
@@zero_initializer

@@load_checkpoint
@@list_variables
@@load_variable
@@init_from_checkpoint
@@load_and_remap_matrix_initializer
@@load_embedding_initializer
@@load_linear_multiclass_bias_initializer
@@load_variable_slot_initializer

@@argsort
@@py_func
@@sort

@@get_placeholders

@@smart_cond
@@smart_constant_value
@@smart_case

@@CriticalSection

@@BoundedTensorSpec
@@TensorSpec

@@RecordInput
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.framework.python.framework import *
from tensorflow.contrib.framework.python.framework import nest
from tensorflow.contrib.framework.python.ops import *
# pylint: enable=unused-import,wildcard-import

from tensorflow.python.framework.ops import prepend_name_scope
from tensorflow.python.framework.ops import strip_name_scope
from tensorflow.python.framework.smart_cond import smart_case
from tensorflow.python.framework.smart_cond import smart_cond
from tensorflow.python.framework.smart_cond import smart_constant_value
from tensorflow.python.framework.tensor_spec import BoundedTensorSpec
from tensorflow.python.framework.tensor_spec import TensorSpec
from tensorflow.python.ops.data_flow_ops import RecordInput
from tensorflow.python.ops.init_ops import convolutional_delta_orthogonal
from tensorflow.python.ops.init_ops import convolutional_orthogonal_1d
from tensorflow.python.ops.init_ops import convolutional_orthogonal_2d
from tensorflow.python.ops.init_ops import convolutional_orthogonal_3d
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['nest']
_nest_allowed_symbols = [
    'assert_same_structure',
    'is_sequence',
    'is_sequence_or_composite',
    'flatten',
    'flatten_dict_items',
    'pack_sequence_as',
    'map_structure',
    'map_structure_with_paths',
    'assert_shallow_structure',
    'flatten_up_to',
    'map_structure_up_to',
    'get_traverse_shallow_structure',
    'yield_flat_paths',
    'flatten_with_joined_string_paths',
]

remove_undocumented(nest.__name__, allowed_exception_list=_nest_allowed_symbols)
remove_undocumented(__name__, allowed_exception_list=_allowed_symbols)
