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
@@assert_scalar_int
@@convert_to_tensor_or_sparse_tensor
@@get_graph_from_inputs
@@is_numeric_tensor
@@is_non_decreasing
@@is_strictly_increasing
@@reduce_sum_n
@@safe_embedding_lookup_sparse
@@with_shape
@@with_same_shape


## Arg_Scope
@@arg_scope
@@add_arg_scope
@@has_arg_scope
@@arg_scoped_arguments

## Variables
@@add_model_variable
@@assert_global_step
@@assert_or_get_global_step
@@create_global_step
@@get_global_step
@@get_or_create_global_step
@@get_local_variables
@@get_model_variables
@@get_unique_variable
@@get_variables_by_name
@@get_variables_by_suffix
@@get_variables_to_restore
@@get_variables
@@local_variable
@@model_variable
@@variable
@@VariableDeviceChooser
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.framework.python.framework import *
from tensorflow.contrib.framework.python.ops import *
from tensorflow.python.util.all_util import make_all

__all__ = make_all(__name__)
