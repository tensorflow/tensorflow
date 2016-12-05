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
"""Python wrapper for input_pipeline_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import resource_loader


_input_pipeline_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_input_pipeline_ops.so"))


def obtain_next(string_list_tensor, counter):
  """Basic wrapper for the ObtainNextOp.

  Args:
    string_list_tensor: A tensor that is a list of strings
    counter: an int64 ref tensor to keep track of which element is returned.

  Returns:
    An op that produces the element at counter + 1 in the list, round
    robin style.
  """
  return _input_pipeline_ops.obtain_next(string_list_tensor, counter)


def seek_next(string_list):
  """Returns an op that seeks the next element in a list of strings.

  Seeking happens in a round robin fashion. This op creates a variable called
  counter that is initialized to -1 and is used to keep track of which element
  in the list was returned.

  Args:
    string_list: A list of strings

  Returns:
    An op that produces the next element in the provided list.
  """
  with variable_scope.variable_scope("obtain_next"):
    counter = variable_scope.get_variable(
        name="obtain_next_counter",
        initializer=constant_op.constant([-1], dtype=dtypes.int64),
        dtype=dtypes.int64)
    with ops.device(counter.device):
      string_tensor = constant_op.constant(string_list,
                                           name="obtain_next_string_list")
  return obtain_next(string_tensor, counter)

