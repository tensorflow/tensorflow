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

import random

from tensorflow.contrib.input_pipeline.ops import gen_input_pipeline_ops
from tensorflow.contrib.util import loader
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
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
  return gen_input_pipeline_ops.obtain_next(string_list_tensor, counter)


def _maybe_randomize_list(string_list, shuffle):
  if shuffle:
    random.shuffle(string_list)
  return string_list


def _create_list(string_list, shuffle, seed, num_epochs):
  if shuffle and seed:
    random.seed(seed)
  expanded_list = _maybe_randomize_list(string_list, shuffle)[:]
  if num_epochs:
    for _ in range(num_epochs - 1):
      expanded_list.extend(_maybe_randomize_list(string_list, shuffle))
  return expanded_list


def seek_next(string_list, shuffle=False, seed=None, num_epochs=None):
  """Returns an op that seeks the next element in a list of strings.

  Seeking happens in a round robin fashion. This op creates a variable called
  obtain_next_counter that is initialized to -1 and is used to keep track of
  which element in the list was returned, and a variable
  obtain_next_expanded_list to hold the list. If num_epochs is not None, then we
  limit the number of times we go around the string_list before OutOfRangeError
  is thrown. It creates a variable to keep track of this.

  Args:
    string_list: A list of strings.
    shuffle: If true, we shuffle the string_list differently for each epoch.
    seed: Seed used for shuffling.
    num_epochs: Returns OutOfRangeError once string_list has been repeated
                num_epoch times. If unspecified then keeps on looping.

  Returns:
    An op that produces the next element in the provided list.
  """
  expanded_list = _create_list(string_list, shuffle, seed, num_epochs)

  with variable_scope.variable_scope("obtain_next"):
    counter = variable_scope.get_variable(
        name="obtain_next_counter",
        initializer=constant_op.constant(
            -1, dtype=dtypes.int64),
        dtype=dtypes.int64,
        trainable=False)
    with ops.colocate_with(counter):
      string_tensor = variable_scope.get_variable(
          name="obtain_next_expanded_list",
          initializer=constant_op.constant(expanded_list),
          dtype=dtypes.string,
          trainable=False)
    if num_epochs:
      filename_counter = variable_scope.get_variable(
          name="obtain_next_filename_counter",
          initializer=constant_op.constant(
              0, dtype=dtypes.int64),
          dtype=dtypes.int64,
          trainable=False)
      c = filename_counter.count_up_to(len(expanded_list))
      with ops.control_dependencies([c]):
        return obtain_next(string_tensor, counter)
    else:
      return obtain_next(string_tensor, counter)
