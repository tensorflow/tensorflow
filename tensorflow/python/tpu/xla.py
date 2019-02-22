# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""XLA utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.util import tf_inspect


def is_flat(outputs):
  """Checks if outputs is a flat structure.

    Following structures and values are considered flat:
    1) None
    2) A single object
    3) A list or tuple of Tensors/Operations

    The only structures that this function understands are sequences and
    dictionaries.  E.g. this means that if outputs contains a single
    user-defined Object, it is considered to be flat. Errors are raised later on
    if that Object cannot be converted to a Tensor.

  Args:
    outputs: Output from `computation` inside `xla.compile`.

  Returns:
    A boolean indicates whether outputs is flat.
  """
  # If outputs is a list or tuple, check if it has any nested structure. If
  # there is, then outputs is non-flat.
  if isinstance(outputs, collections.Sequence):
    for o in outputs:
      if isinstance(o, collections.Sequence) or isinstance(o, dict):
        return False

  # If outputs is a dict, it is non-flat.
  if isinstance(outputs, dict):
    return False

  # Getting here means either outputs itself is a single non-structured value
  # or it is a flat list of single non-structured values.
  return True


def check_function_argument_count(func, input_arity, infeed_queue):
  """Validate the number of input arguments to an XLA function.

  Args:
    func: the Python function that will be called to generate the body of an XLA
      computation graph.
    input_arity: the number of explicit arguments supplied by the caller.
    infeed_queue: if not None, the infeed queue that will supply
      additional arguments to the function.

  Returns:
    None if function can be called with the supplied number of
      arguments, or an error string if it cannot.
  """
  def format_error(complaint, quantity):
    return '%s %d argument%s' % (complaint, quantity, ''
                                 if quantity == 1 else 's')

  num_args_supplied = input_arity
  if infeed_queue is not None:
    num_args_supplied += infeed_queue.number_of_tuple_elements
  arg_spec = tf_inspect.getargspec(func)
  num_func_args = len(arg_spec.args)
  if arg_spec.defaults is None:
    num_func_defaults = 0
  else:
    num_func_defaults = len(arg_spec.defaults)
  min_func_args = num_func_args - num_func_defaults
  if num_args_supplied < min_func_args:
    # The required number of arguments is not enough to call the function.
    if num_func_defaults == 0 and arg_spec.varargs is None:
      return format_error('exactly', num_func_args)
    else:
      return format_error('at least', min_func_args)
  if arg_spec.varargs is None and num_args_supplied > num_func_args:
    # The required number of arguments is too many to call the function.
    if num_func_defaults == 0:
      return format_error('exactly', num_func_args)
    else:
      return format_error('at most', num_func_args)
  # Reaching here means either
  # 1) There are varargs, func can accept any number of arguments greater than
  # the minimum.
  # 2) Number of supplied arguments falls in range of acceptable argument count
  # of func.
  return None
