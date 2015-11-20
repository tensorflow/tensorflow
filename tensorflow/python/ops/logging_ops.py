# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Logging Operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_logging_ops
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
def Assert(condition, data, summarize=None, name=None):
  """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).
  """
  return gen_logging_ops._assert(condition, data, summarize, name)


def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Args:
    input_: A tensor passed through this op.
    data: A list of tensors to print out when op is evaluated.
    message: A string, prefix of the error message.
    first_n: Only log `first_n` number of times. Negative numbers log always;
             this is the default.
    summarize: Only print this many entries of each tensor. If None, then a
               maximum of 3 elements are printed per input tensor.
    name: A name for the operation (optional).

  Returns:
    Same tensor as `input_`.
  """
  return gen_logging_ops._print(input_, data, message, first_n, summarize, name)


@ops.RegisterGradient("Print")
def _PrintGrad(op, *grad):
  return list(grad) + [None] * (len(op.inputs) - 1)


ops.RegisterShape("Assert")(common_shapes.no_outputs)
ops.RegisterShape("Print")(common_shapes.unchanged_shape)
