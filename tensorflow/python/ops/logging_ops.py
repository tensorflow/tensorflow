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

"""Logging and Summary Operations."""
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_logging_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_logging_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util.deprecation import deprecated

# The python wrapper for Assert is in control_flow_ops, as the Assert
# call relies on certain conditionals for its dependencies.  Use
# control_flow_ops.Assert.


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
  """Prints a list of tensors.

  This is an identity op with the side effect of printing `data` when
  evaluating.

  Note: This op prints to the standard error. It is not currently compatible
    with jupyter notebook (printing to the notebook *server's* output, not into
    the notebook).

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


def _Collect(val, collections, default_collections):
  if collections is None:
    collections = default_collections
  for key in collections:
    ops.add_to_collection(key, val)


def get_summary_op():
  """Returns a single Summary op that would run all summaries.

  Either existing one from `SUMMARY_OP` collection or merges all existing
  summaries.

  Returns:
    If no summaries were collected, returns None. Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.
  """
  summary_op = ops.get_collection(ops.GraphKeys.SUMMARY_OP)
  if summary_op is not None:
    if summary_op:
      summary_op = summary_op[0]
    else:
      summary_op = None
  if summary_op is None:
    summary_op = merge_all_summaries()
    if summary_op is not None:
      ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
  return summary_op


ops.NotDifferentiable("HistogramSummary")
ops.NotDifferentiable("ImageSummary")
ops.NotDifferentiable("AudioSummary")
ops.NotDifferentiable("AudioSummaryV2")
ops.NotDifferentiable("MergeSummary")
ops.NotDifferentiable("ScalarSummary")
