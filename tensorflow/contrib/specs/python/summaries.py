# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for summarizing and describing TensorFlow graphs.

This contains functions that generate string descriptions from
TensorFlow graphs, for debugging, testing, and model size
estimation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.contrib.specs.python import specs
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

# These are short abbreviations for common TensorFlow operations used
# in test cases with tf_structure to verify that specs_lib generates a
# graph structure with the right operations. Operations outside the
# scope of specs (e.g., Const and Placeholder) are just assigned "_"
# since they are not relevant to testing.

SHORT_NAMES_SRC = """
BiasAdd biasadd
Const _
Conv2D conv
MatMul dot
Placeholder _
Sigmoid sig
Variable var
""".split()

SHORT_NAMES = {
    x: y
    for x, y in zip(SHORT_NAMES_SRC[::2], SHORT_NAMES_SRC[1::2])
}


def _truncate_structure(x):
  """A helper function that disables recursion in tf_structure.

  Some constructs (e.g., HorizontalLstm) are complex unrolled
  structures and don't need to be represented in the output
  of tf_structure or tf_print. This helper function defines
  which tree branches should be pruned. This is a very imperfect
  way of dealing with unrolled LSTM's (since it truncates
  useful information as well), but it's not worth doing something
  better until the new fused and unrolled ops are ready.

  Args:
      x: a Tensor or Op

  Returns:
      A bool indicating whether the subtree should be pruned.
  """
  if "/HorizontalLstm/" in x.name:
    return True
  return False


def tf_structure(x, include_shapes=False, finished=None):
  """A postfix expression summarizing the TF graph.

  This is intended to be used as part of test cases to
  check for gross differences in the structure of the graph.
  The resulting string is not invertible or unabiguous
  and cannot be used to reconstruct the graph accurately.

  Args:
      x: a tf.Tensor or tf.Operation
      include_shapes: include shapes in the output string
      finished: a set of ops that have already been output

  Returns:
      A string representing the structure as a string of
      postfix operations.
  """
  if finished is None:
    finished = set()
  if isinstance(x, ops.Tensor):
    shape = x.get_shape().as_list()
    x = x.op
  else:
    shape = []
  if x in finished:
    return " <>"
  finished |= {x}
  result = ""
  if not _truncate_structure(x):
    for y in x.inputs:
      result += tf_structure(y, include_shapes, finished)
  if include_shapes:
    result += " %s" % (shape,)
  if x.type != "Identity":
    name = SHORT_NAMES.get(x.type, x.type.lower())
    result += " " + name
  return result


def tf_print(x, depth=0, finished=None, printer=print):
  """A simple print function for a TensorFlow graph.

  Args:
      x: a tf.Tensor or tf.Operation
      depth: current printing depth
      finished: set of nodes already output
      printer: print function to use

  Returns:
      Total number of parameters found in the
      subtree.
  """

  if finished is None:
    finished = set()
  if isinstance(x, ops.Tensor):
    shape = x.get_shape().as_list()
    x = x.op
  else:
    shape = ""
  if x.type == "Identity":
    x = x.inputs[0].op
  if x in finished:
    printer("%s<%s> %s %s" % ("  " * depth, x.name, x.type, shape))
    return
  finished |= {x}
  printer("%s%s %s %s" % ("  " * depth, x.name, x.type, shape))
  if not _truncate_structure(x):
    for y in x.inputs:
      tf_print(y, depth + 1, finished, printer=printer)


def tf_num_params(x):
  """Number of parameters in a TensorFlow subgraph.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Returns:
      Total number of elements found in all Variables
      in the subgraph.
  """

  if isinstance(x, ops.Tensor):
    shape = x.get_shape()
    x = x.op
  if x.type in ["Variable", "VariableV2"]:
    return shape.num_elements()
  totals = [tf_num_params(y) for y in x.inputs]
  return sum(totals)


def tf_left_split(op):
  """Split the parameters of op for left recursion.

  Args:
    op: tf.Operation

  Returns:
    A tuple of the leftmost input tensor and a list of the
    remaining arguments.
  """

  if len(op.inputs) < 1:
    return None, []
  if op.type == "Concat":
    return op.inputs[1], op.inputs[2:]
  return op.inputs[0], op.inputs[1:]


def tf_parameter_iter(x):
  """Iterate over the left branches of a graph and yield sizes.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Yields:
      A triple of name, number of params, and shape.
  """

  while 1:
    if isinstance(x, ops.Tensor):
      shape = x.get_shape().as_list()
      x = x.op
    else:
      shape = ""
    left, right = tf_left_split(x)
    totals = [tf_num_params(y) for y in right]
    total = sum(totals)
    yield x.name, total, shape
    if left is None:
      break
    x = left


def _combine_filter(x):
  """A filter for combining successive layers with similar names."""
  last_name = None
  last_total = 0
  last_shape = None
  for name, total, shape in x:
    name = re.sub("/.*", "", name)
    if name == last_name:
      last_total += total
      continue
    if last_name is not None:
      yield last_name, last_total, last_shape
    last_name = name
    last_total = total
    last_shape = shape
  if last_name is not None:
    yield last_name, last_total, last_shape


def tf_parameter_summary(x, printer=print, combine=True):
  """Summarize parameters by depth.

  Args:
      x: root of the subgraph (Tensor, Operation)
      printer: print function for output
      combine: combine layers by top-level scope
  """
  seq = tf_parameter_iter(x)
  if combine:
    seq = _combine_filter(seq)
  seq = reversed(list(seq))
  for name, total, shape in seq:
    printer("%10d %-20s %s" % (total, name, shape))


def tf_spec_structure(spec,
                      inputs=None,
                      input_shape=None,
                      input_type=dtypes.float32):
  """Return a postfix representation of the specification.

  This is intended to be used as part of test cases to
  check for gross differences in the structure of the graph.
  The resulting string is not invertible or unabiguous
  and cannot be used to reconstruct the graph accurately.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: tensor shape (in lieu of inputs)
      input_type: type of the input tensor

  Returns:
      A string with a postfix representation of the
      specification.
  """

  if inputs is None:
    inputs = array_ops.placeholder(input_type, input_shape)
  outputs = specs.create_net(spec, inputs)
  return str(tf_structure(outputs).strip())


def tf_spec_summary(spec,
                    inputs=None,
                    input_shape=None,
                    input_type=dtypes.float32):
  """Output a summary of the specification.

  This prints a list of left-most tensor operations and summarized the
  variables found in the right branches. This kind of representation
  is particularly useful for networks that are generally structured
  like pipelines.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: optional shape of input
      input_type: type of the input tensor
  """

  if inputs is None:
    inputs = array_ops.placeholder(input_type, input_shape)
  outputs = specs.create_net(spec, inputs)
  tf_parameter_summary(outputs)


def tf_spec_print(spec,
                  inputs=None,
                  input_shape=None,
                  input_type=dtypes.float32):
  """Print a tree representing the spec.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: optional shape of input
      input_type: type of the input tensor
  """

  if inputs is None:
    inputs = array_ops.placeholder(input_type, input_shape)
  outputs = specs.create_net(spec, inputs)
  tf_print(outputs)
