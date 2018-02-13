# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Define tflite op hints (intrinsic operations).

This essentially allows defining a TensorFlow API for tflite operations in
Python with hints on how they are represented in TensorFlow Lite. This basically
is a form of tflite intrinsic. It wraps a subpart of a TensorFlow execution
graph and is useful for LSTMs and other complicated TensorFlow constructions
that are difficult to pattern match in TOCO, but are represented by a single
accelerated tflite op.

Example:
  def tflite_cool_activation(input):
    # A cool activation function.
    custom = tf.contrib.lite.OpHint("cool_activation")
    input = custom.add_inputs(input)
    output = tf.sigmoid(input) * input
    custom.add_outputs(output)
    return output

  image = tf.placeholder(tf.float32, (1, 16, 16, 1))
  output = tf.identity(tflite_cool_activation(image))

  session = tf.Session()

  graphdef_to_convert = tf.contrib.lite.convert_op_hints_to_stubs(session)
  tflite_graph = tf.contrib.lite.toco_convert(graphdef_to_convert,
                                              [image], [output])
                                              [image], [output])
  with open("/tmp/graph.fb", "wb") as fp:
    fp.write(tflite_graph)

How does it work?:

OpHint is a helper that you use when defining a vanilla python function.
It allows you to wrap arguments with tf.identities with some custom attributes.
These attributes allow you to find the original block of ops that was created.
For example, if you use cool_activation above you essentially get:

a_input = tf.identity()
result = tf.multiply(tf.sigmoid(a_input), a_input)
output = tf.identity()

a_input, output are identities that have parameters representing
what argument they are, what the name of the function they should turn into
in tf lite as well as a guid that uniquely identifies a particular invocation.

Once you have built your whole tensorflow graph, you can run it and train it
as usual, but after you have done that, you need to convert the graph into
a form that replaces these subgraphs wrapped in identities to stub ops. These
ops don't actually exist in the normal TensorFlow runtime, but will be
understood by toco later.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as _collections
import itertools as _itertools
import uuid as _uuid

from tensorflow.contrib import framework as _framework
from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util.all_util import remove_undocumented


class OpHint(object):
  """A class that helps build tflite function invocations.

  It allows you to take a bunch of TensorFlow ops and annotate the construction
  such that toco knows how to convert it to tflite. This embeds a pseudo
  function in a TensorFlow graph. This allows embedding high-level API usage
  information in a lower level TensorFlow implementation so that an alternative
  implementation can be substituted later.

  Essentially, any "input" into this pseudo op is fed into an identity, and
  attributes are added to that input before being used by the constituent ops
  that make up the pseudo op. A similar process is done to any output that
  is to be exported from the current op.

  TODO(aselle): When TensorFlow functions functionality works for arbitrary
  constructs, this mechanism can be retired and changed to use python defun's.
  """

  # Attr constants that are used for representation in the GraphDef
  FUNCTION_NAME_ATTR = "_tflite_function_name"
  FUNCTION_UUID_ATTR = "_tflite_function_uuid"
  FUNCTION_INPUT_INDEX_ATTR = "_tflite_function_input_index"
  FUNCTION_OUTPUT_INDEX_ATTR = "_tflite_function_output_index"

  def __init__(self, function_name, **kwargs):
    """Create a OpHint.

    Args:
      function_name: Name of the function (the custom op name in tflite)
      **kwargs: Keyword arguments of any constant attributes for the function.
    """
    self._function_name = function_name
    self._unique_function_id = _uuid.uuid1().hex  # TODO(aselle): Unique enough?
    self._curr_input_index = 0
    self._curr_output_index = 0
    self._attrs_to_store_later = kwargs
    self._stored_attrs = False

  def _setattr(self, dest_op, name, value):
    tensor_value = _ops.convert_to_tensor(value)
    dest_op.op.node_def.attr[name].tensor.CopyFrom(
        tensor_value.op.node_def.attr["value"].tensor)

  def add_inputs(self, *args):
    """Add a sequence of inputs to the function invocation.

    Args:
      *args: List of inputs to be converted (should be Tf.Tensor).
    Returns:
      Wrapped inputs (identity standins that have additional metadata). These
      are also are also tf.Tensor's.
    """

    def augmented_identity(arg):
      identity_op = _array_ops.identity(arg)
      # pylint: disable=protected-access
      identity_op.op._set_attr(
          OpHint.FUNCTION_NAME_ATTR,
          _attr_value_pb2.AttrValue(s=self._function_name))
      identity_op.op._set_attr(
          OpHint.FUNCTION_UUID_ATTR,
          _attr_value_pb2.AttrValue(s=self._unique_function_id))
      identity_op.op._set_attr(
          OpHint.FUNCTION_INPUT_INDEX_ATTR,
          _attr_value_pb2.AttrValue(i=self._curr_input_index))
      # pylint: enable=protected-access
      self._curr_input_index += 1
      return identity_op

    return [augmented_identity(arg) for arg in args]

  def add_outputs(self, *args):
    """Add a sequence of outputs to the function invocation.

    Args:
      *args: List of outputs to be converted (should be tf.Tensor).
    Returns:
      Wrapped outputs (identity standins that have additional metadata). These
      are also tf.Tensor's.
    """

    def augmented_identity(arg):
      identity_op = _array_ops.identity(arg)
      # pylint: disable=protected-access
      identity_op.op._set_attr(
          OpHint.FUNCTION_NAME_ATTR,
          _attr_value_pb2.AttrValue(s=self._function_name))
      identity_op.op._set_attr(
          OpHint.FUNCTION_UUID_ATTR,
          _attr_value_pb2.AttrValue(s=self._unique_function_id))
      identity_op.op._set_attr(
          OpHint.FUNCTION_OUTPUT_INDEX_ATTR,
          _attr_value_pb2.AttrValue(i=self._curr_output_index))
      # pylint: enable=protected-access
      self._curr_output_index += 1
      return identity_op

    wrapped_outputs = [augmented_identity(arg) for arg in args]

    if not self._stored_attrs:
      for key, value in self._attrs_to_store_later.iteritems():
        self._setattr(wrapped_outputs[0], "_tflite_attr_" + key, value)
      self._stored_attrs = True

    return wrapped_outputs


class _LiteFuncCall(object):
  """Represent a TensorFlow Lite custom function.

  This is uses to accumulate found hints in the graphdef into a single
  conceptual unit.

  Properties:
    self.inputs: inputs to the op (hash from index # to argument)
    self.outputs: outputs to the op (hash from index # to argument)
    self.function_name: the tflite custom op name to use
    self.uuid: a unique call id for this particular call  (i.e.
      multiple function calls would have the same function_name but different
      uuids.
    self.params: A param name to key value for op constant data. I.e. for
      axis on a reduction, strides on a convolution, etc.
  """

  def __init__(self):
    self.inputs = {}
    self.outputs = {}
    self.function_name = None
    self.uuid = None
    self.params = {}

  def __str__(self):
    return "tflite function %s call %s\n\tinputs: %r\n\toutputs: %r" % (
        self.function_name, self.uuid, self.inputs, self.outputs)


def _find_all_hints_in_graph_def(session):
  """Look at the current default graph and return a list of LiteFuncCall objs.

  Args:
    session: A TensorFlow session that contains the graph to convert.
  Returns:
    a list of `LifeFuncCall` objects in the form

  """
  func_calls = _collections.defaultdict(_LiteFuncCall)
  seen_ops = set()

  for op in session.graph.get_operations():
    for operand in _itertools.chain(op.inputs, op.outputs):
      if operand in seen_ops:
        continue
      seen_ops.add(operand)
      attr = operand.op.node_def.attr
      uuid = attr[OpHint.FUNCTION_UUID_ATTR].s
      if OpHint.FUNCTION_UUID_ATTR not in attr:
        continue
      call_def = func_calls[uuid]
      call_def.uuid = uuid
      if OpHint.FUNCTION_UUID_ATTR in attr:
        call_def.function_name = attr[OpHint.FUNCTION_NAME_ATTR].s
        if OpHint.FUNCTION_INPUT_INDEX_ATTR in attr:
          call_def.inputs[attr[OpHint.FUNCTION_INPUT_INDEX_ATTR].i] = operand
        if OpHint.FUNCTION_OUTPUT_INDEX_ATTR in attr:
          call_def.outputs[attr[OpHint.FUNCTION_OUTPUT_INDEX_ATTR].i] = operand

      for a in attr:
        if a.startswith("_tflite_attr_"):
          # TODO(aselle): Remember the attribute tensors so we can put them
          # in collapse.
          call_def.params[a.replace("_tflite_attr_,", "")] = attr[a].tensor

  return func_calls


def _tensor_name_base(full_tensor_name):
  """Removes the device assignment code from a tensor.

  e.g. _tensor_name_base("foo:3") => "foo"

  Args:
    full_tensor_name: A tensor name that is annotated with a device placement
      (this is what tensor flow introspection gives).
  Returns:
    A name without any device assignment.
  """
  return full_tensor_name.name.split(":")[0]


def convert_op_hints_to_stubs(session):
  """Converts a graphdef with LiteOp hints into stub operations.

  This is used to prepare for toco conversion of complex intrinsic usages.

  Args:
    session: A TensorFlow session that contains the graph to convert.
  Returns:
    A new graphdef with all ops contained in OpHints being replaced by
    a single op call with the right parameters.
  """
  hints = _find_all_hints_in_graph_def(session)
  current_graph_def = session.graph_def
  for call in hints.values():
    input_names = [None] * len(call.inputs)
    output_names = [None] * len(call.outputs)
    output_dtypes = [None] * len(call.outputs)
    output_quantized = False
    for input_index, tensor in call.inputs.items():
      input_names[input_index] = _tensor_name_base(tensor)
    for output_index, tensor in call.outputs.items():
      output_names[output_index] = _tensor_name_base(tensor)
      output_dtypes[output_index] = tensor.dtype.as_datatype_enum
    # TODO(aselle): Support quantized flag properly
    current_graph_def = _framework.fuse_op(
        current_graph_def, input_names, output_names, output_dtypes,
        output_quantized, call.uuid, call.function_name)
    for node in current_graph_def.node:
      if node.name == call.uuid:
        for param, tensor in call.params.items():
          node.attr[param].tensor.CopyFrom(tensor)
  return current_graph_def


_allowed_symbols = ["OpHint", "convert_op_hints_to_stubs"]
remove_undocumented(__name__, _allowed_symbols)
