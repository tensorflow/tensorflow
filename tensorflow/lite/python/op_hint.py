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
    custom = tf.lite.OpHint("cool_activation")
    input, = custom.add_inputs(input)
    output = tf.sigmoid(input) * input
    output, = custom.add_outputs(output)
    return output

  image = tf.compat.v1.placeholder(tf.float32, (1, 16, 16, 1))
  output = tf.identity(tflite_cool_activation(image))

  session = tf.compat.v1.Session()

  graphdef_to_convert = tf.lite.convert_op_hints_to_stubs(session)
  tflite_graph = tf.compat.v1.lite.toco_convert(
      graphdef_to_convert, [image], [output])
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

# TODO(aselle): Make this use generic graph transformations.
# TODO(aselle): _tensor_name_base should be called _tensor_name_to_op_name.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections as _collections
import copy as _copy
import json as _json
import uuid as _uuid
import six as _six

from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.core.framework import node_def_pb2 as _node_def_pb2
from tensorflow.python.framework import ops as _ops
# TODO(aselle): publicize these apis if we continue to use these.
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.util import compat as _compat
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow.python.util.tf_export import tf_export as _tf_export


@_tf_export(v1=["lite.OpHint"])
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

  """
  # TODO(aselle): When TensorFlow functions functionality works for arbitrary
  # constructs, this mechanism can be retired and changed to use python defun's.

  # Attr constants that are used for representation in the GraphDef. These
  # will be used on every Identity op that is involved in a total OpHint.

  # Name of the OpHint function (cosmetic).
  FUNCTION_NAME_ATTR = "_tflite_function_name"
  # UUID of the function (each OpHint gets a new uuid).
  FUNCTION_UUID_ATTR = "_tflite_function_uuid"
  # The input index of the input (or nothing if it is an output).
  FUNCTION_INPUT_INDEX_ATTR = "_tflite_function_input_index"
  # The output index of the output (or nothing if it is an input).
  FUNCTION_OUTPUT_INDEX_ATTR = "_tflite_function_output_index"
  # An index that orders aggregate arguments. Aggregate arguments are ones
  # that are separate but will be fused horizontally. For example a static LSTM
  # has a lstm cell for each time step. Each one has a separate opHint, but a
  # fused SequentialLSTM will treat this as a single tensor.
  FUNCTION_SORT_INDEX_ATTR = "_tflite_function_sort_index"
  # The way in which multiple parts of the aggregate argument will be joined
  # into a fused operand. Valid options are OpHint.AGGREGATE_FIRST,
  # OpHint.AGGREGATE_LAST, OpHint.AGGREGATE_STACK.
  FUNCTION_AGGREGATE_ATTR = "_tflite_function_aggregate"
  # On fused OpHint stub, the order of inputs that the final LSTM call will
  # have. What this means is that the TensorFlow order might be
  # "foo", "bar", "stuff" and you might want the TF lite op order to be
  # "stuff", "foo", "bar", -1 (where -1 is unused). So you would set this
  # attribute to [2, 0, 1, -1].
  TFLITE_INPUT_INDICES = "_tflite_input_indices"
  # OpHint level.
  FUNCTION_LEVEL_ATTR = "_tflite_ophint_level"
  # Ophint internal mapping, this is for high level Ophint only.
  # This basically contains three kinds of mapping:
  #   1) How parental ophinted inputs map to the first child ophinted inputs;
  #   2) How internal children nodes are connected;
  #   3) How parental ophinted outputs map to the last child ophinted outputs.
  CHILDREN_INPUTS_MAPPINGS = "_tflite_children_ophint_inputs_mapping"

  # Types of aggregations
  #  stack: stacks all ophints with matching tags. i.e. for a static rnn.
  #   specifically, this is good for an input or output to a static rnn cell.
  AGGREGATE_STACK = "stack"
  # first: only takes the first output (one with lowest sort index)
  # of matching tags. This is good for the input state to an RNN.
  AGGREGATE_FIRST = "first"
  # aggregation last takes only the last tag (one with highest sort index).
  # This is good for an output value on the last stack item of a
  # static rnn.
  AGGREGATE_LAST = "last"

  class OpHintArgumentTracker(object):
    """Conceptually tracks indices of arguments of "OpHint functions".

    The inputs and arguments of these functions both use an instance
    of the class so they can have independent numbering.
    """

    def __init__(self,
                 function_name,
                 unique_function_id,
                 node_name_prefix,
                 attr_name,
                 level=1,
                 children_inputs_mappings=None):
      """Initialize ophint argument.

      Args:
        function_name: Name of the function that this tracks arguments for.
        unique_function_id: UUID of function that this tracks arguments for.
        node_name_prefix: How identities that are created are named.
        attr_name: Name of attribute to use to store the index for this hint.
          i.e. FUNCTION_INPUT_INDEX or FUNCTION_OUTPUT_INDEX
        level: Hierarchical level of the Ophint node, a number.
        children_inputs_mappings: Inputs/Outputs mapping for children hints.
      """

      # The global index is the argument index of the op. This is in contrast
      # to the sort index which is the sequence number of a particular instance
      # of a given global index. For example, you may have called add hint
      # twice with the tag "foo". Then the global index will be 0 for both
      # and the sort index will be 0 for the first added and 1 for the second.
      self._function_name = function_name
      self._unique_function_id = unique_function_id
      self._next_global_index = 0  # The absolute global index
      self._used_global_indices = set()
      self._tag_to_global_index = {}  # The argument index a given tag maps to
      self._tag_to_next_sort_index = {}  # The current index for each tag
      self._node_name_prefix = node_name_prefix
      self._attr_name = attr_name
      self._level = level
      self._children_inputs_mappings = children_inputs_mappings

    def _get_new_global_index(self, index_override):
      """Return the next unused argument index in order or use an override.

      Args:
        index_override: An index to use instead of the next available or None
          to use the next available.

      Returns:
        A valid global_index to use for the next hint argument.

      Raises:
        ValueError: If the index_override is already used by another hint.
      """
      if index_override is None:
        global_index = self._next_global_index
      else:
        if index_override in self._used_global_indices:
          raise ValueError("Index %d was already used by another call to add")
        global_index = index_override
      # Make next_global_index valid
      self._used_global_indices.add(global_index)
      while self._next_global_index in self._used_global_indices:
        self._next_global_index += 1
      return global_index

    def add(self, arg, tag=None, name=None, aggregate=None,
            index_override=None):
      """Return a wrapped tensor of an input tensor as an argument.

      Args:
        arg: A TensorFlow tensor that should be considered an argument.
        tag: String tag to identify arguments that should be packed.
        name: Name of argument. This is included in the Identity hint op names.
        aggregate: Strategy to aggregate.
        Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
          Note, aggregate is only valid if tag is specified.
        index_override: Specify what input/output index should this be in the
          final stub. i.e. add(arg0, index=1); add(arg1, index=0) will make the
          final stub be as stub_func(inputs[arg1, arg0], outputs=[]) rather than
          the default call order based ordering.

      Returns:
        A tensor representing the wrapped argument.

      Raises:
        ValueError: When indices are not consistent.
      """

      # Find the appropriate index
      if tag is None:
        if aggregate is not None:
          raise ValueError("You must specify `tag` if using aggregate.")
        global_index = self._get_new_global_index(index_override)
        sort_index = None
      else:
        if aggregate is None:
          raise ValueError("You must specify `aggregate` if using tag.")
        if tag not in self._tag_to_global_index:
          self._tag_to_global_index[tag] = (
              self._get_new_global_index(index_override))
          self._tag_to_next_sort_index[tag] = 0
        elif (index_override and
              index_override != self._tag_to_global_index[tag]):
          raise ValueError(
              "Tag %r was called with two indices %r and %r" %
              (tag, index_override, self._tag_to_global_index[tag]))
        global_index = self._tag_to_global_index[tag]
        sort_index = self._tag_to_next_sort_index[tag]
        self._tag_to_next_sort_index[tag] += 1

      uuid = self._unique_function_id
      name = "%s-%s-%s-%r-%r-%s" % (self._node_name_prefix, self._function_name,
                                    uuid, global_index, sort_index, name)

      identity_op = _array_ops.identity(arg, name=name)

      # pylint: disable=protected-access
      identity_op.op._set_attr(
          OpHint.FUNCTION_NAME_ATTR,
          _attr_value_pb2.AttrValue(
              s=_compat.as_bytes(self._function_name)))
      identity_op.op._set_attr(
          OpHint.FUNCTION_UUID_ATTR,
          _attr_value_pb2.AttrValue(
              s=_compat.as_bytes(self._unique_function_id)))
      identity_op.op._set_attr(
          self._attr_name, _attr_value_pb2.AttrValue(i=global_index))
      identity_op.op._set_attr(OpHint.FUNCTION_LEVEL_ATTR,
                               _attr_value_pb2.AttrValue(i=self._level))
      if self._children_inputs_mappings:
        identity_op.op._set_attr(
            OpHint.CHILDREN_INPUTS_MAPPINGS,
            _attr_value_pb2.AttrValue(
                s=_compat.as_bytes(_json.dumps(
                    self._children_inputs_mappings))))

      if sort_index is not None:
        identity_op.op._set_attr(
            OpHint.FUNCTION_SORT_INDEX_ATTR,
            _attr_value_pb2.AttrValue(i=sort_index))
      if aggregate is not None:
        identity_op.op._set_attr(
            OpHint.FUNCTION_AGGREGATE_ATTR,
            _attr_value_pb2.AttrValue(s=_compat.as_bytes((aggregate))))
      # pylint: enable=protected-access
      return identity_op

  def __init__(self,
               function_name,
               level=1,
               children_inputs_mappings=None,
               **kwargs):
    """Create a OpHint.

    Args:
      function_name: Name of the function (the custom op name in tflite)
      level: OpHint level.
      children_inputs_mappings: Children OpHint inputs/outputs mapping.
        children_inputs_mappings should like below:
        "parent_first_child_input":
            [{"parent_input_index": num, "child_input_index": num}, ...]
        "parent_last_child_output":
            [{"parent_output_index": num, "child_output_index": num}, ...]
        "internal_children_input_output":
            [{"child_input_index": num, "child_output_index": num}, ...]
      **kwargs: Keyword arguments of any constant attributes for the function.
    """
    self._function_name = function_name
    self._level = level
    if self._level == 1:
      assert children_inputs_mappings is None
    else:
      assert isinstance(children_inputs_mappings, dict)
    self._children_inputs_mappings = children_inputs_mappings
    if self._children_inputs_mappings is not None:
      self._validate_children_inputs_mappings(self._children_inputs_mappings)
    self._unique_function_id = _uuid.uuid1().hex  # TODO(aselle): Unique enough?
    self._attrs_to_store_later = kwargs
    self._stored_attrs = False
    self._inputs = OpHint.OpHintArgumentTracker(
        self._function_name, self._unique_function_id, "InputHint",
        OpHint.FUNCTION_INPUT_INDEX_ATTR, level, self._children_inputs_mappings)
    self._outputs = OpHint.OpHintArgumentTracker(
        self._function_name, self._unique_function_id, "OutputHint",
        OpHint.FUNCTION_OUTPUT_INDEX_ATTR, level,
        self._children_inputs_mappings)

  def _validate_children_inputs_mappings(self, children_inputs_mappings):
    """Validate children inputs mappings is in the right format.

    Args:
      children_inputs_mappings: the Children ophint inputs/outputs mapping.
    """
    assert isinstance(children_inputs_mappings, dict)
    assert "parent_first_child_input" in children_inputs_mappings
    assert "parent_last_child_output" in children_inputs_mappings
    assert "internal_children_input_output" in children_inputs_mappings

    # validate parent_first_child_input.

    def assert_dictlist_has_keys(dictlist, keys):
      for dikt in dictlist:
        assert isinstance(dikt, dict)
        for key in keys:
          assert key in dikt

    assert_dictlist_has_keys(
        children_inputs_mappings["parent_first_child_input"],
        ["parent_ophint_input_index", "first_child_ophint_input_index"])
    assert_dictlist_has_keys(
        children_inputs_mappings["parent_last_child_output"],
        ["parent_output_index", "child_output_index"])
    assert_dictlist_has_keys(
        children_inputs_mappings["internal_children_input_output"],
        ["child_input_index", "child_output_index"])

  def _setattr(self, dest_op, name, value):
    tensor_value = _ops.convert_to_tensor(value)
    # pylint: disable=protected-access
    dest_op.op._set_attr(name, _attr_value_pb2.AttrValue(
        tensor=tensor_value.op.node_def.attr["value"].tensor))
    # pylint: enable=protected-access

  def add_input(self, *args, **kwargs):
    """Add a wrapped input argument to the hint.

    Args:
      *args: The input tensor.
      **kwargs:
        "name" label
        "tag" a tag to group multiple arguments that will be aggregated. I.e.
          a string like 'cool_input'. Basically multiple inputs can be added
          to the same hint for parallel operations that will eventually be
          combined. An example would be static_rnn which creates multiple copies
          of state or inputs.
        "aggregate" aggregation strategy that is valid only for tag non None.
          Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
        "index_override" The global index to use. This corresponds to the
          argument order in the final stub that will be generated.
    Returns:
      The wrapped input tensor.
    """
    return self._inputs.add(*args, **kwargs)

  def add_output(self, *args, **kwargs):
    """Add a wrapped output argument to the hint.

    Args:
      *args: The output tensor.
      **kwargs:
        "name" label
        "tag" a tag to group multiple arguments that will be aggregated. I.e.
          a string like 'cool_input'. Basically multiple inputs can be added
          to the same hint for parallel operations that will eventually be
          combined. An example would be static_rnn which creates multiple copies
          of state or inputs.
        "aggregate" aggregation strategy that is valid only for tag non None.
          Acceptable values are OpHint.AGGREGATE_FIRST, OpHint.AGGREGATE_LAST,
          and OpHint.AGGREGATE_STACK.
        "index_override" The global index to use. This corresponds to the
          argument order in the final stub that will be generated.
    Returns:
      The wrapped output tensor.
    """
    return self._outputs.add(*args, **kwargs)

  def add_inputs(self, *args, **kwargs):
    """Add a sequence of inputs to the function invocation.

    Args:
      *args: List of inputs to be converted (should be Tf.Tensor).
      **kwargs: This allows 'names' which should be a list of names.
    Returns:
      Wrapped inputs (identity standins that have additional metadata). These
      are also are also tf.Tensor's.
    """
    if "names" in kwargs:
      return [
          self._inputs.add(arg, name=name)
          for arg, name in zip(args, kwargs["names"])
      ]
    else:
      return [self._inputs.add(arg) for arg in args]

  def add_outputs(self, *args, **kwargs):
    """Add a sequence of outputs to the function invocation.

    Args:
      *args: List of outputs to be converted (should be tf.Tensor).
      **kwargs: See
    Returns:
      Wrapped outputs (identity standins that have additional metadata). These
      are also tf.Tensor's.
    """
    if "names" in kwargs:
      return [
          self._outputs.add(arg, name=name)
          for arg, name in zip(args, kwargs["names"])
      ]
    else:
      return [self._outputs.add(arg) for arg in args]


class _LiteOperand(object):
  """Abstract operand for a tflite hint function._dynamic_rnn_loop.

  This is a base class that handles representing arguments to an OpHint.
  It also is able to serialize operands to the stubbed graph_def.
  Child classes are responsible for being able to
  store information about the hint identity operators. They are also responsible
  for knowing how to serialize to output graphdefs.

  Typically this will be implemented by holding one or more identity nodes
  that were previously discovered as hints.
  """

  def aggregate_and_return_name_for_input(self, out_graphdef):
    """This adds the node(s) to out_graphdef and returns the input node name.

    Args:
      out_graphdef: A graphdef that is ready to have this input added.

    Returns:
      The output that the stub should use as an input for this operand.

    Raises:
      RuntimeError: if the method is not implemented.
    """
    del out_graphdef
    raise RuntimeError("Unimplemented abstract method.")

  def aggregate_and_return_name_for_output(self, fused_op_name, output_index,
                                           out_graphdef):
    """Add node(s) to graph representing output operands and returns type.

    Args:
      fused_op_name: name of the fused op stub name.
      output_index: Output index that we are currently processing from stub.
      out_graphdef: The destination graphdef we are currently building up.

    Returns:
      The datatype of this identity.

    Raises:
      RuntimeError: if the method is not implemented.
    """
    del fused_op_name, output_index, out_graphdef
    raise RuntimeError("Unimplemented abstract method.")


class _LiteSingleOperand(_LiteOperand):
  """A simple operand that is non-aggregated (i.e. most hints)."""

  def __init__(self, node):
    _LiteOperand.__init__(self)
    self.node = node
    self.name = _tensor_name_base(node.name)

  def flatten(self):
    return [self.name]

  def aggregate_and_return_name_for_input(self, out_graphdef):
    return self.name

  def aggregate_and_return_name_for_output(self, fused_op_name, index,
                                           out_graphdef):
    output_node = _copy.deepcopy(self.node)
    del output_node.input[:]
    output_node.input.append(_tensorflow_output_name(fused_op_name, index))
    out_graphdef.node.extend([output_node])
    return self.node.attr["type"].i

  def __str__(self):
    return str(self.name)


class _LiteAggregateOperand(_LiteOperand):
  """An operand for a tflite hint function that is aggregated from many.

  For example, an LSTM is a grid of operators that are all related. Inputs
  going into them may need to be fused, so they should all be tracked as
  related arguments.
  """

  def __init__(self, aggregation):
    _LiteOperand.__init__(self)
    self.aggregation = aggregation
    self.names = {}
    self.nodes = {}
    self.flattened = None

  def add(self, sort, node):
    self.names[sort] = _tensor_name_base(node.name)
    self.nodes[sort] = node

  def flatten_nodes(self):
    """Return a list of all the node protos in aggregation sorted order."""
    if not self.flattened:
      self.flattened = [None] * len(self.nodes)
      for idx, node in _six.iteritems(self.nodes):
        self.flattened[idx] = node
      for n in self.nodes:
        if n is None:
          raise RuntimeError("Aggregate was missing argument.")
      if self.aggregation == OpHint.AGGREGATE_FIRST:
        self.flattened = self.flattened[:1]
      elif self.aggregation == OpHint.AGGREGATE_LAST:
        self.flattened = self.flattened[-1:]
      elif self.aggregation == OpHint.AGGREGATE_STACK:
        pass
      else:
        raise ValueError(
            "Invalid aggregation type %r specified" % self.aggregation)
    return self.flattened

  def flatten(self):
    """Return a list of all node names in aggregation sorted sorter."""
    return [_tensor_name_base(x.name) for x in self.flatten_nodes()]

  def aggregate_and_return_name_for_input(self, out_graphdef):
    """This adds the nodes to out_graphdef and returns an aggregated output.

    In particular, if you have 4 inputs to a hint stub, this will be the
    node that you can use as an output. I.e. you have 4 timesteps from a
    static rnn, then a fused UnidriecitonalLSTM will expect 1 input with
    all 4 time steps. So here we make a pack and return the output name of
    that pack.

    Args:
      out_graphdef: A graphdef that is ready to have this input added.

    Returns:
      The name of a pack that aggregates this node.
    """
    flattened = self.flatten_nodes()
    if (self.aggregation == OpHint.AGGREGATE_FIRST) or (
        self.aggregation == OpHint.AGGREGATE_LAST):
      assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
      return _tensor_name_base(flattened[0].name)
    else:
      new_node = _node_def_pb2.NodeDef()
      new_node.op = "Pack"
      new_node.name = "OpHintStack-%s" % flattened[0].name
      new_node.attr["N"].i = len(flattened)
      new_node.attr["T"].type = flattened[0].attr["T"].type
      for discrete in flattened:
        new_node.input.append(_tensor_name_base(discrete.name))
      out_graphdef.node.extend([new_node])
      return new_node.name

  def aggregate_and_return_name_for_output(self, fused_op_name, output_index,
                                           out_graphdef):
    """This adds to `out_graphdef` all the unaggregated outputs.

    I.e. we are outputting from a fused stub, but we need to make it compatible
    with the unfused original graph so we insert an unpack. Ideally in a later
    stage the unpack -> pack sequences will be removed.

    Args:
      fused_op_name: The name of the stub we are in the process of fusing.
      output_index: The output output_index this object represents.
      out_graphdef: The graphdef we are in the process of buildings

    Returns:
      The type of the aggregated output (so we can finish building the stub
      op).
    """
    flattened = self.flatten_nodes()
    if (self.aggregation == OpHint.AGGREGATE_FIRST) or (
        self.aggregation == OpHint.AGGREGATE_LAST):
      assert len(flattened) == 1
    if len(flattened) == 1 and self.aggregation != OpHint.AGGREGATE_STACK:
      temp_op = _LiteSingleOperand(flattened[0])
      return temp_op.aggregate_and_return_name_for_output(
          fused_op_name, output_index, out_graphdef)
    else:
      stack_node = _node_def_pb2.NodeDef()
      stack_node.op = "Unpack"
      stack_node.name = "OpHintUnstack-%s" % flattened[0].name
      stack_node.attr["num"].i = len(flattened)
      output_type = flattened[0].attr["T"].type
      stack_node.attr["T"].type = output_type
      stack_node.input.append(_tensorflow_output_name(
          fused_op_name, output_index))
      out_graphdef.node.extend([stack_node])

      for idx, discrete in enumerate(flattened):
        output_node = _copy.deepcopy(discrete)
        del output_node.input[:]
        output_node.input.append(_tensorflow_output_name(stack_node.name, idx))
        out_graphdef.node.extend([output_node])

      return output_type

  def __str__(self):
    s = "\t\t\tAGGREGATE %s\n" % self.aggregation
    for sort, val in self.names.iteritems():
      s += "\t\t\t%d: %s\n" % (sort, val)
    return s


class _LiteFuncCall(object):
  """Represent a TensorFlow Lite custom function.

  This is uses to accumulate found hints in the graphdef into a single
  conceptual unit.

  Attributes:
    inputs: inputs to the op (hash from index # to argument)
    outputs: outputs to the op (hash from index # to argument)
    function_name: the tflite custom op name to use
    uuid: a unique call id for this particular call  (i.e.
      multiple function calls would have the same function_name but different
      uuids.
    params: A param name to key value for op constant data. I.e. for
      axis on a reduction, strides on a convolution, etc.
    level: Level of the OpHint.
    children_inputs_mappings: If the Ophint has children, children inputs
      mappings indicate how their inputs & outputs are mapped.
  """

  def __init__(self):
    self.inputs = {}
    self.outputs = {}
    self.function_name = None
    self.uuid = None
    self.params = {}
    self.level = -1
    self.children_inputs_mappings = {}

  def flattened_inputs_and_outputs(self):
    """Return a list of inputs and outputs in a flattened format.

    Returns:
      Tuple of (inputs, outputs). where input and output i a list of names.
    """
    def _flatten(input_or_output_dict):
      flattened_items = []
      for item in input_or_output_dict.values():
        flattened_items.extend(item.flatten())
      return flattened_items

    return _flatten(self.inputs), _flatten(self.outputs)

  def __str__(self):
    def format_args(items):
      s = ""
      for idx, item in items.iteritems():
        s += ("\t\t%d:\n" % idx) + str(item)
      return s

    inputs_str = "\tInputs\n" + format_args(self.inputs)
    outputs_str = "\tOutputs\n" + format_args(self.outputs)

    return (
        "tflite function %s call %s level %d "
        "\n\tinputs:\n\t\t%s\n\toutputs:\n\t\t%s" %
        (self.function_name, self.uuid, self.level, inputs_str, outputs_str))


def _find_all_hints_in_nodes(nodes):
  """Look at the all the input nodes and return a list of LiteFuncCall objs.

  Args:
    nodes: A TensorFlow graph_def to look for LiteFuncCalls.

  Returns:
    a list of `LifeFuncCall` objects in the form

  """
  func_calls = _collections.defaultdict(_LiteFuncCall)

  for node in nodes:
    attr = node.attr
    # This is an op hint if it has a FUNCTION_UUID_ATTR, otherwise skip
    if (OpHint.FUNCTION_UUID_ATTR not in attr
        or not attr[OpHint.FUNCTION_UUID_ATTR].s):
      continue
    uuid = attr[OpHint.FUNCTION_UUID_ATTR].s

    # Start building function
    call_def = func_calls[uuid]
    call_def.uuid = uuid
    call_def.function_name = attr[OpHint.FUNCTION_NAME_ATTR].s
    call_def.level = attr[OpHint.FUNCTION_LEVEL_ATTR].i
    # Get sorting and aggregation information

    sort = (attr[OpHint.FUNCTION_SORT_INDEX_ATTR].i
            if OpHint.FUNCTION_SORT_INDEX_ATTR in attr else None)
    if sort == -1: sort = None
    aggregation = None
    if OpHint.FUNCTION_AGGREGATE_ATTR in attr:
      aggregation = _compat.as_text(attr[OpHint.FUNCTION_AGGREGATE_ATTR].s)

    if OpHint.CHILDREN_INPUTS_MAPPINGS in attr:
      call_def.children_inputs_mappings = _json.loads(
          _compat.as_text(attr[OpHint.CHILDREN_INPUTS_MAPPINGS].s))

    # Add the input or output
    def put_operand(stuff, index, sort, operand, aggregation):
      """Add a given index into the function structure."""
      if sort is None:
        stuff[index] = _LiteSingleOperand(operand)
      else:
        if index not in stuff:
          stuff[index] = _LiteAggregateOperand(aggregation)
        stuff[index].add(sort, operand)

    if OpHint.FUNCTION_INPUT_INDEX_ATTR in attr:
      put_operand(call_def.inputs, attr[OpHint.FUNCTION_INPUT_INDEX_ATTR].i,
                  sort, node, aggregation)
    if OpHint.FUNCTION_OUTPUT_INDEX_ATTR in attr:
      put_operand(call_def.outputs, attr[OpHint.FUNCTION_OUTPUT_INDEX_ATTR].i,
                  sort, node, aggregation)

    # Remember attributes
    for a in attr:
      if a.startswith("_tflite_attr_"):
        call_def.params[a.replace("_tflite_attr_,", "")] = attr[a].tensor

  return func_calls


def _extract_topology_sequence_mapping(nodes):
  return dict(
      (_tensor_name_base(node.name), idx) for idx, node in enumerate(nodes))


def _find_children_hints_in_while_loop(function_def, nodes_mapping):
  """Find children hints and all nodes inside the while loop.

  Args:
    function_def: Function def of the while loop.
    nodes_mapping: While loop input_arg : real node name.

  Returns:
    Ordered children hints and all re-mapped nodes inside the while loop.
  """
  new_nodes = []

  # Make nodes inside function def inputs point to the real nodes.
  for node in function_def.node_def:
    for i, _ in enumerate(node.input):
      if node.input[i] in nodes_mapping:
        node.input[i] = nodes_mapping[node.input[i]]
    new_nodes.append(_copy.deepcopy(node))
  name_to_seq_num = _extract_topology_sequence_mapping(function_def.node_def)
  children_hints = _find_all_hints_in_nodes(new_nodes)
  children_hints_q = []
  # Ordered by the outputs.
  for hint in _six.itervalues(children_hints):
    _, output_names = hint.flattened_inputs_and_outputs()
    seq = name_to_seq_num[output_names[0]]
    for output_name in output_names:
      seq = min(seq, name_to_seq_num[output_name])
    children_hints_q.append((seq, hint))
  children_hints_q.sort(key=lambda tup: tup[0])
  ordered_children_hints = [x[1] for x in children_hints_q]
  return ordered_children_hints, new_nodes


def _find_children_hints(call, graph_def):
  """Find all children hints.

  For a given OpHint, we find all children hints inside it, we also copy all the
  nodes inside function defs (if applicable) to the original graph_def, they are
  returned in a list as well.

  Args:
    call: Parent OpHint that contains children ophints.
    graph_def: Original graph def.

  Returns:
    Ordered children hints inside the parent ophint; new graph def that contains
    nodes inside function defs (if applicable); nodes inside function defs.
  """
  name_to_input_name, _, _ = _extract_graph_summary(graph_def)
  input_names, output_names = call.flattened_inputs_and_outputs()

  reachable_by_input = _bfs_for_reachable_nodes(input_names, name_to_input_name)
  reachable_by_output = _bfs_for_reachable_nodes(output_names,
                                                 name_to_input_name)
  output_nodes_set = set(output_names)
  children_hints = []
  out = _graph_pb2.GraphDef()
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)
  function_def_nodes = set()
  for node in graph_def.node:
    out.node.extend([_copy.deepcopy(node)])
    n = _tensor_name_base(node.name)
    if n in reachable_by_output:
      if n not in reachable_by_input and n not in output_nodes_set:
        # special handle for while loop function def.
        if node.op == "While" or node.op == "StatelessWhile":
          body_name = node.attr["body"].func.name
          inputs_outside_loop = node.input
          for function_def in graph_def.library.function:
            if function_def.signature.name == body_name:
              function_inputs = function_def.signature.input_arg
              assert len(inputs_outside_loop) == len(function_inputs)
              nodes_mapping = {}
              for i, function_input in enumerate(function_inputs):
                nodes_mapping[function_input.name] = inputs_outside_loop[i]
              # TODO(b/123050804): Consider use grappler.
              (children_hints_in_loop,
               new_nodes) = _find_children_hints_in_while_loop(
                   function_def, nodes_mapping)
              function_def_nodes.update([x.name for x in new_nodes])
              children_hints.extend(children_hints_in_loop)
              out.node.extend(new_nodes)

  return children_hints, out, function_def_nodes


def _tensor_name_base(full_tensor_name):
  """Removes the device assignment code from a tensor.

  e.g. _tensor_name_base("foo:3") => "foo"

  Args:
    full_tensor_name: A tensor name that is annotated with a device placement
      (this is what tensor flow introspection gives).
  Returns:
    A name without any device assignment.
  """
  if full_tensor_name.startswith("^"):
    return full_tensor_name[1:]
  return full_tensor_name.split(":")[0]


def _tensorflow_output_name(tensor_name, output_index):
  return tensor_name if output_index == 0 else "%s:%d" % (tensor_name,
                                                          output_index)


# TODO(aselle): This should be converted to grappler in the future.
def _check_subgraph_closed(n, reachable_by_input, input_nodes_set,
                           name_to_input_name):
  """Checks to make sure node only connects to predecessor graph through inputs.

  Args:
    n: Node to check
    reachable_by_input: Nodes that are reachable by all inputs of subgraph
    input_nodes_set: The set of nodes that are "inputs".
    name_to_input_name: Maps from name to the list of inputs.

  Raises:
    TypeError: If the given node uses items past inputs directly.
  """
  next_to_visit = [n]
  visited = set()
  while next_to_visit:
    current_node = next_to_visit.pop()
    visited.add(current_node)
    if (current_node in reachable_by_input
        and current_node not in input_nodes_set):
      raise TypeError(
          "Node %s uses input %s not in input_nodes." % (n, current_node))
    if current_node not in input_nodes_set:
      next_to_visit += [
          input_node for input_node in name_to_input_name[current_node]
          if input_node not in visited
      ]


# TODO(aselle): This should be converted to grappler in the future.
def _convert_single_op_hint_to_stub(call,
                                    graph_def,
                                    function_def_nodes=None,
                                    is_last_run=True):
  """Given a graph_def, converts `call` into a stub and returns a new graph_def.

  Args:
    call: A single function call to be converted.
    graph_def: A graph_def to use as input (that has call obviously).
    function_def_nodes: Nodes inside the function def those are not connected to
      the graph.
    is_last_run: Whether it is the last run for a given pass (for OpHint has
      children).

  Returns:
    A new transformed graph-def that has call as a stub (single op).

  Note: after this process, the graph_def can no longer be loaded into
      the tensorflow runtime, so all future manipulations are done in graph_def
      level.
  """
  if function_def_nodes is None:
    function_def_nodes = set()
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      graph_def)
  input_names, output_names = call.flattened_inputs_and_outputs()

  reachable_by_input = _bfs_for_reachable_nodes(input_names, name_to_input_name)
  reachable_by_output = _bfs_for_reachable_nodes(output_names,
                                                 name_to_input_name)
  output_nodes_set = set(output_names)
  nodes_after_fuse = []
  nodes_deleted_by_fuse = set()
  # Classify each node. We want to keep everything reachable by input, but
  # we don't know if things that are not reachable by output or input (things
  # after fusing).
  for node in graph_def.node:
    n = _tensor_name_base(node.name)
    if n in reachable_by_output:
      if n not in reachable_by_input and n not in output_nodes_set:
        nodes_deleted_by_fuse.add(n)
    elif n not in reachable_by_input and n not in function_def_nodes:
      # n is a node that after all the fusings, so keep it.
      nodes_after_fuse.append(n)
    else:
      # In the last run, n is a node that is randomly in the graph but not
      # connected to the chain of dependencies, we will delete n, otherwise
      # we keep them.
      if not is_last_run:
        nodes_after_fuse.append(n)

  # Make a new graphdef with all the pre-input and input nodes
  out = _graph_pb2.GraphDef()
  reachable_by_input_sorted = sorted(
      list(reachable_by_input), key=lambda n: name_to_seq_num[n])
  for node in reachable_by_input_sorted:
    out.node.extend([_copy.deepcopy(name_to_node[node])])

  # Create any stacks to aggregate arguments into to a single input
  # i.e. for static_rnn's.
  # TODO(aselle): Check that the inputs are complete i.e. 0 to n-1
  sorted_input_indices = list(call.inputs.keys())
  sorted_input_indices.sort()
  sorted_output_indices = list(call.outputs.keys())
  sorted_output_indices.sort()
  new_node = _node_def_pb2.NodeDef()
  # Delegate to each operand to produce the proper new input for this stub node.
  # In particular, an aggregate input will now be a Pack of some previously
  # non-fused things.
  for input_index in sorted_input_indices:
    inputs = call.inputs[input_index]
    input_name = inputs.aggregate_and_return_name_for_input(out)
    new_node.input.append(input_name)
  new_node.attr[OpHint.TFLITE_INPUT_INDICES].list.i.extend(sorted_input_indices)

  # Create the function
  new_node.op = call.function_name
  new_node.name = call.uuid
  out.node.extend([new_node])

  # Now call each output argument to give them a chance to make the proper
  # output type and add it to our new_node.
  output_dtypes = []
  for output_index in sorted_output_indices:
    output = call.outputs[output_index]
    output_dtype = (
        output.aggregate_and_return_name_for_output(new_node.name, output_index,
                                                    out))
    output_dtypes.append(output_dtype)
  new_node.attr["_output_types"].list.type[:] = output_dtypes
  # TODO(aselle): what is right here?
  new_node.attr["_output_quantized"].b = False

  # Add post output nodes that do not depend on the outputs
  for n in nodes_after_fuse:
    should_keep = True
    for input_name in name_to_input_name[n]:
      if input_name in nodes_deleted_by_fuse:
        should_keep = False
    if should_keep:
      out.node.extend([_copy.deepcopy(name_to_node[n])])

  # Misc. graph_def data that needs copying.
  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)

  return out


# TODO(aselle): This should be converted to grappler in the future.
def _remove_one_redundant_stack_unstack(in_graph_def):
  """Removes a stack->unstack pattern from in_graph_def in a returned graph.

  Args:
    in_graph_def: Graph def to use as input.
  Returns:
    Simplified tuple (graph_def, changed_something) where changed_something
    is true if anything was done.
  """
  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      in_graph_def)
  del name_to_seq_num

  # TODO(aselle): Make this not hardcoded.
  do_generic_pack_unpack = True

  out = _graph_pb2.GraphDef()
  out.library.CopyFrom(in_graph_def.library)
  out.versions.CopyFrom(in_graph_def.versions)
  for n in in_graph_def.node:
    node_name = _tensor_name_base(n.name)
    if not node_name.startswith("OpHintStack") and not n.op.startswith("Pack"):
      continue
    next_to_visit = [node_name]
    visited = set()

    unpack_nodes = set()
    pack_node = node_name

    # Find a pattern of unstack connected to a stack (with identities
    # in between.
    matches_pattern = True
    is_hint_created_stack = False
    while next_to_visit:
      current_node_name = next_to_visit[0]
      visited.add(current_node_name)
      del next_to_visit[0]
      node = name_to_node[current_node_name]
      is_op_hint_stack = node.name.startswith("OpHintStack")
      is_op_hint_unstack = node.name.startswith("OpHintUnstack")
      if (node.op == "Identity" or is_op_hint_stack
          or (do_generic_pack_unpack and node.op == "Pack")):
        is_hint_created_stack |= is_op_hint_stack
        next_to_visit += [
            input_node for input_node in name_to_input_name[current_node_name]
            if input_node not in visited
        ]
      elif (is_op_hint_unstack
            or (do_generic_pack_unpack and node.op == "Unpack")):
        unpack_nodes.add(node.name)
        is_hint_created_stack &= is_op_hint_unstack
      else:
        matches_pattern = False
        break
      visited.add(node.name)

    if matches_pattern and len(unpack_nodes) == 1:
      pack_node = node_name

      # Check to see if anyone depends on the intermediate identity or the
      # Unstacked form
      no_external_dependency = True
      for other_n in in_graph_def.node:
        if other_n.name in visited: continue
        for input_tensor in name_to_input_name[other_n.name]:
          input_op = _tensor_name_base(input_tensor)
          if input_op in visited and input_op != pack_node:
            no_external_dependency = False
      # Proceed with the substitution if the stack/unstack pair was created
      # through hints, or that it was not, but nobody is consuming things
      # between the stack and unstack.
      if is_hint_created_stack or no_external_dependency:
        end = unpack_nodes.pop()
        end_input = name_to_node[end].input[0]
        # All nodes that depend on the final stack need to be redone to use
        for other_n in in_graph_def.node:
          node_name = _tensor_name_base(other_n.name)
          if node_name not in visited:
            new_node = _copy.deepcopy(other_n)
            new_node.input[:] = [
                (end_input if stripped == pack_node else
                 non_stripped) for stripped, non_stripped in zip(
                     name_to_input_name[node_name], new_node.input[:])
            ]
            out.node.extend([new_node])
        return out, True
  return in_graph_def, False


def _remove_redundant_stack_unstack(graph_def):
  curr = graph_def
  del graph_def
  changed_stuff = True
  while changed_stuff:
    curr, changed_stuff = _remove_one_redundant_stack_unstack(curr)
  return curr


def _get_correct_mapping(original_index, nodes):
  # Special handle for the index is -1 case.
  # If it is -1, return the last index.
  if original_index == -1:
    node_indices = nodes.keys()
    node_indices = sorted(node_indices)
    return node_indices[-1]
  else:
    return original_index
  return original_index


def _convert_op_hints_to_stubs_helper(
    graph_def, write_callback=lambda sess, graph_def: None):
  """Converts a graph_def to a new graph_def where all op hints are stubbed.

  Args:
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).
  Returns:
    A new stubbed graph_def.
  """
  hints = _find_all_hints_in_nodes(graph_def.node)

  hints_q = []
  for hint in _six.itervalues(hints):
    hints_q.append((hint.level, hint.uuid))

  hints_q.sort(key=lambda tup: tup[0])
  for i in range(len(hints_q) - 1, -1, -1):
    level, hint_uuid = hints_q[i]

  curr_graph_def = graph_def
  del graph_def  # prevent using graph_def again (common source of error)
  for i in range(len(hints_q) - 1, -1, -1):
    level, hint_uuid = hints_q[i]
    if level >= 2:
      children_hints, curr_graph_def, function_def_nodes = _find_children_hints(
          hints[hint_uuid], curr_graph_def)
      # pylint: disable=superfluous-parens
      assert (len(children_hints) > 0)  #  pylint: disable=g-explicit-length-test
      # pylint: enable=superfluous-parens

      # Re-wire the children hints inputs/outputs, so latter child's inputs
      # connect to previous child node's outputs.
      children_inputs_mappings = hints[hint_uuid].children_inputs_mappings
      for j, child_hint in enumerate(children_hints):
        if j == 0:
          for mapping in children_inputs_mappings["parent_first_child_input"]:
            parent_input_index = _get_correct_mapping(
                mapping["parent_ophint_input_index"], hints[hint_uuid].inputs)
            child_input_index = _get_correct_mapping(
                mapping["first_child_ophint_input_index"], child_hint.inputs)
            child_hint.inputs[child_input_index] = hints[hint_uuid].inputs[
                parent_input_index]
        else:
          for mapping in children_inputs_mappings[
              "internal_children_input_output"]:
            input_index = _get_correct_mapping(mapping["child_input_index"],
                                               child_hint.inputs)
            output_index = _get_correct_mapping(mapping["child_output_index"],
                                                children_hints[j - 1].outputs)
            child_hint.inputs[input_index] = children_hints[
                j - 1].outputs[output_index]
        if j == len(children_hints) - 1:
          for mapping in children_inputs_mappings["parent_last_child_output"]:
            parent_output_index = _get_correct_mapping(
                mapping["parent_output_index"], hints[hint_uuid].outputs)
            child_output_index = _get_correct_mapping(
                mapping["child_output_index"], child_hint.outputs)
            child_hint.outputs[child_output_index] = hints[hint_uuid].outputs[
                parent_output_index]

      for j, child_hint in enumerate(children_hints):
        curr_graph_def = _convert_single_op_hint_to_stub(
            child_hint, curr_graph_def, function_def_nodes,
            j == len(children_hints) - 1)
    else:
      curr_graph_def = _convert_single_op_hint_to_stub(hints[hint_uuid],
                                                       curr_graph_def)
      write_callback(curr_graph_def, "initial")
  # The stubbing process can create stacks/unstacks in the case of LSTMs
  # remove them.
  curr_graph_def = _remove_redundant_stack_unstack(curr_graph_def)
  return curr_graph_def


def find_all_hinted_output_nodes(session=None, graph_def=None):
  """Find all Ophints output nodes in the graph.

  This is used to get all the output nodes those are ophinted, it is important
  for operation like convert_variables_to_constants keep all ophints structure.
  Note: only one of session or graph_def should be used, not both.
  Why this can be useful? Some TensorFlow ops (e.g. bidirectional rnn), can
  generate multiple outputs for unfused subgraph. If not all output nodes are
  consumed, graph optimization can potentially drop the unused nodes and cause
  ophints in an invalid states (due to missing ophinted output nodes). So it's
  important for us to find all those hinted output nodes and make sure they're
  not discarded away.

  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.

  Returns:
    A list of OpHints output nodes.
  Raises:
    ValueError: If both session and graph_def are provided.
  """
  if session is not None and graph_def is not None:
    raise ValueError("Provide only one of session and graph_def.")
  hinted_outputs_nodes = []
  if session is not None:
    hints = _find_all_hints_in_nodes(session.graph_def.node)
  elif graph_def is not None:
    hints = _find_all_hints_in_nodes(graph_def.node)
  for hint in _six.itervalues(hints):
    _, output_nodes = hint.flattened_inputs_and_outputs()
    hinted_outputs_nodes.extend(output_nodes)
  return hinted_outputs_nodes


@_tf_export(v1=["lite.experimental.convert_op_hints_to_stubs"])
def convert_op_hints_to_stubs(session=None,
                              graph_def=None,
                              write_callback=lambda graph_def, comments: None):
  """Converts a graphdef with LiteOp hints into stub operations.

  This is used to prepare for toco conversion of complex intrinsic usages.
  Note: only one of session or graph_def should be used, not both.

  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).
  Returns:
    A new graphdef with all ops contained in OpHints being replaced by
    a single op call with the right parameters.
  Raises:
    ValueError: If both session and graph_def are provided.
  """

  if session is not None and graph_def is not None:
    raise ValueError("Provide only one of session and graph_def.")

  if session is not None:
    return _convert_op_hints_to_stubs_helper(session.graph_def, write_callback)
  elif graph_def is not None:
    return _convert_op_hints_to_stubs_helper(graph_def, write_callback)
  else:
    raise ValueError("Must specify session or graph_def as input.")


_allowed_symbols = [
    "OpHint", "convert_op_hints_to_stubs", "convert_op_hints_to_stubs_new",
    "find_all_hinted_output_nodes"
]
remove_undocumented(__name__, _allowed_symbols)
