# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to convert variables to constants in TensorFlow 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import object_identity

# Lazy load the single eager module to avoid introducing new dependencies for
# graph_util:convert_variables_to_constants (eg in
# tensorflow/contrib/session_bundle:session_bundle_py_test).
wrap_function = lazy_loader.LazyLoader(
    "wrap_function", globals(),
    "tensorflow.python.eager.wrap_function")

_CONDITIONAL_OPS = set(["If", "StatelessIf"])
_LOOP_OPS = set(["While", "StatelessWhile"])
_CONTROL_FLOW_OPS = _CONDITIONAL_OPS.union(_LOOP_OPS)


class _TensorData(
    collections.namedtuple("_TensorData", ["numpy", "dtype", "index"])):
  """Data about a tensor that was converted to a constant."""
  __slots__ = ()

  @property
  def dtype_attr(self):
    return attr_value_pb2.AttrValue(type=self.dtype)


class _EndPoint(collections.namedtuple("_EndPoint", ["convertible", "index"])):
  """An endpoint in a graph."""
  __slots__ = ()

  def __str__(self):
    return "{}[{}]".format(self.convertible, self.index)


class _Edge(collections.namedtuple("_Edge", ["source", "destination"])):
  """A directed graph edge."""
  __slots__ = ()

  def __str__(self):
    return "{} -> {}".format(self.source, self.destination)


class _Convertible(object):
  """An entity that can have variables converted to constants."""

  def __init__(self, enclosing_graph):
    self._enclosing_graph = enclosing_graph
    self._outgoing_edges = []
    self._converted_self = None

  def converted_self(self):
    """A copy of this Convertible to be modified during conversion.

    Returns:
      Implementations should return the copied instance, which in turn should
      be contained in converted_enclosing_graph(). This instance is the one that
      will be modified during conversion. Its main use will be in the
      implementations of convert_variable_to_constant().
    """
    raise NotImplementedError()

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    """Converts a variable in this Convertible and its dependencies.

    This method should make sure that a converted copy of itself is present in
    the converted graph, and that all Convertibles depending on this one also go
    through the same process.

    Args:
      incoming_edge: The graph edge into this Convertible that is being
        converted to a constant.
      tensor_data: The tensor representing the constant.
    """
    raise NotImplementedError()

  def create_edges(self):
    """Calls add_outgoing_edge for all edges known to this Convertible.

    This is used to build the graph dependencies, so that conversion of
    variables to constants can be properly propagated through the graph. Usually
    this method will call add_outgoing_edge() to all the Convertible inputs.
    """
    raise NotImplementedError()

  def add_outgoing_edge(self, edge):
    """Adds an outgoing edge to the Convertible's list of edges.

    Args:
      edge: The outgoing edge (its source should be 'self').
    """
    self._outgoing_edges.append(edge)

  @property
  def converted_enclosing_graph(self):
    """The graph being converted."""
    return self._enclosing_graph.converted_self()

  @property
  def outgoing_edges(self):
    """The list of edges starting at this Convertible."""
    return self._outgoing_edges


class _Function(_Convertible):
  """A library function Convertible.

  Edges into functions are edges from node _inputs_ into function _inputs_:
  Functions get their input from their callers, not from node outputs, and the
  callers in turn get those values as inputs.
  """

  def __init__(self, function, enclosing_graph):
    super(_Function, self).__init__(enclosing_graph)
    self._function = function
    self._nodes = {
        n.name:
        _Node.new(node=n, function=self, enclosing_graph=enclosing_graph)
        for n in function.node_def
    }

  def __str__(self):
    return self.function.signature.name

  @property
  def function(self):
    return self._function

  @property
  def nodes(self):
    return self._nodes

  def converted_self(self):
    """The Function copy to be converted.

    The copy will be renamed according to the graph's converted_function_name
    map, to ensure the name does not match anything currently in TensorFlow's
    function cache.

    Returns:
      The function instance to be converted.
    """
    if self._converted_self is None:
      old_name = self.function.signature.name
      new_name = self._enclosing_graph.converted_function_names[old_name]
      self.converted_enclosing_graph.rename_function(old_name, new_name)
      self._converted_self = self.converted_enclosing_graph.functions[new_name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    """Converts one function argument into a constant.

    Args:
      incoming_edge: The edge into the argument to be converted.
      tensor_data: The constant value.
    """
    function = self.converted_self().function
    index = incoming_edge.destination.index
    function.signature.input_arg[index].type = tensor_data.dtype

    for edge in self.outgoing_edges:
      if edge.source.index == index:
        edge.destination.convertible.convert_variable_to_constant(
            edge, tensor_data)

  def create_edges(self):
    for n in self._nodes.values():
      n.create_edges()


class _Node(_Convertible):
  """A Convertible NodeDef."""

  def __init__(self, node, function, enclosing_graph):
    super(_Node, self).__init__(enclosing_graph)
    self._node = node
    self._function = function

  def __str__(self):
    return self._node.name

  @staticmethod
  def new(node, function, enclosing_graph):
    """Creates a new _Node base on its operation type."""
    if node.op in ["VariableV2", "VarHandleOp", "Placeholder"]:
      return _VarHandle(node, function, enclosing_graph)
    elif node.op == "Case":
      return _Case(node, function, enclosing_graph)
    elif node.op == "Merge":
      return _Merge(node, function, enclosing_graph)
    elif node.op == "PartitionedCall":
      return _PartitionedCall(node, function, enclosing_graph)
    elif node.op == "ReadVariableOp":
      return _ReadVariable(node, function, enclosing_graph)
    elif node.op == "ResourceGather":
      return _ResourceGather(node, function, enclosing_graph)
    elif node.op == "ResourceGatherNd":
      return _ResourceGatherNd(node, function, enclosing_graph)
    elif node.op in ["If", "StatelessIf"]:
      return _If(node, function, enclosing_graph)
    elif node.op in ["While", "StatelessWhile"]:
      return _While(node, function, enclosing_graph)
    elif node.op in [
        "Enter", "Exit", "Identity", "NextIteration", "Switch", "_SwitchN"]:
      return _Intermediate(node, function, enclosing_graph)
    else:
      return _Node(node, function, enclosing_graph)

  @property
  def node(self):
    return self._node

  @property
  def container(self):
    """The node container (either a graph or a function)."""
    if self._function is not None:
      return self._function.function
    return self._enclosing_graph.graph_def

  def converted_self(self):
    """The NodeDef to be converted.

    Returns:
      The NodeDef to be converted, which can come from either a graph for a
      function. Derived classes should call this (via 'super') to make sure the
      node is retrieved from the right place.
    """
    if self._converted_self is None:
      source = self._function or self._enclosing_graph
      self._converted_self = source.converted_self().nodes[self._node.name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    pass

  def create_edges(self):
    for index, name in enumerate(self._node.input):
      # Discard edges from control inputs.
      if name[0] == "^":
        continue
      source = self.resolve_input(name)
      source.convertible.add_outgoing_edge(
          _Edge(source, _EndPoint(self, index)))

  def resolve_input(self, input_name):
    """Resolves an input into its _EndPoint.

    A NodeDef's input name can refer to either global NodeDefs (in the
    GraphDef's node list), a NodeDef in a function's node list, or a Function
    (in the GraphDef's function library). The name can also carry semantic
    information, depending on whether it starts with "^". This method handles
    all that logic in order to find the object to which the input name refers
    to.

    Args:
      input_name: The input name to resolve.

    Returns:
      The object referred to by 'input_name'.
    """

    # The logic below oversimplifes the semantics, but is good enough for the
    # purposes of converting to constants. The introduction of new types of
    # operations may change this, forcing the code to be more generic.
    #
    # In particular, we are assuming that the lack of an index suffix means
    # ":0", when it could mean "all the outputs of a node." This works now
    # because converting to constants relies very little on output types, and
    # when it does it specializes its treatment in dedicated classes.
    name_elts = input_name.split(":")
    source_name = name_elts[0]
    if source_name[0] == "^":
      source_name = source_name[1:]
    source_index = 0
    if len(name_elts) > 1 and name_elts[-1].isnumeric():
      source_index = int(name_elts[-1])

    if self._function is None:
      return _EndPoint(self._enclosing_graph.nodes[source_name], source_index)

    if source_index != 0 or source_name in self._function.nodes:
      return _EndPoint(self._function.nodes[source_name], source_index)

    inputs = [i.name for i in self._function.function.signature.input_arg]
    return _EndPoint(self._function, inputs.index(source_name))

  def update_dtype(self, attr_name, index, dtype):
    """Changes the type of a given input.

    Args:
      attr_name: The NodeDef attribute containing the type to change.
      index: The index of the input type to change.
      dtype: The type to change to.
    """
    attr = self._node.attr[attr_name]
    num_types = 0
    # Check for various 'oneof' possibilities, and update the type if
    # index in range.
    if attr.HasField("list"):
      types = attr.list.type
      num_types = len(types)
      if num_types > index:
        types[index] = dtype
        return
    elif attr.HasField("type"):
      num_types = 1
      if index == 0:
        attr.type = dtype
        return
    raise ValueError(
        "Index %d out of range for node(%s).attr(%s), which has %d elements." %
        (index, self._node.name, attr_name, num_types))


class _Intermediate(_Node):
  """Specialization of _Node to intermediate ops."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    node = self.converted_self()
    node.update_dtype("T", incoming_edge.destination.index, tensor_data.dtype)
    if "_output_shapes" in node.node.attr:
      del node.node.attr["_output_shapes"]
    for edge in self.outgoing_edges:
      edge.destination.convertible.convert_variable_to_constant(
          edge, tensor_data)


class _Merge(_Node):
  """Specialization of _Node to Merge ops."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # The Merge operation has a single type for all its inputs, the number of
    # which is reflected in the "N" attribute. For the time being, we assume
    # that unilaterally changing all of them at once is ok.
    super(_Merge, self).convert_variable_to_constant(
        _Edge(incoming_edge.source,
              _Edge(incoming_edge.destination.convertible, 0)), tensor_data)


class _VarHandle(_Node):
  """Specialization of _Node to VarHandleOp."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    tensor_proto = tensor_util.make_tensor_proto(tensor_data.numpy,
                                                 tensor_data.dtype,
                                                 tensor_data.numpy.shape)

    node = self.converted_self().node
    node.Clear()
    node.name = self._node.name
    node.op = "Const"
    node.attr["dtype"].CopyFrom(tensor_data.dtype_attr)
    node.attr["value"].tensor.CopyFrom(tensor_proto)

    for edge in self.outgoing_edges:
      edge.destination.convertible.convert_variable_to_constant(
          edge, tensor_data)


class _ResourceGather(_Node):
  """Specialization of _Node to ResourceGather."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    # We currently skip the conversion if this is inside a function.
    if self._function is not None:
      return
    if self._node.attr["batch_dims"].i != 0:
      raise ValueError("batch_dims != 0 is not supported by freeze_graph.")
    axis_node_name = self._node.name + "/axis"
    axis_dtype = self._node.attr["Tindices"]
    axis_data = np.array(self._node.attr["batch_dims"].i)
    output_axis_node = self.converted_self().container.node.add()
    output_axis_node.name = axis_node_name
    output_axis_node.op = "Const"
    output_axis_node.attr["dtype"].CopyFrom(axis_dtype)
    tensor = tensor_util.make_tensor_proto(
        axis_data, dtype=axis_dtype.type, shape=axis_data.shape)
    output_axis_node.attr["value"].tensor.CopyFrom(tensor)

    output_node = self.converted_self().node
    output_node.Clear()
    output_node.name = self._node.name
    output_node.op = "GatherV2"
    output_node.input.extend(
        [self._node.input[0], self._node.input[1], axis_node_name])
    output_node.attr["Tparams"].CopyFrom(self._node.attr["dtype"])
    output_node.attr["Tindices"].CopyFrom(self._node.attr["Tindices"])
    output_node.attr["Taxis"].CopyFrom(axis_dtype)
    if "_class" in self._node.attr:
      output_node.attr["_class"].CopyFrom(self._node.attr["_class"])


class _ResourceGatherNd(_Node):
  """Specialization of _Node to ResourceGatherNd."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    output_node = self.converted_self().node
    output_node.Clear()
    output_node.name = self._node.name
    output_node.op = "GatherNd"
    output_node.input.extend([self._node.input[0], self._node.input[1]])
    output_node.attr["Tparams"].CopyFrom(self._node.attr["dtype"])
    output_node.attr["Tindices"].CopyFrom(self._node.attr["Tindices"])
    if "_class" in self._node.attr:
      output_node.attr["_class"].CopyFrom(self._node.attr["_class"])


class _ReadVariable(_Node):
  """Specialization of _Node to ReadVariableOp."""

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    node = self.converted_self().node
    node.Clear()
    node.name = self._node.name
    node.op = "Identity"

    node.input.append(self._node.input[0])
    node.attr["T"].CopyFrom(self._node.attr["dtype"])
    if "_class" in self._node.attr:
      node.attr["_class"].CopyFrom(self._node.attr["_class"])

    # If the ReadVariableOp is part of a function, then every node having the
    # ReadVariableOp one as its input will refer to it using a ":value"
    # syntax. We need to change that to ":output".
    if self._function is not None:
      for edge in self.outgoing_edges:
        index = edge.destination.index
        dest = edge.destination.convertible.converted_self()
        if isinstance(dest, _Node):
          input_name_parts = dest.node.input[index].split(":")
          if len(input_name_parts) > 1 and input_name_parts[1] == "value":
            input_name_parts[1] = "output"
            dest.node.input[index] = ":".join(input_name_parts)


class _FunctionCaller(_Node):
  """A base class for Convertibles that reference functions."""

  def __init__(self, node, function, enclosing_graph, first_function_input,
               type_attribute, function_attributes):
    """Initializes a _FunctionCaller.

    Args:
      node: As in _Node.
      function: As in _Node.
      enclosing_graph: As in _Node.
      first_function_input: The index of the first NodeDef input that is tied to
        the function inputs. It is assumed that the rest of the NodeDef inputs
        map one to one to function inputs.
      type_attribute: The name of the NodeDef attribute that defines the input
        types. It is assumed that the types listed here map one-to-one with the
        function inputs (that is, they do _not_ specify types for inputs that
        are not passed to functions).
      function_attributes: The names of the NodeDef attributes containing
        references to functions.
    """
    super(_FunctionCaller, self).__init__(node, function, enclosing_graph)
    self._first_function_input = first_function_input
    self._type_attribute = type_attribute
    self._function_attributes = function_attributes

  def converted_self(self):
    if self._converted_self is None:
      node = super(_FunctionCaller, self).converted_self().node
      converted_names = self._enclosing_graph.converted_function_names
      for attr_name in self._function_attributes:
        attr = node.attr[attr_name]
        if attr.HasField("func"):
          attr.func.name = converted_names[attr.func.name]
        elif attr.HasField("list"):
          for func in attr.list.func:
            func.name = converted_names[func.name]
    return self._converted_self

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    node = self.converted_self()
    index = incoming_edge.destination.index
    if index >= self._first_function_input:
      node.update_dtype(self._type_attribute,
                        index - self._first_function_input, tensor_data.dtype)

    # The loop below is reasonable but not correct in general:
    # The outgoing edges going into the functions are correct, because the
    # inputs map to the function inputs. But the edges going into other nodes do
    # not take into account the logic of the body function, which may do
    # arbitrary things to the node's output:
    #
    #   while x < 0:
    #     return y
    #
    # In this case, the node's ":0" output may map to its ":1 input". For the
    # time being, then, we only process edges into functions.
    for edge in self.outgoing_edges:
      dest = edge.destination.convertible
      if edge.source.index == index and isinstance(dest, _Function):
        dest.convert_variable_to_constant(edge, tensor_data)

  def create_edges(self):
    """Creates edges related to a function caller.

    Edges from a function caller to its called functions are always edges from
    _inputs_ to _inputs_: a FunctionDef input is given by the caller, based on
    its own inputs.
    """
    super(_FunctionCaller, self).create_edges()
    for attr_name in self._function_attributes:
      attr = self._node.attr[attr_name]
      if attr.HasField("func"):
        function = self._enclosing_graph.functions[attr.func.name]
        for index in range(len(self._node.input) - self._first_function_input):
          self.add_outgoing_edge(
              _Edge(
                  _EndPoint(self, index + self._first_function_input),
                  _EndPoint(function, index)))
      elif attr.HasField("list"):
        for func in attr.list.func:
          function = self._enclosing_graph.functions[func.name]
          for index in range(
              len(self._node.input) - self._first_function_input):
            self.add_outgoing_edge(
                _Edge(
                    _EndPoint(self, index + self._first_function_input),
                    _EndPoint(function, index)))


class _If(_FunctionCaller):
  """Specialization of _Node to If-like operations."""

  def __init__(self, node, function, enclosing_graph):
    super(_If, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=1,
        type_attribute="Tin",
        function_attributes=["then_branch", "else_branch"])


class _Case(_FunctionCaller):
  """Specialization of _Node to Case-like operations."""

  def __init__(self, node, function, enclosing_graph):
    super(_Case, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=1,
        type_attribute="Tin",
        function_attributes=["branches"])


class _PartitionedCall(_FunctionCaller):
  """Specialization of _Node to PartitionedCall-like operations."""

  def __init__(self, node, function, enclosing_graph):
    super(_PartitionedCall, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=0,
        type_attribute="Tin",
        function_attributes=["f"])


class _While(_FunctionCaller):
  """Specialization of _Node to While-like operations."""

  def __init__(self, node, function, enclosing_graph):
    super(_While, self).__init__(
        node,
        function,
        enclosing_graph,
        first_function_input=0,
        type_attribute="T",
        function_attributes=["body", "cond"])

  def convert_variable_to_constant(self, incoming_edge, tensor_data):
    super(_While, self).convert_variable_to_constant(incoming_edge, tensor_data)
    node = self.converted_self()
    if node.node.attr["output_shapes"].list.shape:
      node.node.attr["output_shapes"].list.shape[
          incoming_edge.destination.index].CopyFrom(
              tensor_shape_pb2.TensorShapeProto(dim=[
                  tensor_shape_pb2.TensorShapeProto.Dim(size=dim)
                  for dim in tensor_data.numpy.shape
              ]))

    # The while's body inputs and outputs have the same type, so here we can go
    # ahead and change that function's output type.
    body_name = self._node.attr["body"].func.name
    body = self._enclosing_graph.functions[body_name].converted_self().function
    body.signature.output_arg[
        incoming_edge.destination.index].type = tensor_data.dtype


class _GraphDef(_Convertible):
  """A convertible GraphDef."""

  def __init__(self, graph_def):
    super(_GraphDef, self).__init__(enclosing_graph=None)
    self._graph_def = graph_def
    self._nodes = {
        n.name: _Node.new(node=n, function=None, enclosing_graph=self)
        for n in graph_def.node
    }
    self._functions = {
        f.signature.name: _Function(f, enclosing_graph=self)
        for f in graph_def.library.function
    }
    self.create_edges()
    self._converted_function_names = None

  @property
  def graph_def(self):
    return self._graph_def

  @property
  def nodes(self):
    return self._nodes

  @property
  def functions(self):
    return self._functions

  @property
  def converted_function_names(self):
    """Map from original to new function names.

    In order to avoid conflicts (two functions with the same name, one converted
    and one not), we need to change the name of every converted function to
    something that is hopefully unique.

    Returns:
      Map from original to new suggested function names.
    """
    if self._converted_function_names is None:
      parsed_names = []  # List of (id, base_name, original_name)
      for name in self.functions:
        elements = name.rsplit("_", 1)
        if len(elements) == 2 and elements[1].isnumeric():
          parsed_names.append((int(elements[1]), elements[0], name))
        else:
          parsed_names.append((-1, name, name))
      self._converted_function_names = {
          name: "{}_frozen_{}".format(base_name, ops.uid())
          for (_, base_name, name) in sorted(parsed_names)
      }

    return self._converted_function_names

  def rename_function(self, old_name, new_name):
    func = self.functions.pop(old_name)
    func.function.signature.name = new_name
    self.functions[new_name] = func

  def converted_self(self):
    if self._converted_self is None:
      copied_graph = graph_pb2.GraphDef()
      copied_graph.CopyFrom(self._graph_def)
      self._converted_self = _GraphDef(copied_graph)
    return self._converted_self

  def create_edges(self):
    for n in self._nodes.values():
      n.create_edges()
    for f in self._functions.values():
      f.create_edges()


class _ConverterData(object):
  """Container for constant conversion supporting data.

  The data includes the graph being converted, and the pre-converted
  tensors. This class will be specialized for ConcreteFunction and Session-based
  conversions, as the means to obtain that data is different for each case.
  """

  def __init__(self,
               graph_def,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    self._graph_def = graph_def
    self._tensor_data = {}
    self._build_node_defs_list()
    self._variable_names_allowlist = variable_names_allowlist
    self._variable_names_denylist = variable_names_denylist

  @property
  def graph_def(self):
    """The graph to be converted."""
    return self._graph_def

  @property
  def node_defs(self):
    """All the node defs in the graph to be converted.

    Returns:
      A map from node name to the NodeDef for all NodeDefs in the graph, as well
      as all control flow NodeDefs in the functions.
    """
    return self._node_defs

  @property
  def tensor_data(self):
    """A map from tensor name to its converted _TensorData."""
    return self._tensor_data

  def _should_convert(self, name):
    """Checks whether to convert the given variable name to a constant."""
    return (self._variable_names_allowlist is None or
            name in self._variable_names_allowlist) and (
                self._variable_names_denylist is None or
                name not in self._variable_names_denylist)

  def _build_node_defs_list(self):
    """Builds the list of NodeDefs in the GraphDef.

    This list consists of all NodeDefs in the main graph as well as all control
    flow NodeDefs in the functions.

    The remaining NodeDefs in the functions are not included because the op
    names
    are not unique and the variables are handled differently than the main
    graph.
    The control flow ops need to be extracted because they are need their
    attributes to be updated similar to the control flow ops in the main graph.
    """
    self._node_defs = {node.name: node for node in self._graph_def.node}

    if self._graph_def.library:
      for func in self._graph_def.library.function:
        self._node_defs.update({
            node.name: node
            for node in func.node_def
            if node.op in _CONTROL_FLOW_OPS
        })


class _FunctionConverterData(_ConverterData):
  """Container for ConcreteFunction-based conversion data."""

  def __init__(self,
               func,
               lower_control_flow,
               aggressive_inlining,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    """Creates the conversion data for the given function.

    Args:
      func: ConcreteFunction.
      lower_control_flow: Boolean indicating whether or not to lower control
        flow ops such as If and While.
      aggressive_inlining: Boolean indicating whether or not to to aggressive
        function inlining (might be unsafe if function has stateful ops, not
        properly connected to control outputs).
      variable_names_allowlist: The set of variable names to convert (by
        default, all variables are converted).
      variable_names_denylist: The set of variable names to omit converting to
        constants.
    """

    self._func = func
    # Inline the graph in order to remove functions when possible.
    graph_def = _run_inline_graph_optimization(func, lower_control_flow,
                                               aggressive_inlining)
    super(_FunctionConverterData, self).__init__(
        graph_def,
        variable_names_allowlist=variable_names_allowlist,
        variable_names_denylist=variable_names_denylist)
    self._build_tensor_data()

  def _build_tensor_data(self):
    """Caches the tensor data for all Placeholders in the given function."""
    map_index_to_variable = {}
    for var in self._func.graph.variables:
      for idx, captured_input in enumerate(self._func.captured_inputs):
        if var.handle is captured_input:  # pylint: disable=protected-access
          map_index_to_variable[idx] = var
          break

    # Iterates through all captures which are represented as Placeholders.
    for idx, (val_tensor, name_tensor) in enumerate(self._func.graph.captures):
      tensor_name = name_tensor.name.split(":")[0]
      if not self._should_convert(tensor_name):
        continue
      if idx in map_index_to_variable:
        data = map_index_to_variable[idx].numpy()
      else:
        data = val_tensor.numpy()
      self._tensor_data[tensor_name] = _TensorData(
          numpy=data,
          dtype=dtypes.as_dtype(data.dtype).as_datatype_enum,
          index=idx)

    # Get data for VariableV2 ops (reference variables) that cannot be lifted.
    for node in self.node_defs.values():
      if node.op == "VariableV2":
        if not self._should_convert(node.name):
          continue
        if node.name not in self.tensor_data:
          with self._func.graph.as_default():
            identity_node = array_ops.identity(
                self._func.graph.as_graph_element(node.name + ":0"))
          pruned_graph = self._func.prune([], [identity_node.name])()[0]
          self._tensor_data[node.name] = _TensorData(
              numpy=pruned_graph.numpy(),
              dtype=node.attr["dtype"].type,
              index=None)


class _SessionConverterData(_ConverterData):
  """Container for Session-based conversion data."""

  def __init__(self,
               session,
               graph_def,
               output_node_names,
               variable_names_allowlist=None,
               variable_names_denylist=None):
    graph_def = graph_util.extract_sub_graph(graph_def, output_node_names)
    super(_SessionConverterData, self).__init__(
        graph_def,
        variable_names_allowlist=variable_names_allowlist,
        variable_names_denylist=variable_names_denylist)

    nodes_to_convert = []
    tensor_names_to_convert = []
    for node in self.graph_def.node:
      if node.op in ["Variable", "VariableV2", "VarHandleOp"]:
        tensor_name = node.name
        if not self._should_convert(tensor_name):
          continue
        if node.op == "VarHandleOp":
          tensor_name = tensor_name + "/Read/ReadVariableOp"
        nodes_to_convert.append(node)
        tensor_names_to_convert.append(tensor_name + ":0")

    if tensor_names_to_convert:
      converted_tensors = session.run(tensor_names_to_convert)
      for node, tensor_value in zip(nodes_to_convert, converted_tensors):
        self._tensor_data[node.name] = _TensorData(
            numpy=tensor_value, dtype=node.attr["dtype"].type, index=None)


def disable_lower_using_switch_merge(graph_def):
  """Set '_lower_using_switch_merge' attributes to False.

  Sets the attribute to False in the NodeDefs in the main graph and the NodeDefs
  in each function's graph.

  Args:
    graph_def: GraphDef proto.

  Returns:
    GraphDef
  """
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.CopyFrom(graph_def)

  def disable_control_flow_lowering(node):
    if node.op in _CONTROL_FLOW_OPS:
      node.attr["_lower_using_switch_merge"].b = False

  for node in output_graph_def.node:
    disable_control_flow_lowering(node)

  if output_graph_def.library:
    for func in output_graph_def.library.function:
      for node in func.node_def:
        disable_control_flow_lowering(node)
  return output_graph_def


def _run_inline_graph_optimization(func, lower_control_flow,
                                   aggressive_inlining):
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops not
      properly connected to control outputs).

  Returns:
    GraphDef
  """
  graph_def = func.graph.as_graph_def()
  if not lower_control_flow:
    graph_def = disable_lower_using_switch_merge(graph_def)

  # In some cases, a secondary implementation of the function (e.g. for GPU) is
  # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
  # TF2 produces a CuDNN-based RNN for GPU).
  # This function suppose to inline all functions calls, but "api_implements"
  # prevents this from happening. Removing the attribute solves the problem.
  # To learn more about "api_implements", see:
  #   tensorflow/core/grappler/optimizers/implementation_selector.h
  for function in graph_def.library.function:
    if "api_implements" in function.attr:
      del function.attr["api_implements"]

  meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph)

  # Clear the initializer_name for the variables collections, since they are not
  # needed after saved to saved_model.
  for name in [
      "variables", "model_variables", "trainable_variables", "local_variables"
  ]:
    raw_list = []
    for raw in meta_graph.collection_def["variables"].bytes_list.value:
      variable = variable_pb2.VariableDef()
      variable.ParseFromString(raw)
      variable.ClearField("initializer_name")
      raw_list.append(variable.SerializeToString())
    meta_graph.collection_def[name].bytes_list.value[:] = raw_list

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Initialize RewriterConfig with everything disabled except function inlining.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  rewrite_options.optimizers.append("function")
  if aggressive_inlining:
    rewrite_options.function_optimization =\
      rewriter_config_pb2.RewriterConfig.AGGRESSIVE
  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _construct_concrete_function(func, output_graph_def,
                                 converted_input_indices):
  """Constructs a concrete function from the `output_graph_def`.

  Args:
    func: ConcreteFunction
    output_graph_def: GraphDef proto.
    converted_input_indices: Set of integers of input indices that were
      converted to constants.

  Returns:
    ConcreteFunction.
  """
  # Create a ConcreteFunction from the new GraphDef.
  input_tensors = func.graph.internal_captures
  converted_inputs = object_identity.ObjectIdentitySet(
      [input_tensors[index] for index in converted_input_indices])
  not_converted_inputs = [
      tensor for tensor in func.inputs if tensor not in converted_inputs
  ]
  not_converted_inputs_map = {
      tensor.name: tensor for tensor in not_converted_inputs
  }

  new_input_names = [tensor.name for tensor in not_converted_inputs]
  new_output_names = [tensor.name for tensor in func.outputs]
  new_func = wrap_function.function_from_graph_def(output_graph_def,
                                                   new_input_names,
                                                   new_output_names)

  # Manually propagate shape for input tensors where the shape is not correctly
  # propagated. Scalars shapes are lost when wrapping the function.
  for input_tensor in new_func.inputs:
    input_tensor.set_shape(not_converted_inputs_map[input_tensor.name].shape)
  return new_func


def _replace_variables_by_constants(converter_data):
  """Replaces variables by constants on a given graph.

  Given a _ConverterData instance with converted variables in its tensor_data
  field, create a new graph where the respective variables are replaced with the
  converted constants.

  Args:
    converter_data: A pre-populated _ConverterData instance.

  Returns:
    The converted graph.
  """
  input_graph = _GraphDef(converter_data.graph_def)

  for tensor_name, tensor_data in converter_data.tensor_data.items():
    input_graph.nodes[tensor_name].convert_variable_to_constant(
        None, tensor_data)

  converted_graph = input_graph.converted_self().graph_def

  converted_input_indices = {
      t.index
      for t in converter_data.tensor_data.values()
      if t.index is not None
  }

  return converted_graph, converted_input_indices


def convert_variables_to_constants_v2(func,
                                      lower_control_flow=True,
                                      aggressive_inlining=False):
  """Replaces all the variables in a graph with constants of the same values.

  TensorFlow 2.0 function for converting all Variable ops into Const ops holding
  the same values. This makes it possible to describe the network fully with a
  single GraphDef file, and allows the removal of a lot of ops related to
  loading and saving the variables. This function runs Grappler's function
  inlining optimization in order to return a single subgraph.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs). (default False)

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """

  converter_data = _FunctionConverterData(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  return _construct_concrete_function(func, output_graph_def,
                                      converted_input_indices)


def convert_variables_to_constants_v2_as_graph(func,
                                               lower_control_flow=True,
                                               aggressive_inlining=False):
  """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  returns the intermediate `GraphDef` as well. This `GraphDef` contains all the
  debug information after all the transformations in the frozen phase.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs).

  Returns:
    ConcreteFunction containing a simplified version of the original, and also
    the intermediate GraphDef containing the node debug information for the
    transformations in the frozen phase.
  """
  converter_data = _FunctionConverterData(
      func=func,
      lower_control_flow=lower_control_flow,
      aggressive_inlining=aggressive_inlining)

  output_graph_def, converted_input_indices = _replace_variables_by_constants(
      converter_data=converter_data)

  frozen_func = _construct_concrete_function(func, output_graph_def,
                                             converted_input_indices)
  return frozen_func, output_graph_def


def convert_variables_to_constants_from_session_graph(
    session,
    graph_def,
    output_node_names,
    variable_names_allowlist=None,
    variable_names_denylist=None):
  """Replaces all the variables in a graph with constants of the same values.

  This function works similarly to convert_variables_to_constants_v2, but it
  retrieves the constant values from a Session instead of from a
  ConcreteFunction. This is useful when converting graphs generated from
  TensorFlow V1, where ConcreteFunctions are not available. This also differs
  from graph_util.convert_variables_to_constants in that it supports resource
  variables when V2 control flow constructions are present.

  Args:
    session: Active TensorFlow session containing the variables.
    graph_def: A GraphDef to convert.
    output_node_names: List of name strings for the result nodes of the graph.
    variable_names_allowlist: The set of variable names to convert (by default,
      all variables are converted).
    variable_names_denylist: The set of variable names to omit converting to
      constants.

  Returns:
    An optimized GraphDef.
  """
  graph_def, _ = _replace_variables_by_constants(
      converter_data=_SessionConverterData(
          session=session,
          graph_def=graph_def,
          output_node_names=output_node_names,
          variable_names_allowlist=variable_names_allowlist,
          variable_names_denylist=variable_names_denylist))
  return graph_def
