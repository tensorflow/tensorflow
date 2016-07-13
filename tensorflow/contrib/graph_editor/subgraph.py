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
"""SubGraphView: a subgraph view on an existing tf.Graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from six import iteritems
from six import StringIO

from tensorflow.contrib.graph_editor import select
from tensorflow.contrib.graph_editor import util
from tensorflow.python.framework import ops as tf_ops


def _check_within_range(mapping, n, repetition):
  """Check is the mapping is valid.

  Args:
    mapping: an iterable of integer.
    n: define the input domain as [0, n-1]. Note that the mapping can be
      under-complete, that is, it can only contain a subset of the integers on
      [0, n-1].
    repetition: if True repetition are allowed (the function is surjective)
      otherwise repetition are not allowed (the function is injective).
  Raises:
    ValueError: if the mapping is out of range ot if repetition is False and
      the mapping has some repetition.
  """
  for i in mapping:
    if not 0 <= i < n:
      raise ValueError("Out of [0, {}[ range: {}".format(n, i))
  if not repetition and len(set(mapping)) != len(mapping):
    raise ValueError("Found repetition in mapping: {}".format(mapping))


class SubGraphView(object):
  """A subgraph view on an existing tf.Graph.

  An instance of this class is a subgraph view on an existing tf.Graph.
  "subgraph" means that it can represent part of the whole tf.Graph.
  "view" means that it only provides a passive observation and do not to act
  on the tf.Graph. Note that in this documentation, the term "subgraph" is often
  used as substitute to "subgraph view".

  A subgraph contains:
  - a list of input tensors, accessible via the "inputs" property.
  - a list of output tensors, accessible via the "outputs" property.
  - and the operations in between, accessible via the "ops" property.

  An subgraph can be seen as a function F(i0, i1, ...) -> o0, o1, ... It is a
  function which takes as input some input tensors and returns as output some
  output tensors. The computation that the function performs is encoded in the
  operations of the subgraph.

  The tensors (input or output) can be of two kinds:
  - connected: a connected tensor connects to at least one operation contained
  in the subgraph. One example is a subgraph representing a single operation
  and its inputs and outputs: all the input and output tensors of the op
  are "connected".
  - passthrough: a passthrough tensor does not connect to any operation
  contained in the subgraph. One example is a subgraph representing a
  single tensor: this tensor is passthrough. By default a passthrough tensor is
  present both in the input and output tensors of the subgraph. It can however
  be remapped to only appear as an input (or output) only.

  The input and output tensors can be remapped. For instance, some input tensor
  can be ommited. For instance, a subgraph representing an operation with two
  inputs can be remapped to only take one input. Note that this does not change
  at all the underlying tf.Graph (remember, it is a view). It means that
  the other input is being ignored, or is being treated as "given".
  The analogy with functions can be extended like this: F(x,y) is the original
  function. Remapping the inputs from [x, y] to just [x] means that the subgraph
  now represent the function F_y(x) (y is "given").

  The output tensors can also be remapped. For instance, some output tensor can
  be ommited. Other output tensor can be duplicated as well. As mentioned
  before, this does not change at all the underlying tf.Graph.
  The analogy with functions can be extended like this: F(...)->x,y is the
  original function. Remapping the outputs from [x, y] to just [y,y] means that
  the subgraph now represent the function M(F(...)) where M is the function
  M(a,b)->b,b.

  It is useful to describe three other kind of tensors:
  - internal: an internal tensor is a tensor connecting operations contained
  in the subgraph. One example in the subgraph representing the two operations
  A and B connected sequentially: -> A -> B ->. The middle arrow is an internal
  tensor.
  - actual input: an input tensor of the subgraph, regardless of whether it is
    listed in "inputs" or not (masked-out).
  - actual output: an output tensor of the subgraph, regardless of whether it is
    listed in "outputs" or not (masked-out).
  - hidden input: an actual input which has been masked-out using an
    input remapping. In other word, a hidden input is a non-internal tensor
    not listed as a input tensor and one of whose consumers belongs to
    the subgraph.
  - hidden output: a actual output which has been masked-out using an output
    remapping. In other word, a hidden output is a non-internal tensor
    not listed as an output and one of whose generating operations belongs to
    the subgraph.

  Here are some usefull guarantees about an instance of a SubGraphView:
  - the input (or output) tensors are not internal.
  - the input (or output) tensors are either "connected" or "passthrough".
  - the passthrough tensors are not connected to any of the operation of
  the subgraph.

  Note that there is no guarantee that an operation in a subgraph contributes
  at all to its inputs or outputs. For instance, remapping both the inputs and
  outputs to empty lists will produce a subgraph which still contains all the
  original operations. However, the remove_unused_ops function can be used to
  make a new subgraph view whose operations are connected to at least one of
  the input or output tensors.

  An instance of this class is meant to be a lightweight object which is not
  modified in-place by the user. Rather, the user can create new modified
  instances of a given subgraph. In that sense, the class SubGraphView is meant
  to be used like an immutable python object.

  A common problem when using views is that they can get out-of-sync with the
  data they observe (in this case, a tf.Graph). This is up to the user to insure
  that this doesn't happen. To keep on the safe sife, it is recommended that
  the life time of subgraph views are kept very short. One way to achieve this
  is to use subgraphs within a "with make_sgv(...) as sgv:" Python context.

  To alleviate the out-of-sync problem, some functions are granted the right to
  modified subgraph in place. This is typically the case of graph manipulation
  functions which, given some subgraphs as arguments, can modify the underlying
  tf.Graph. Since this modification is likely to render the subgraph view
  invalid, those functions can modify the argument in place to reflect the
  change. For instance, calling the function swap_inputs(svg0, svg1) will modify
  svg0 and svg1 in place to reflect the fact that their inputs have now being
  swapped.
  """

  def __init__(self, inside_ops=(), passthrough_ts=()):
    """Create a subgraph containing the given ops and the "passthrough" tensors.

    Args:
      inside_ops: an object convertible to a list of tf.Operation. This list
        defines all the operations in the subgraph.
      passthrough_ts: an object convertible to a list of tf.Tensor. This list
        define all the "passthrough" tensors. A passthrough tensor is a tensor
        which goes directly from the input of the subgraph to it output, without
        any intermediate operations. All the non passthrough tensors are
        silently ignored.
    Raises:
      TypeError: if inside_ops cannot be converted to a list of tf.Operation or
        if passthrough_ts cannot be converted to a list of tf.Tensor.
    """
    inside_ops = util.make_list_of_op(inside_ops)
    passthrough_ts = util.make_list_of_t(passthrough_ts)
    ops_and_ts = inside_ops + passthrough_ts
    if ops_and_ts:
      self._graph = util.get_unique_graph(ops_and_ts)
    else:
      self._graph = None
    self._ops = inside_ops

    # Compute inside and outside tensor
    inputs, outputs, insides = select.compute_boundary_ts(inside_ops)

    # Compute passthrough tensors, silently ignoring the non-passthrough ones.
    all_tensors = frozenset(inputs + outputs + list(insides))
    self._passthrough_ts = [t for t in passthrough_ts if t not in all_tensors]

    # Set inputs and outputs.
    self._input_ts = inputs + self._passthrough_ts
    self._output_ts = outputs + self._passthrough_ts

  def __copy__(self):
    """Create a copy of this subgraph.

    Note that this class is a "view", copying it only create another view and
    does not copy the underlying part of the tf.Graph.

    Returns:
      A new identical instance of the original subgraph view.
    """
    cls = self.__class__
    result = cls.__new__(cls)
    for k, v in iteritems(self.__dict__):
      if k == "_graph":
        setattr(result, k, v)
      else:
        setattr(result, k, list(v))  # copy the list
    return result

  def _assign_from(self, other):
    """Assign other to itself.

    Args:
      other: another subgraph-view.
    Returns:
      a new instance identical to the original one.
    Raises:
      TypeError: if other is not an SubGraphView.
    """
    if not isinstance(other, SubGraphView):
      raise TypeError("Expected SubGraphView, got: {}".format(type(other)))
    # pylint: disable=protected-access
    self._graph = other._graph
    self._ops = list(other._ops)
    self._passthrough_ts = list(other._passthrough_ts)
    self._input_ts = list(other._input_ts)
    self._output_ts = list(other._output_ts)
    # pylint: enable=protected-access

  def copy(self):
    """Return a copy of itself.

    Note that this class is a "view", copying it only create another view and
    does not copy the underlying part of the tf.Graph.

    Returns:
      a new instance identical to the original one.
    """
    return copy.copy(self)

  def _remap_default(self, remove_input_map=True, remove_output_map=True):
    """Remap in the place the inputs and/or outputs to the default mapping.

    Args:
      remove_input_map: if True the input map is reset to the default one.
      remove_output_map: if True the output map is reset to the default one.
    """
    if not remove_input_map and not remove_output_map:
      return

    # Compute inside and outside tensor
    inputs, outputs, _ = select.compute_boundary_ts(self._ops)
    if remove_input_map:
      self._input_ts = list(inputs) + self._passthrough_ts
    if remove_output_map:
      self._output_ts = list(outputs) + self._passthrough_ts

  def remap_default(self, remove_input_map=True, remove_output_map=True):
    """Remap the inputs and/or outputs to the default mapping.

    Args:
      remove_input_map: if True the input map is reset to the default one.
      remove_output_map: if True the output map is reset to the default one.
    Returns:
      A new modified instance of the original subgraph view with its
        input and/or output mapping reset to the default one.
    """
    res = self.copy()
    res._remap_default(remove_input_map, remove_output_map)  # pylint: disable=protected-access
    return res

  def _remap_inputs(self, new_input_indices):
    """Remap the inputs of the subgraph in-place."""
    _check_within_range(new_input_indices, len(self._input_ts),
                        repetition=False)
    self._input_ts = [self._input_ts[i] for i in new_input_indices]

  def _remap_outputs(self, new_output_indices):
    """Remap the outputs of the subgraph in-place."""
    _check_within_range(new_output_indices, len(self._output_ts),
                        repetition=True)
    self._output_ts = [self._output_ts[i] for i in new_output_indices]

  def _remap_outputs_make_unique(self):
    """Remap the outputs in place so that all the tensors appears only once."""
    output_ts = list(self._output_ts)
    self._output_ts = []
    util.concatenate_unique(self._output_ts, output_ts)

  def _remap_outputs_to_consumers(self):
    """Remap the outputs in place to match the number of consumers."""
    self._remap_outputs_make_unique()
    output_ts = list(self._output_ts)
    self._output_ts = []
    for t in output_ts:
      self._output_ts += [t]*len(t.consumers())

  def remap_outputs_make_unique(self):
    """Remap the outputs so that all the tensors appears only once."""
    res = copy.copy(self)
    res._remap_outputs_make_unique()  # pylint: disable=protected-access
    return res

  def remap_outputs_to_consumers(self):
    """Remap the outputs to match the number of consumers."""
    res = copy.copy(self)
    res._remap_outputs_to_consumers()  # pylint: disable=protected-access
    return res

  def _remove_unused_ops(self, control_inputs=True):
    """Remove unused ops in place.

    Args:
      control_inputs: if True, control inputs are used to detect used ops.
    Returns:
      A new subgraph view which only contains used operations.
    """
    ops = select.get_walks_union_ops(self.connected_inputs,
                                     self.connected_outputs,
                                     within_ops=self._ops,
                                     control_inputs=control_inputs)
    self._ops = [op for op in self._ops if op in ops]

  def remove_unused_ops(self, control_inputs=True):
    """Remove unused ops.

    Args:
      control_inputs: if True, control inputs are used to detect used ops.
    Returns:
      A new subgraph view which only contains used operations.
    """
    res = copy.copy(self)
    res._remove_unused_ops(control_inputs)  # pylint: disable=protected-access
    return res

  def remap_inputs(self, new_input_indices):
    """Remap the inputs of the subgraph.

    If the inputs of the original subgraph are [t0, t1, t2], remapping to [2,0]
    will create a new instance whose inputs is [t2, t0].

    Note that this is only modifying the view: the underlying tf.Graph is not
    affected.

    Args:
      new_input_indices: an iterable of integers representing a mapping between
        the old inputs and the new ones. This mapping can be under-complete and
        must be without repetitions.
    Returns:
      A new modified instance of the original subgraph view with remapped
        inputs.
    """
    res = self.copy()
    res._remap_inputs(new_input_indices)  # pylint: disable=protected-access
    return res

  def remap_outputs(self, new_output_indices):
    """Remap the output of the subgraph.

    If the output of the original subgraph are [t0, t1, t2], remapping to
    [1,1,0] will create a new instance whose outputs is [t1, t1, t0].

    Note that this is only modifying the view: the underlying tf.Graph is not
    affected.

    Args:
      new_output_indices: an iterable of integers representing a mapping between
        the old outputs and the new ones. This mapping can be under-complete and
        can have repetitions.
    Returns:
      A new modified instance of the original subgraph view with remapped
        outputs.
    """
    res = copy.copy(self)
    res._remap_outputs(new_output_indices)  # pylint: disable=protected-access
    return res

  def remap(self, new_input_indices=None, new_output_indices=None):
    """Remap the inputs and outputs of the subgraph.

    Note that this is only modifying the view: the underlying tf.Graph is not
    affected.

    Args:
      new_input_indices: an iterable of integers representing a mapping between
        the old inputs and the new ones. This mapping can be under-complete and
        must be without repetitions.
      new_output_indices: an iterable of integers representing a mapping between
        the old outputs and the new ones. This mapping can be under-complete and
        can have repetitions.
    Returns:
      A new modified instance of the original subgraph view with remapped
        inputs and outputs.
    """
    res = copy.copy(self)
    if new_input_indices is not None:
      res._remap_inputs(new_input_indices)  # pylint: disable=protected-access
    if new_output_indices is not None:
      res._remap_outputs(new_output_indices)  # pylint: disable=protected-access
    return res

  def find_op_by_name(self, op_name):
    """Return the op named op_name.

    Args:
      op_name: the name to search for
    Returns:
      The op named op_name.
    Raises:
      ValueError: if the op_name could not be found.
      AssertionError: if the name was found multiple time.
    """
    res = [op for op in self._ops if op.name == op_name]
    if not res:
      raise ValueError("{} not in subgraph.".format(op_name))
    if len(res) > 1:
      raise AssertionError("More than 1 op named: {}!".format(op_name))
    return res[0]

  def __getitem__(self, op_name):
    return self.find_op_by_name(op_name)

  def __str__(self):
    res = StringIO()
    def tensor_name(t):
      if t in self._passthrough_ts:
        return "{} *".format(t.name)
      else:
        return t.name
    print("SubGraphView:", file=res)
    print("** ops:", file=res)
    print("\n".join([op.name for op in self._ops]), file=res)
    print("** inputs:", file=res)
    print("\n".join([tensor_name(t) for t in self._input_ts]), file=res)
    print("** outputs:", file=res)
    print("\n".join([tensor_name(t) for t in self._output_ts]), file=res)
    return res.getvalue()

  @property
  def graph(self):
    """The underlying tf.Graph."""
    return self._graph

  @property
  def ops(self):
    """The operations in this subgraph view."""
    return self._ops

  @property
  def inputs(self):
    """The input tensors of this subgraph view."""
    return util.ListView(self._input_ts)

  @property
  def connected_inputs(self):
    """The connected input tensors of this subgraph view."""
    return [t for t in self._input_ts if t not in self._passthrough_ts]

  @property
  def outputs(self):
    """The output tensors of this subgraph view."""
    return util.ListView(self._output_ts)

  @property
  def connected_outputs(self):
    """The connected output tensors of this subgraph view."""
    return [t for t in self._output_ts if t not in self._passthrough_ts]

  @property
  def passthroughs(self):
    """The passthrough tensors, going straight from input to output."""
    return util.ListView(self._passthrough_ts)

  def __nonzero__(self):
    """Allows for implicit boolean conversion."""
    return self._graph is not None

  def op(self, op_id):
    """Get an op by its index."""
    return self._ops[op_id]

  def is_passthrough(self, t):
    """Check whether a tensor is passthrough."""
    return t in self._passthrough_ts

  def __enter__(self):
    """Allow Python context to minize the life time of a subgraph view.

    A subgraph view is meant to be a lightweight and transient object. A short
    lifetime will alleviate the "out-of-sync" issue mentioned earlier. For that
    reason, a SubGraphView instance can be used within a Python context. For
    example:

    from tensorflow.contrib import graph_editor as ge
    with ge.make_sgv(...) as sgv:
      print(sgv)

    Returns:
      Itself.
    """
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def input_index(self, t):
    """Find the input index corresponding to the given input tensor t.

    Args:
      t: the input tensor of this subgraph view.
    Returns:
      the index in the self.inputs list.
    Raises:
      Error: if t in not an input tensor.
    """
    try:
      subgraph_id = self._input_ts.index(t)
    except:
      raise ValueError("Can't find {} in inputs of subgraph {}.".format(
          t.name, self.name))
    return subgraph_id

  def output_index(self, t):
    """Find the output index corresponding to given output tensor t.

    Args:
      t: the output tensor of this subgraph view.
    Returns:
      the index in the self.outputs list.
    Raises:
      Error: if t in not an output tensor.
    """
    try:
      subgraph_id = self._output_ts.index(t)
    except:
      raise ValueError("Can't find {} in outputs of subgraph {}.".format(
          t.name, self.name))
    return subgraph_id

  def consumers(self):
    """Return a Python set of all the consumers of this subgraph view."""
    res = []
    for output in self._output_ts:
      util.concatenate_unique(res, output.consumers())
    return res


def _check_graph(sgv, graph):
  """Check if sgv belongs to the given graph.

  Args:
    sgv: a SubGraphView.
    graph: a graph or None.
  Returns:
    The SubGraphView sgv.
  Raises:
    TypeError: if sgv is not a SubGraphView or if graph is not None and not
      a tf.Graph.
    ValueError: if the graph of sgv and the given graph are not None and
      different.
  """
  if not isinstance(sgv, SubGraphView):
    raise TypeError("Expected a SubGraphView, got: {}".format(type(graph)))
  if graph is None or sgv.graph is None:
    return sgv
  if not isinstance(graph, tf_ops.Graph):
    raise TypeError("Expected a tf.Graph, got: {}".format(type(graph)))
  if sgv.graph is not graph:
    raise ValueError("Graph mismatch.")
  return sgv


def make_view(*args, **kwargs):
  """Create a SubGraphView from selected operations and passthrough tensors.

  Args:
    *args: list of 1) regular expressions (compiled or not) or  2) (array of)
      tf.Operation 3) (array of) tf.Tensor. Those objects will be converted
      into a list of operations and a list of candidate for passthrough tensors.
    **kwargs: keyword graph is used 1) to check that the ops and ts are from
      the correct graph 2) for regular expression query
  Returns:
    A subgraph view.
  Raises:
    TypeError: if the optional keyword argument graph is not a tf.Graph
      or if an argument in args is not an (array of) tf.Tensor
      or an (array of) tf.Operation or a string or a regular expression.
    ValueError: if one of the keyword arguments is unexpected.
  """
  # get keywords arguments
  graph = kwargs["graph"] if "graph" in kwargs else None

  # already a view?
  if len(args) == 1 and isinstance(args[0], SubGraphView):
    return _check_graph(args[0], graph)

  ops, ts = select.select_ops_and_ts(*args, **kwargs)
  sgv = SubGraphView(ops, ts)
  return _check_graph(sgv, graph)


def make_view_from_scope(scope, graph):
  """Make a subgraph from a name scope.

  Args:
    scope: the name of the scope.
    graph: the tf.Graph.
  Returns:
    A subgraph view representing the given scope.
  """
  ops = select.get_name_scope_ops(graph, scope)
  return SubGraphView(ops)
