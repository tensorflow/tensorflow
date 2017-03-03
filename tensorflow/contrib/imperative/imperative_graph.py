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
"""Imperative mode graph for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import uuid

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import compat

# Stateful operators (with ref type input/outputs) that are allowed to be
# present in an ImperativeGraph.
_REF_OPS_WHITELIST = frozenset(['Variable', 'VariableV2', 'Assign', 'AssignAdd',
                                'AssignSub', 'ScatterAdd', 'ScatterSub',
                                'ScatterUpdate'])

# These ops are returned as is in create_op without the extra logic. This
# saves some space used for auxiliary variables.
_PASS_THROUGH_OPS = frozenset(['Identity'])


class ImperativeGraph(ops.Graph):
  """A class implementing an imperative mode TensorFlow graph.

  The ops constructed in an ImperativeGraph are augmented with extra logic to
  enable its execution in an imperative manner. Imperative graphs are organized
  hierarchically. A new step created from an `ImperativeMode` object creates a
  new graph that is a child of this graph. In that case, an object of this
  class is expected to be initialized with a parent graph, passed as
  `parent_graph` to the initializer. Note that `parent_graph` is expected to
  be set only when initialized from the `ImperativeMode` initializer.
  """

  def __init__(self, parent_graph=None):
    """Initializes an ImperativeGraph.

    Args:
      parent_graph: (Optional) An ImperativeGraph.
    """
    self._parent_graph = parent_graph
    # Whether the create_op function should augment an op with extra logic for
    # imperative execution.
    self._return_as_is = False
    # Operation -> list of Tensors map. Used for overriding the op.outputs
    # property, useful during gradient computation.
    self._outputs_map = {}
    # Operation -> function map. Used for overriding the gradient function
    # for an op.
    self._gradient_function_map = {}
    # Unique name for the graph. Used for naming the container in which
    # temporary variables are placed.
    self._name = uuid.uuid4().hex
    # Names for op types used for marking ops so we can override their
    # gradient functions.
    self._merge_op_type = 'ImperativeMerge' + self._name
    self._imperative_op_type = 'ImperativeOp' + self._name
    # The list of 'assign' ops that initialize variables.
    self._init_ops = []
    # Names of variables whose init ops have been already recorded in _init_ops.
    self._init_variable_names = set()
    # A flag to indicate whether a variable and the corresponding initialization
    # ops are being created. Typically set by the initializer of Variable class.
    self._in_variable_creation = False
    self._variable_cleanup_ops = []
    # Call the parent's initializer.
    super(ImperativeGraph, self).__init__()

    # Register a simple 'pass through' function to be used for ops that have
    # _merge_op_type as the _gradient_op_type attribute.
    ops.RegisterGradient(self._merge_op_type)(
        lambda op, grad, _: [grad] * len(op.inputs))

    # For ops that have _imperative_op_grad as the _gradient_op_type attribute,
    # temporarily replace their outputs with the values in _output_map before
    # calling the original gradient function.
    def _imperative_op_grad(op, *grad):
      with self.replace_outputs(op):
        return self._gradient_function_map[op.name](op, *grad)

    ops.RegisterGradient(self._imperative_op_type)(_imperative_op_grad)

  def op_in_graph(self, op):
    """Checks if op belongs in this graph or its ancestors."""
    # pylint: disable=protected-access
    if op._graph == self:
      return True
    # pylint: enable=protected-access
    if self._parent_graph:
      return self._parent_graph.op_in_graph(op)
    return False

  def is_child_graph(self, child_graph):
    """Checks if this graph is an ancestor of `child_graph`."""
    # pylint: disable=protected-access
    if not child_graph or not child_graph._parent_graph:
      return False
    if child_graph._parent_graph == self:
      return True
    return self.is_child_graph(child_graph._parent_graph)
    # pylint: enable=protected-access

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def record_variable_inits(self):
    """Context manager to record Variable initializations.

    Sets _in_variable_creation to True before a Variable is initialized.

    NOTE(keveman): This is used for recording the list of assign ops
    that are used to initialize variables. It relies on the fact that
    the constructor of Variable class creates exactly one assign op that is
    used for initializing the variable. Variable ops not created using the
    variables.Variable class are not added to _init_ops and hence not
    initialized automatically.

    """
    old_init = getattr(variables.Variable, '__init__')

    def record(*args, **kwargs):
      self._in_variable_creation = True
      old_init(*args, **kwargs)
      self._in_variable_creation = False

    setattr(variables.Variable, '__init__', record)
    yield
    setattr(variables.Variable, '__init__', old_init)
  # pylint: enable=g-doc-return-or-yield

  @contextlib.contextmanager
  def return_as_is(self):
    """Prevents adding the extra logic during `create_op`."""
    old_return_as_is = self._return_as_is
    self._return_as_is = True
    yield
    self._return_as_is = old_return_as_is

  @contextlib.contextmanager
  def replace_outputs(self, op):
    """Replaces the outputs of `op` with values recorded in `_outputs_map`."""
    # pylint: disable=protected-access
    old_outputs = op._outputs
    op._outputs = self._outputs_map[op.name]
    yield
    op._outputs = old_outputs
    # pylint: enable=protected-access

  def add_pending_init(self, init_op):
    """Records assign ops in `_init_ops`."""
    if init_op.type != 'Assign':
      raise TypeError('Init op should be an Assign')

    var_name = init_op.inputs[0].op.name
    if var_name not in self._init_variable_names:
      self._init_variable_names.add(var_name)
      self._init_ops.append(init_op)

  def run_pending_inits(self, session):
    """Runs the pending variable initializations using `session`."""
    while self._init_ops:
      session.run(self._init_ops.pop(0))

  def _wrap(self, op):
    return OperationProxy(op)

  def create_op(self, *args, **kwargs):
    """Creates an `Operation`.

    For operations of the following form

      orig_value = op(*args, **kwargs)

    this function constructs the following subgraph :

      v = Variable()
      if v is not initialized:
        orig_value = op(*args, **kwargs)
        v.assign(orig_value) # Initializes v
        return orig_value
      else:
        return v

    The above transformation is not performed and the original op is returned
    as is if any of the following is true:
    * `_return_as_is` flag is set to true.
    * op_type is listed in _PASS_THROUGH_OPS
    * op has no outputs.
    * One of the op's return value has a ref type.

    Args:
      *args: Arguments for create_op()
      **kwargs: Keyword arguments for create_op(). Refer to
        tensorflow.python.framework.ops.Graph.create_op() for the mandatory
        and optional arguments.

    Returns:
      An Operation.

    Raises:
      UnimplementedError: if output type is a reference and the op's type
        is not one of the supported types in `_REF_OPS_WHITELIST`.
    """
    op_type = kwargs['op_type'] if 'op_type' in kwargs else args[0]
    output_dtypes = kwargs['dtypes'] if 'dtypes' in kwargs else args[2]
    output_dtypes = [dtypes.as_dtype(d) for d in output_dtypes]

    if self._return_as_is or op_type in _PASS_THROUGH_OPS:
      return self._wrap(super(ImperativeGraph, self).create_op(*args, **kwargs))

    if not output_dtypes:
      return self._wrap(
          super(ImperativeGraph, self).create_op(*args, **kwargs))

    output_has_ref = any([dtype._is_ref_dtype for dtype in output_dtypes])  # pylint: disable=protected-access

    if output_has_ref:
      if op_type not in _REF_OPS_WHITELIST:
        raise errors.UnimplementedError(None, None,
                                        op_type + ' op not supported in '
                                        'imperative graph')

      ret = super(ImperativeGraph, self).create_op(*args, **kwargs)

      if self._in_variable_creation:
        if op_type == 'Assign':
          self.add_pending_init(ret)

      return self._wrap(ret)

    with self.return_as_is():
      # Declares the variables to hold the output values of this op.
      op_output_var = [state_ops.variable_op_v2(
          tensor_shape.TensorShape(None), dtype, container=self._name)
                       for dtype in output_dtypes]
      # Ops to free the resources used by the temporary cache variables.
      # The following two ops are created for each cache variable,
      # having no control dependencies on any other ops :
      # var_handle_op ----> destroy_resource_op
      for dtype, v in zip(output_dtypes, op_output_var):
        with ops.control_dependencies(None):
          self._variable_cleanup_ops += [
              gen_resource_variable_ops.destroy_resource_op(
                  gen_resource_variable_ops.var_handle_op(
                      dtype, tensor_shape.TensorShape(None),
                      container=self._name, shared_name=v.op.name),
                  ignore_lookup_error=True)]

      # Create the conditional to run the original op only when the variable
      # corresponding to the first output is not initialized.
      inited = state_ops.is_variable_initialized(op_output_var[0])
      v_f, v_t = control_flow_ops.ref_switch(op_output_var[0], inited)
      # pylint: disable=protected-access
      v_f_op = gen_array_ops._ref_identity(v_f)
      v_t_op = gen_array_ops._ref_identity(v_t)
      # pylint: enable=protected-access

      with ops.control_dependencies([v_f_op.op]):
        # Create the original op
        orig_op = self._wrap(
            super(ImperativeGraph, self).create_op(*args, **kwargs))
      shapes = [val.get_shape() for val in orig_op.outputs]

      controls = []
      for var, val in zip(op_output_var, orig_op.outputs):
        if (not val.get_shape().is_fully_defined() or
            val.get_shape().num_elements() > 0):
          assign_op = state_ops.assign(var, val, validate_shape=False)
          assign_op.set_shape(val.get_shape())
          controls.append(assign_op)

      values = []
      if len(controls) > 1:
        if control_flow_ops.IsSwitch(orig_op):
          # pylint: disable=protected-access
          controls = gen_control_flow_ops._ref_merge(controls)
          # pylint: enable=protected-access
        else:
          controls = control_flow_ops.tuple(controls)

      for var, val in zip(op_output_var, orig_op.outputs):
        with ops.control_dependencies(controls):
          with self.colocate_with(v_f_op):
            real_val = array_ops.identity(val)
        with ops.control_dependencies([v_t_op.op]):
          with self.colocate_with(v_t_op):
            stored_val = array_ops.identity(var)
          stored_val.set_shape(val.get_shape())
          real_val, _ = control_flow_ops.merge([real_val, stored_val])
        real_val.op.node_def.attr['_gradient_op_type'].CopyFrom(
            attr_value_pb2.AttrValue(s=compat.as_bytes(self._merge_op_type)))
        values.append(real_val)

      for i, _ in enumerate(shapes):
        values[i].set_shape(shapes[i])
      self._outputs_map[orig_op.name] = values
      try:
        self._gradient_function_map[orig_op.name] = ops.get_gradient_function(
            orig_op)
      except (KeyError, LookupError):
        pass
      else:
        orig_op.node_def.attr['_gradient_op_type'].CopyFrom(
            attr_value_pb2.AttrValue(
                s=compat.as_bytes(self._imperative_op_type)))

      return MultiOutputOperation(values)


class MultiOutputOperation(object):
  """A 'duck-type' wrapper class for a list of Tensors, acting as an Operation.

  NOTE(keveman): `create_op` produces a list of values but collected from
  multiple ops. So there is no one `Operation` that we can pass to the
  consumers of `create_op`. But the consumers of `create_op` only require
  the object passed in to have the `outputs` property defined. This class
  simply defines the `outputs` property, so the consumers of
  `create_op` work correctly.
  """

  def __init__(self, outputs):
    self.outputs = outputs


class OperationProxy(ops.Operation):
  """A proxy for the `ops.Operation` class.

  Imperative graphs are organized hierarchically. Operations in an imperative
  graph can be constructed out of operations belonging to any of the parent
  graphs available in the lexical scope. This class provides the illusion that
  all such operations belong to the current default graph.
  """
  __slots__ = ['_name', '_original_graph']

  def __init__(self, real_op):
    # object.__setattr__ is used for setting '_name' and '_original_graph'
    # attributes (instead of self._name, for eg.) as this class provides
    # its own __setattr__ method for proxying purposes.
    object.__setattr__(self, '_name', real_op.name)
    object.__setattr__(self, '_original_graph', real_op.graph)

    # pylint: disable=protected-access
    for output in real_op._outputs:
      output._op = self
    real_op._outputs = [TensorProxy(output) for output in real_op._outputs]
    # pylint: enable=protected-access

  def __getattribute__(self, name):
    """Forwards to the methods in the current graph's `Operation` object."""
    op_name = object.__getattribute__(self, '_name')
    graph = ops.get_default_graph()

    # Short-circuit getting some of these attributes that are readily
    # available without forwarding to the actual operation. This is done
    # because `get_operation_by_name` tries to acquire the parent graph's
    # lock protecting the nodes_by_* data structures, and these attributes
    # (not requiring the lock) could be queried by other function holding
    # the lock.
    if name == 'name':
      return op_name
    elif name == '_as_graph_element':
      return lambda: self
    elif name == '__class__':
      return OperationProxy
    elif name == 'graph':
      original_graph = object.__getattribute__(self, '_original_graph')
      if original_graph.is_child_graph(graph):
        return graph
      else:
        return original_graph
    else:
      op = graph.get_operation_by_name(op_name)
      return getattr(op, name)

  def __setattr__(self, name, value):
    # `replace_outputs` overrides _outputs temporarily, so support
    # setting that attribute.
    if name != '_outputs':
      raise NotImplementedError('"op.%s = ..." not implemented' % name)
    op_name = object.__getattribute__(self, '_name')
    graph = ops.get_default_graph()
    op = graph.get_operation_by_name(op_name)
    setattr(op, name, value)


class TensorProxy(ops.Tensor):
  """Forwards to the methods in the current graph's `Tensor` object."""
  __slots__ = ['_name', '_original_tensor', '_original_graph']

  def __init__(self, real_tensor):
    setattr(self, '_name', real_tensor.name)
    setattr(self, '_original_tensor', real_tensor)
    setattr(self, '_original_graph', real_tensor.graph)

  def __str__(self):
    sess = getattr(ops.Tensor, 'session', None)
    if sess:
      return str(sess.run(self))
    else:
      return ops.Tensor.__str__(self)

  def __repr__(self):
    sess = getattr(ops.Tensor, 'session', None)
    if sess:
      return repr(sess.run(self))
    else:
      return ops.Tensor.__repr__(self)

  def __bool__(self):
    sess = getattr(ops.Tensor, 'session', None)
    if sess:
      return bool(sess.run(self))
    else:
      return ops.Tensor.__bool__(self)

  def __nonzero__(self):
    sess = getattr(ops.Tensor, 'session', None)
    if sess:
      return bool(sess.run(self))
    else:
      return ops.Tensor.__nonzero__(self)

  def __getattribute__(self, name):
    tensor_name = object.__getattribute__(self, '_name')
    graph = ops.get_default_graph()

    if name == 'name':
      return tensor_name
    elif name == '_as_graph_element':
      return lambda: self
    elif name == '__class__':
      return TensorProxy
    elif name == 'graph':
      original_graph = object.__getattribute__(self, '_original_graph')
      if original_graph.is_child_graph(graph):
        return graph
      else:
        return original_graph
    elif name == 'value':
      sess = getattr(ops.Tensor, 'session', None)
      if sess:
        return sess.run(self)
      raise AttributeError('Current session not set on Tensor')
    else:
      tensor = object.__getattribute__(
          graph.get_tensor_by_name(tensor_name), '_original_tensor')
      return getattr(tensor, name)


@contextlib.contextmanager
def add_session_attr(typename, session):
  """Sets the `session` property on the typename for the duration of a context.

  This allows us to convert a `tf.Tensor` to numpy array by calling run()
  using the `.session` property.

  Args:
    typename: The class to which value attribute should be added.
    session: Session to be stored.

  Yields:
    None.
  """
  old_session = getattr(typename, 'session', None)
  setattr(typename, 'session', session)
  yield
  if old_session:
    setattr(typename, 'session', old_session)
