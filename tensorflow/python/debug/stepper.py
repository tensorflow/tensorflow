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
"""TensorFlow Debugger (tfdbg) Stepper Module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug import debug_data
from tensorflow.python.framework import ops
from tensorflow.python.ops import session_ops


class NodeStepper(object):
  """TensorFlow Debugger (tfdbg) stepper.

  The stepper provides ability to perform "continue to" actions on a graph,
  given fetch and feeds. The stepper calculates the transitive closure of the
  fetch. cont() (continue to) calls can only be performed on members of the
  transitive closure.

  On a cont() call, the stepper performs depth-first tracing of the input
  tree of the target. When it reaches an input where one of the following is
  available, it will supply the available value to the feed_dict of the cont()
  call:
    (1) TensorHandles from previous cont() calls.
    (2) Overriding (injected) values from the client.
    (3) Feeds supplied during the construction of the stepper instance.

  Once the tracing is complete, it will issue a run() call on the
  underlying session, using the aforementioned feed_dict prepared by the input
  tracing, to achieve the "continue-to" action. The above process takes into
  account whether the transitive closure of an input contains Variables that
  are updated during previous cont() calls on this stepper instance. If such
  updates exist, we say the transitive closure is "dirty" and the stepper
  can restore the "clean" state of the Variable and avoid using the
  TensorHandle.

  Example of basic usage:
    a = tf.Variable(1.0, name="a")
    b = tf.Variable(2.0, anme="b")
    c = tf.add(a, b, name="c")
    d = tf.mul(a, c, name="d")

    sess = tf.Session()
    sess.run(tf.initialize_all_varialbes())
    stepper = NodeStepper(sess, d)

    stepper.cont(c)  # Caches the handle to Tensor c:0.
    stepper.cont(d)  # Uses handle to Tensor c:0, avoiding recomputing c.
  """

  # Possible types of feed used during cont() calls.
  FEED_TYPE_CLIENT = "client"
  FEED_TYPE_HANDLE = "handle"
  FEED_TYPE_OVERRIDE = "override"

  # TODO(cais): The following member constant is currently unused. Use it when
  # the stepper is capable of using dumped intermediate tensors.
  FEED_TYPE_INTERMEDIATE = "intermediate"

  def __init__(self, sess, fetch, feed_dict=None):
    """Constructor for Debugger.

    Args:
      sess: (Session) the TensorFlow Session to step in.
      fetch: (str or TensorFlow graph element) A single fetched Tensor or Op,
        or a name (str) representing the Tensor or Op. In the case of a name
        str, the graph will be searched to find the corresponding Tensor or Op.
      feed_dict: (dict or None) feed dict to be used in this stepper instance.

    TODO(cais): Currently the stepper supports a single fetch. Support list,
      tuple or dict of feeds, as in the Session run() interface.
    """

    self._sess = sess

    if isinstance(fetch, str):
      # Fetch target is a string. Assume it is the name of the Tensor or Op and
      # will attempt to find it in the Session's graph.
      self._fetch_name = fetch
    elif isinstance(fetch, list) or isinstance(fetch, tuple) or isinstance(
        fetch, dict):
      raise NotImplementedError(
          "list, tuple or dict fetches are not supported yet.")
    else:
      self._fetch_name = fetch.name
    self._fetch = self._sess.graph.as_graph_element(self._fetch_name)

    # A map from Variable name to initializer op.
    self._variable_initializers = {}

    # A map from Variable name to initial value, used when overriding or
    # restoring Variable values.
    self._variable_initial_values = {}

    # Initialize the map for output recipients (targets).
    self._non_control_output_targets = {}

    # Sorted transitive closure of the fetched node.
    self._sorted_transitive_closure = self._dfs_visit(self._sess.graph,
                                                      self._fetch)
    self._transitive_closure_set = set(self._sorted_transitive_closure)

    # A map from Variable name to the old values (before any cont() calls).
    self._cached_variable_values = {}

    # A cache map from tensor name to what variables may invalidate the tensor
    self._cached_invalidation_path = {}

    # Keep track of which variables are in a dirty state.
    self._dirty_variables = set()

    # Cached tensor handles: a dict with keys as tensor names and values as
    # tensor handles.
    self._tensor_handles = {}

    # Feed dict from the client.
    self._client_feed_dict = feed_dict
    if not self._client_feed_dict:
      self._client_feed_dict = {}

    # Overriding tensor values.
    self._override_tensors = {}

    # What the feed types were used by the last cont() call.
    self._last_feed_types = {}

  def _dfs_visit(self, graph, elem):
    """Trace back the input of a graph element, using depth-first search.

    Uses non-recursive implementation to prevent stack overflow for deep
    graphs.

    Also performs the following action(s):
      1) When encountering a Variable, obtain its initializer op, to
         facilitate possible subsequent restoration / overriding of variable
         value.

    Args:
      graph: A TF graph instance.
      elem: A graph element: a Tensor or an Operation.

    Returns:
      (list of str) A topologically-sorted list of all graph element names
        in the transitive closure of elem. Obviously, the topological sort is
        not unique in general. The return value here is just an arbitrary one
        of potentially many possible topological sorts.
    """

    # These set should hold only strings, i.e, names of the nodes.
    done = set()  # Keep track of visited nodes.

    # A list of str: Names of the topologically-sorted graph elements.
    sorted_node_list = [elem.name]

    elem_stack = [elem]

    while elem_stack:
      curr_elem = elem_stack.pop()
      curr_node = self._get_node(curr_elem)

      done.add(curr_node.name)

      non_control_inputs = [inp for inp in curr_node.inputs]
      control_inputs = [inp for inp in curr_node.control_inputs]
      all_inputs = set(non_control_inputs + control_inputs)

      # Iterate through the (non-control) inputs.
      for inp in all_inputs:
        is_non_control_input = inp in non_control_inputs

        # Set up the non-control output map.
        if is_non_control_input:
          if inp.name not in self._non_control_output_targets:
            self._non_control_output_targets[inp.name] = set([curr_elem.name])
          else:
            self._non_control_output_targets[inp.name].add(curr_elem.name)

          if (inp.op.type == "Variable" and
              inp.name not in self._variable_initializers):
            # Obtain the initializer op of the variable, in case the Variable's
            # value needs to be restored later.
            initializer = graph.as_graph_element(inp.op.name + "/Assign")
            self._variable_initializers[inp.name] = initializer
            self._variable_initial_values[inp.name] = initializer.inputs[1]

        inp_node = self._get_node(inp)
        if inp_node.name in done:
          # Already visited.
          continue

        elem_stack.append(inp)
        sorted_node_list.append(inp.name)

    sorted_node_list.reverse()
    return sorted_node_list

  def sorted_transitive_closure(self):
    """Get a sorted list of transitive inputs to the fetch of the stepper.

    Returns:
      (list of str): Sorted transitive inputs to the fetch of the stepper
        instance. The fetch itself is included in the list.
    """

    return self._sorted_transitive_closure

  def is_feedable(self, name):
    """Determine if a graph element if feedable.

    Args:
      name: (str) name of the graph element (Tensor or Operation)

    Returns:
      (bool) whether the graph element is feedable.
    """

    if not isinstance(name, str):
      raise TypeError("Expected type str; got type %s" % type(name))

    elem = self._sess.graph.as_graph_element(name)
    return self._sess.graph.is_feedable(elem)

  def override_tensor(self, tensor_name, overriding_val):
    """Override the value of a tensor.

    Args:
      tensor_name: (str) Name of the tensor to override.
      overriding_val: (numpy.ndarray) Overriding tensor value.

    Raises:
      ValueError: If tensor_name does not correspond to a tensor in the input
        tree to the fetched graph element of this stepper instance.
    """

    if not isinstance(tensor_name, str):
      raise TypeError("Expected type str; got type %s" % type(tensor_name))

    if tensor_name not in self._transitive_closure_set:
      raise ValueError(
          "Cannot override tensor \"%s\" because it does not exist in the "
          "input tree to the fetch \"%s\"" % (tensor_name, self._fetch_name))

    self._override_tensors[tensor_name] = overriding_val

    # Invalidate cache by tracing outputs.
    self._invalidate_transitively_outgoing_cache(tensor_name)

  def remove_override(self, tensor_name):
    """Remove the overriding value on a tensor.

    Args:
      tensor_name: (str) name of the tensor to remove the overriding value
        from.

    Raises:
      ValueError: If no overriding value exists for tensor_name.
    """

    if tensor_name not in self._override_tensors:
      raise ValueError("No overriding value exists for tensor \"%s\"." %
                       tensor_name)

    del self._override_tensors[tensor_name]

    # Invalidate cache by tracing outputs.
    self._invalidate_transitively_outgoing_cache(tensor_name)

  def last_feed_types(self):
    """Obtain information about the feed in the last cont() call.

    Returns:
      (dict) A dict mapping tensor names to feed types.
    """

    return self._last_feed_types

  def cont(self,
           target,
           use_tensor_handles=True,
           use_overrides=True,
           restore_variable_values=False):
    """Continue till the completion of the specified target tensor.

    Args:
      target: A single fetched Tensor or Op, or a name (str) representing the
        Tensor or Op. In the case of a name str, the graph will be searched
        to find the corresponding Tensor or Op.
        # TODO(cais): Support multiple fetches as in Session.run() interface.
      use_tensor_handles: (bool) Whether this cont() run will use cached tensor
        handles to avoid recomputation. Default: True.
      use_overrides: (bool) Whether the overriding tensor values supplied by
        the client are to be used in this cont() call. Default: True.
      restore_variable_values: (bool) Whether the old values of the variables
        (before any cont() calls in this object) are to be restored.

    Returns:
      Value from Session.run() of the target.

    Raises:
      ValueError: If the target is specified as a string and the string does
        not correspond to any tensors in the Session graph.
        Or if the target of this cont() is not in the input list of the Stepper
        object's target.
        Or if target is a Placeholder.
    """

    self._last_feed_types = {}

    if isinstance(target, str):
      # Fetch target is a string. Assume it is the name of the Tensor or Op and
      # will attempt to find it in the Session's graph.
      target_name = target
    else:
      target_name = target.name

    graph_element = self._sess.graph.as_graph_element(target_name)
    if (isinstance(graph_element, ops.Tensor) and
        graph_element.op.type == "Placeholder"):
      raise ValueError("Should not call cont() on a Placeholder")

    # Verify that the target is in the transitive closure of the stepper's
    # fetch.
    if target_name not in self._transitive_closure_set:
      raise ValueError(
          "Target \"%s\" is not in the transitive closure for the fetch of the "
          "stepper: \"%s\"." % (target_name, self._fetch_name))

    # Check if a cached tensor handle can be used on the fetch directly.
    if use_tensor_handles and target_name in self._tensor_handles:
      self._last_feed_types[target_name] = self.FEED_TYPE_HANDLE
      return self._tensor_handles[target_name].eval()

    # Check if an overriding tensor value can be used directly.
    if use_overrides and target_name in self._override_tensors:
      # Override is available. Return the value right away.
      self._last_feed_types[target_name] = self.FEED_TYPE_OVERRIDE
      return self._override_tensors[target_name]

    # The feeds to be used in the Session.run() call.
    feeds = {}

    # Keep track of which variables are restored in this cont() call.
    restored_variables = set()

    # Keep track of which variables are "touched" (i.e., possibly updated) in
    # this cont() call.
    touched_variables = set()

    # =========================================================================
    # Use a non-recursive method to trace the inputs from the node and set up
    # the feeds.
    fetched = self._sess.graph.as_graph_element(target_name)
    elem_stack = [fetched]
    done = set()

    while elem_stack:
      curr_elem = elem_stack.pop()
      curr_node = self._get_node(curr_elem)

      done.add(curr_node.name)

      non_control_inputs = [inp for inp in curr_node.inputs]
      control_inputs = [inp for inp in curr_node.control_inputs]
      all_inputs = set(non_control_inputs + control_inputs)

      # Iterate through the (non-control) inputs.
      for inp in all_inputs:
        # Determine whether the input is feedable. Reference-type tensors,
        # e.g., Variables, should not be fed, because they can change.
        if isinstance(inp, ops.Tensor):
          is_inp_ref = inp.dtype._is_ref_dtype   # pylint: disable=protected-access
          can_feed = self._sess.graph.is_feedable(inp) and not is_inp_ref
        else:
          is_inp_ref = False
          can_feed = False

        if (restore_variable_values and inp.name in self._dirty_variables and
            inp.name not in restored_variables and
            inp.name not in touched_variables):
          # Do not restore Variables touched or restored previously in this
          # cont() call.
          initializer_op = self._variable_initializers[inp.name]
          initial_value_tensor = self._variable_initial_values[inp.name]
          self._sess.run(initializer_op,
                         feed_dict={
                             initial_value_tensor:
                                 self._cached_variable_values[inp.name]
                         })

          # Mark the variable as restored.
          restored_variables.add(inp.name)

        # Determine if this is a reference-type input from a variable, and
        # the recipient node is not Identity. In that case, the Variable
        # needs to be marked as dirty and its current value recorded, due to
        # the fact that the receiving op may mutate the value of the Variable.
        if (is_inp_ref and inp.op.type == "Variable" and
            curr_node.type != "Identity"):
          # Mark the variable as dirty.
          touched_variables.add(inp.name)

          # Obtain the old value of the variable and cache it.
          if inp.name not in self._cached_variable_values:
            old_value = self._sess.run(inp)
            self._cached_variable_values[inp.name] = old_value

        # N.B.: The order of the logical branches matters. For example,
        # _client_feed_dict comes after _tensor_handles, so that tensor
        # handles stored in cont() calls can override the original client
        # feeds. Also for example, _override_tensors comes the first, so
        # the manual overriding, if exists, can always take effect.
        if use_overrides and can_feed and inp.name in self._override_tensors:
          # Use client-supplied overriding tensor value.
          feeds[inp] = self._override_tensors[inp.name]
          self._last_feed_types[inp.name] = self.FEED_TYPE_OVERRIDE
        elif (use_tensor_handles and can_feed and
              inp.name in self._tensor_handles and inp not in feeds):
          # Tensor handle found in cache.
          feeds[inp] = self._tensor_handles[inp.name].eval()
          self._last_feed_types[inp.name] = self.FEED_TYPE_HANDLE
        elif inp in self._client_feed_dict:
          # This input is available in the client feed_dict.
          feeds[inp] = self._client_feed_dict[inp]
          self._last_feed_types[inp.name] = self.FEED_TYPE_CLIENT
        else:
          # There is no feed available for this input. So keep tracing its
          # input(s).
          inp_node = self._get_node(inp)
          if inp_node.name in done:
            # Already visited.
            continue

          elem_stack.append(inp)
          done.add(inp_node.name)

    # =========================================================================

    if touched_variables:
      self._dirty_variables.update(touched_variables)

    for variable in restored_variables:
      self._dirty_variables.remove(variable)

    # Prepare RunOptions for DebugTensorWatches
    run_options = config_pb2.RunOptions()
    # TODO(cais): Add fields for watching intermediate tensors.

    if isinstance(fetched, ops.Operation):
      # The fetched is an Operation: Will not get tensor handle.
      self._sess.run(fetched, feed_dict=feeds, options=run_options)
      # No return value for a run of an Operation
    else:
      # This is a Tensor: Will get tensor handle and cache it.
      target_handle = self._sess.run(session_ops.get_session_handle(fetched),
                                     feed_dict=feeds,
                                     options=run_options)
      self._tensor_handles[target_name] = target_handle

      return target_handle.eval()

    # Invalidate caches at the end.
    for touched_variable in touched_variables:
      self._invalidate_transitively_outgoing_cache(touched_variable)

  def _invalidate_transitively_outgoing_cache(self, source_element):
    """Invalidate the cached tensor handles by tracing output.

    This method is used to invalidate caches such as cached TensorHandles
    and intermediate tensor values when Variable mutation happens or when
    client overrides tensor values.

    Uses non-recursive implementation to avoid stack overflow on deep networks.

    TODO(cais): Currently, only TensorHandle caches are invalidated. Invalidate
      cached intermediate tensor values from dumps when dumps are added.

    Args:
      source_element: The source graph element (e.g., a Variable output slot)
        to trace the output from.
    """

    if not self._tensor_handles:
      return

    # First, use cached invalidation paths to eliminate some cached tensor
    # handles.
    for handle_name in self._tensor_handles:
      if (handle_name in self._cached_invalidation_path and
          source_element in self._cached_invalidation_path[handle_name]):
        del self._tensor_handles[handle_name]

    if not self._tensor_handles:
      return

    stack = [source_element]
    done = set()

    while stack:
      curr_element = stack.pop()

      done.add(curr_element)

      if curr_element in self._tensor_handles:
        # Cache the invalidation path for potential future use.
        if curr_element not in self._cached_invalidation_path:
          self._cached_invalidation_path[curr_element] = set([source_element])
        else:
          self._cached_invalidation_path[curr_element].add(source_element)

        del self._tensor_handles[curr_element]

      targets = self._non_control_output_targets.get(curr_element, [])
      for target in targets:
        if target in done:
          continue
        else:
          stack.append(target)

  def finalize(self):
    """Run the final fetch(es).

    Restore the dirty variables; ignore the client-supplied overriding tensor
    values.

    Returns:
      The same return value as self.cont() as called on the final fetch.
    """

    return self.cont(
        self._fetch,
        use_tensor_handles=False,
        use_overrides=False,
        restore_variable_values=True)

  def handle_names(self):
    """Return names of the TensorHandles that the debugger is holding.

    Returns:
      (list of str) Name of the tensors for which TensorHandle is available.
    """
    return [name for name in self._tensor_handles]

  def dirty_variables(self):
    """Get the set of variables that are currently "dirty".

    "dirty" means:
      previous cont() calls have updated the value of the Variable,
      and the Variable's old value (the value before any cont() calls
      happened) was not restored.

    Returns:
      (set) A set of dirty variables.
    """

    return self._dirty_variables

  def get_tensor_value(self, tensor_name):
    """Get the value of a tensor that the stepper has access to.

    Args:
      tensor_name: (str) Name of the tensor.

    Returns:
      Value of the tensor, from overriding values or cached tensor handles.

    Raises:
      ValueError: If the value is not available as an overriding value
        or through a TensorHandle.
    """

    if tensor_name in self._override_tensors:
      return self._override_tensors[tensor_name]
    elif tensor_name in self._tensor_handles:
      return self._tensor_handles[tensor_name].eval()
    else:
      raise ValueError(
          "This stepper instance does not have access to the value of "
          "tensor \"%s\"" % tensor_name)

  def get_fetch_result(self):
    return self.get_tensor_value(self._fetch_name)

  def override_names(self):
    """Return names of the TensorHandles that the debugger is holding.

    Returns:
      (list of str) Name of the tensor for which overriding tensor values are
        available.
    """
    return [name for name in self._override_tensors]

  def _get_node(self, element):
    """Get the node of a graph element.

    Args:
      element: A graph element (Op, Tensor or Node)

    Returns:
      The node associated with element in the graph.
    """

    node_name, _ = debug_data.parse_node_or_tensor_name(element.name)
    return self._sess.graph.as_graph_element(node_name)
