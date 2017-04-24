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

import copy
import os
import shutil
import tempfile
import time

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import session_ops


# TODO(cais): Use nest.flatten once it handles nest Dicts correctly.
def _flatten_fetches(fetches):
  """Flatten list, tuple of fetches, or a single fetch into a list of fetches.

  Args:
    fetches: The fetches to flatten: Can be a single Tensor, Op, or a
      potentially nested list, tuple or dict of such individual fetches.

  Returns:
    The fetches flattened to a list.
  """

  flattened = []
  if isinstance(fetches, (list, tuple)):
    for fetch in fetches:
      flattened.extend(_flatten_fetches(fetch))
  elif isinstance(fetches, dict):
    for key in fetches:
      flattened.extend(_flatten_fetches(fetches[key]))
  else:
    flattened.append(fetches)

  return flattened


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
    (1) Overriding (injected) values from the client.
    (2) TensorHandles from previous cont() calls.
    (3) Dumped intermediate Tensors from previous cont() calls.
    (4) Feeds supplied during the construction of the stepper instance.

  During the cont() call, intermediate Tensors are dumped to temporary
  directories. The dumped Tensor values will be used in subsequent cont() calls
  when they are required as data dependencies.

  The temporary directories are automatically clean when the NodeStepper
  instance exits as a context mananger.

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
    d = tf.multiply(a, c, name="d")

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
  FEED_TYPE_DUMPED_INTERMEDIATE = "dumped_intermediate"

  def __init__(self, sess, fetches, feed_dict=None):
    """Constructor for Debugger.

    Args:
      sess: (Session) the TensorFlow Session to step in.
      fetches: Same as the fetches input argument to `Session.run()`.
      feed_dict: Same as the feed_dict input argument to `Session.run()`.
    """

    self._sess = sess

    self._fetches = fetches
    flattened_fetches = _flatten_fetches(fetches)

    self._fetch_names, self._fetch_list = self._get_fetch_and_name_lists(
        flattened_fetches)

    # A map from Variable name to initializer op.
    self._variable_initializers = {}

    # A map from Variable name to initial value, used when overriding or
    # restoring Variable values.
    self._variable_initial_values = {}

    # Initialize the map for output recipients (targets).
    self._output_targets = {}

    # Sorted transitive closure of the fetched node.
    # We also collect the list of the names of the reference-type Tensors,
    # because we later need to avoid using intermediate dumps for such Tensors.
    (self._sorted_nodes,
     self._closure_elements,
     self._ref_tensor_names) = self._dfs_visit(self._sess.graph,
                                               self._fetch_list)

    self._transitive_closure_set = set(self._sorted_nodes)

    # A map from Variable name to the old values (before any cont() calls).
    self._cached_variable_values = {}

    # A cache map from tensor name to what variables may invalidate the tensor
    self._cached_invalidation_path = {}

    # Keep track of which variables are in a dirty state.
    self._dirty_variables = set()

    # Variables updated in the last cont() call.
    self._last_updated = None

    # Cached tensor handles: a dict with keys as tensor names and values as
    # tensor handles.
    self._tensor_handles = {}

    # Cached intermediate tensor values: a dict mapping tensor names to
    # DebugTensorDatum.
    self._dumped_intermediate_tensors = {}
    self._dump_session_root = tempfile.mkdtemp(prefix="tfdbg_stepper_")

    # Feed dict from the client.
    self._client_feed_dict = {}
    if feed_dict:
      for key in feed_dict:
        if isinstance(key, ops.Tensor):
          self._client_feed_dict[key.name] = feed_dict[key]
        else:
          self._client_feed_dict[key] = feed_dict[key]

    # Overriding tensor values.
    self._override_tensors = {}

    # What the feed types were used by the last cont() call.
    self._last_feed_types = {}

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    if os.path.isdir(self._dump_session_root):
      shutil.rmtree(self._dump_session_root)

  def _get_fetch_and_name_lists(self, flattened_fetches):
    """Get the lists of fetches and their names.

    Args:
      flattened_fetches: A list of fetches or their names. Can mix fetches and
        names.

    Returns:
      (list of str): A list of the names of the fetches.
      (list): A list of the fetches.
    """

    fetch_names = []
    fetch_list = []
    for fetch in flattened_fetches:
      if isinstance(fetch, six.string_types):
        fetch_names.append(fetch)
        fetch_list.append(self._sess.graph.as_graph_element(fetch))
      else:
        fetch_names.append(fetch.name)
        fetch_list.append(fetch)

    return fetch_names, fetch_list

  def _dfs_visit(self, graph, elem_list):
    """Trace back the input of a graph element, using depth-first search.

    Uses non-recursive implementation to prevent stack overflow for deep
    graphs.

    Also performs the following action(s):
      1) When encountering a Variable, obtain its initializer op, to
         facilitate possible subsequent restoration / overriding of variable
         value.

    Args:
      graph: A TF graph instance.
      elem_list: list of graph elements: a Tensor or an Operation.

    Returns:
      (list of str) A topologically-sorted list of all nodes (not tensors)
        in the transitive closure of elem_list. Obviously, the topological sort
         is not unique in general. The return value here is just an arbitrary
         one of potentially many possible topological sorts.
      (list of str) A list of all graph elements (nodes and/or tensors) in the
        transitive closure.
    """

    # These set should hold only strings, i.e, names of the nodes.
    done = set()  # Keep track of visited graph elements.

    # A list of str: Names of the topologically-sorted graph elements.
    node_inputs = dict()  # New: Input map of nodes in the transitive closure.

    elem_stack = copy.copy(elem_list)

    # Graph elements in the transitive closure, including the nodes and tensors.
    closure_elements = [elem.name for elem in elem_list]

    ref_tensor_names = set()
    for element in elem_list:
      if isinstance(element, ops.Tensor) and element.dtype._is_ref_dtype:  # pylint: disable=protected-access
        ref_tensor_names.add(element.name)

    while elem_stack:
      curr_elem = elem_stack.pop()
      curr_node = self._get_node(curr_elem)

      done.add(curr_node.name)

      non_control_inputs = [inp for inp in curr_node.inputs]
      control_inputs = [inp for inp in curr_node.control_inputs]
      all_inputs = set(non_control_inputs + control_inputs)

      if curr_node.name not in node_inputs:
        all_input_nodes = set()
        for inp in all_inputs:
          all_input_nodes.add(self._get_node(inp).name)
        node_inputs[curr_node.name] = all_input_nodes

      # Iterate through the (non-control) inputs.
      for inp in all_inputs:
        # Set up the non-control output map.
        # if is_non_control_input:
        if inp.name not in self._output_targets:
          self._output_targets[inp.name] = set([curr_elem.name])
        else:
          self._output_targets[inp.name].add(curr_elem.name)

        if (isinstance(inp, ops.Tensor) and
            inp.op.type in ["Variable", "VariableV2"] and
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
        closure_elements.append(inp.name)
        if isinstance(inp, ops.Tensor) and inp.dtype._is_ref_dtype:  # pylint: disable=protected-access
          ref_tensor_names.add(inp.name)

    # Now that we have traversed the transitive closure and obtained the
    # node-input map, we can topologically sort them.
    sorted_nodes = []
    stack = []
    for node in node_inputs:
      if not node_inputs[node]:
        stack.append(node)
    for node in stack:
      del node_inputs[node]

    while stack:
      curr_node = stack.pop()
      sorted_nodes.append(curr_node)

      # Iterate through the node-input map and remove the child.
      pushes = []
      for node in node_inputs:
        if curr_node in node_inputs[node]:
          node_inputs[node].remove(curr_node)
          if not node_inputs[node]:
            pushes.append(node)

      # Delete new pushes from node-input map.
      for node in pushes:
        del node_inputs[node]

      stack.extend(pushes)

    return sorted_nodes, closure_elements, ref_tensor_names

  def sorted_nodes(self):
    """Get a topologically-sorted list of node names of the stepper.

    These are the names of the nodes (i.e., not Tensors) in the transitive
    closure of the stepper, in a topologically-sorted order.

    Returns:
      (list of str): Sorted transitive inputs to the fetch of the stepper
        instance. The fetch itself is included in the list.
    """

    return self._sorted_nodes

  def closure_elements(self):
    """Get a name list of the graph elements of the stepper.

    Returns:
      (list of str): names of the graph elements (i.e., nodes and tensors) in
    the transitive closure of the stepper, in a random order.
    """

    return self._closure_elements

  def output_slots_in_closure(self, node_name):
    """Get the output tensors in the transitive closure from node.

    Args:
      node_name: (str) Name of the node in question.

    Returns:
      (list of int) Output slots of the output tensors of the node that are in
        the transitive closure of the stepper.
    """

    node = self._sess.graph.as_graph_element(node_name)

    tensor_slots = []
    for i, _ in enumerate(node.outputs):
      tensor_name = node_name + ":%d" % i
      if tensor_name in self._closure_elements:
        tensor_slots.append(i)

    return tensor_slots

  def is_feedable(self, name):
    """Determine if a graph element if feedable.

    Args:
      name: (str) name of the graph element (Tensor or Operation)

    Returns:
      (bool) whether the graph element is feedable.
    """

    if not isinstance(name, six.string_types):
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

    if not isinstance(tensor_name, six.string_types):
      raise TypeError("Expected type str; got type %s" % type(tensor_name))

    node_name = self._get_node_name(tensor_name)
    if node_name not in self._transitive_closure_set:
      raise ValueError(
          "Cannot override tensor \"%s\" because it does not exist in the "
          "input tree to the fetch \"%s\"" %
          (tensor_name, repr(self._fetch_names)))

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
           use_dumped_intermediates=True,
           use_overrides=True,
           invalidate_from_updated_variables=False,
           restore_variable_values=False):
    """Continue till the completion of the specified target tensor.

    Args:
      target: A single fetched Tensor or Op, or a name (str) representing the
        Tensor or Op. In the case of a name str, the graph will be searched
        to find the corresponding Tensor or Op.
        # TODO(cais): Support multiple fetches as in Session.run() interface.
      use_tensor_handles: (bool) Whether this cont() run will use cached tensor
        handles to avoid recomputation. Default: True.
      use_dumped_intermediates: (bool) Whether this cont() call will use dumped
        intermediate tensors to avoid recomputation.
      use_overrides: (bool) Whether the overriding tensor values supplied by
        the client are to be used in this cont() call. Default: True.
      invalidate_from_updated_variables: (bool) Whether to invalidate the
        tensor handles and intermediate tensor handles affected by the
        Variable updates that happen in this cont() call.
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

    if isinstance(target, six.string_types):
      # Fetch target is a string. Assume it is the name of the Tensor or Op and
      # will attempt to find it in the Session's graph.
      target_name = target
    else:
      target_name = target.name

    graph_element = self._sess.graph.as_graph_element(target_name)
    # Any additional tensor handles to obtain in this cont() action.
    additional_handle_requests = []

    if (isinstance(graph_element, ops.Tensor) and
        graph_element.op.type == "Placeholder"):
      self._last_feed_types[graph_element.name] = self.FEED_TYPE_CLIENT
      return self._client_feed_dict[graph_element.name]
    elif (isinstance(graph_element, ops.Operation) and
          graph_element.type == "Placeholder"):
      tensor_name = graph_element.name + ":0"
      self._last_feed_types[tensor_name] = self.FEED_TYPE_CLIENT
      return self._client_feed_dict[tensor_name]

    if isinstance(graph_element, ops.Operation) and graph_element.outputs:
      # Check if this op has any output tensors that also fall into this
      # stepper's transitive closure.
      node_outputs = [
          output.name for output in graph_element.outputs
          if output.name in self._closure_elements
      ]
      if node_outputs:
        # The target is an op with at least one output within the transitive
        # closure. The cont() action will amount to using the 0-th
        # output Tensor as the target, as well as obtaining handles to it
        # and to the rest of the outputs tensors in the transitive closure
        # (if any).
        target_name = node_outputs[0]
        additional_handle_requests = node_outputs[1:]

    # Verify that the target is in the transitive closure of the stepper's
    # fetch.
    target_node_name = self._get_node_name(target_name)
    if target_node_name not in self._transitive_closure_set:
      raise ValueError(
          "Target \"%s\" is not in the transitive closure for the fetch of the "
          "stepper: \"%s\"." % (target_name, repr(self._fetch_names)))

    # Check if a cached tensor handle can be used on the fetch directly.
    if use_tensor_handles and target_name in self._tensor_handles:
      self._last_feed_types[target_name] = self.FEED_TYPE_HANDLE
      return self._tensor_handles[target_name].eval()

    # Check if a dumped intermediate tensor can be used on the fetch directly.
    if (use_dumped_intermediates and
        target_name in self._dumped_intermediate_tensors):
      self._last_feed_types[target_name] = self.FEED_TYPE_DUMPED_INTERMEDIATE
      return self._dumped_intermediate_tensors[target_name].get_tensor()

    # Check if an overriding tensor value can be used directly.
    if use_overrides and target_name in self._override_tensors:
      # Override is available. Return the value right away.
      self._last_feed_types[target_name] = self.FEED_TYPE_OVERRIDE
      return self._override_tensors[target_name]

    # Keep track of which variables are restored in this cont() call.
    restored_variables = set()

    # Keep track of which variables are "touched" (i.e., possibly updated) in
    # this cont() call.
    self._last_updated = set()

    # =========================================================================
    # Use a non-recursive method to trace the inputs from the node and set up
    # the feeds.
    feeds = {}  # The feeds to be used in the Session.run() call.
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
          is_inp_ref = inp.dtype._is_ref_dtype  # pylint: disable=protected-access
          can_feed = self._sess.graph.is_feedable(inp) and not is_inp_ref
        else:
          is_inp_ref = False
          can_feed = False

        if (restore_variable_values and inp.name in self._dirty_variables and
            inp.name not in restored_variables and
            inp.name not in self._last_updated):
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
        if (is_inp_ref and inp.op.type in ["Variable", "VariableV2"] and
            curr_node.type != "Identity"):
          # Mark the variable as dirty.
          self._last_updated.add(inp.name)

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
        elif (can_feed and inp not in feeds and
              use_tensor_handles and inp.name in self._tensor_handles):
          # Tensor handle found in cache.
          feeds[inp] = self._tensor_handles[inp.name]
          self._last_feed_types[inp.name] = self.FEED_TYPE_HANDLE
        elif (can_feed and inp not in feeds and
              use_dumped_intermediates and
              inp.name in self._dumped_intermediate_tensors):
          # Dumped intermediate Tensor found.
          feeds[inp] = self._dumped_intermediate_tensors[inp.name].get_tensor()
          self._last_feed_types[inp.name] = self.FEED_TYPE_DUMPED_INTERMEDIATE
        elif inp.name in self._client_feed_dict:
          # This input is available in the client feed_dict.
          feeds[inp] = self._client_feed_dict[inp.name]
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

    if self._last_updated:
      self._dirty_variables.update(self._last_updated)

    for variable in restored_variables:
      self._dirty_variables.remove(variable)

    (dump_path,
     run_options) = self._prepare_cont_call_dump_path_and_run_options()
    if isinstance(fetched, ops.Operation):
      # The fetched is an Operation: Will not get tensor handle.
      self._sess.run(fetched, feed_dict=feeds, options=run_options)
      return_value = None
    else:
      # This is a Tensor: Will get tensor handle and cache it.
      # Will also get the additional requested tensor handles (if any).
      tensors_to_get_handles_for = [fetched]
      handle_names = [target_name]

      tensors_to_get_handles_for.extend([
          self._sess.graph.as_graph_element(h)
          for h in additional_handle_requests
      ])
      handle_names.extend(additional_handle_requests)

      handles = self._sess.run(
          [session_ops.get_session_handle(tensor) for tensor in
           tensors_to_get_handles_for],
          feed_dict=feeds,
          options=run_options)
      for handle_name, handle in zip(handle_names, handles):
        self._tensor_handles[handle_name] = handle

      return_value = self._tensor_handles[target_name].eval()

    self._load_dumped_intermediate_tensors(dump_path, target_name)

    if invalidate_from_updated_variables:
      # Invalidate caches at the end.
      for last_updated_variable in self._last_updated:
        self._invalidate_transitively_outgoing_cache(last_updated_variable)

    return return_value

  def _prepare_cont_call_dump_path_and_run_options(self):
    """Prepare the dump path and RunOptions for next cont() call.

    Returns:
      dump_path: (str) Directory path to which the intermediate tensor will be
        dumped.
      run_options: (config_pb2.RunOptions) The RunOptions containing the tensor
        watch options for this graph.
    """
    run_options = config_pb2.RunOptions()
    dump_path = self._cont_call_dump_path()
    for element_name in self._closure_elements:
      if ":" in element_name:
        debug_utils.add_debug_tensor_watch(
            run_options,
            debug_data.get_node_name(element_name),
            output_slot=debug_data.get_output_slot(element_name),
            debug_urls=["file://" + dump_path])

    return dump_path, run_options

  def _cont_call_dump_path(self):
    return os.path.join(self._dump_session_root,
                        "cont_%d" % int(time.time() * 1e6))

  def _load_dumped_intermediate_tensors(self, dump_path, target_name):
    dump_dir = debug_data.DebugDumpDir(dump_path, validate=False)
    for dump in dump_dir.dumped_tensor_data:
      if (dump.tensor_name not in self._ref_tensor_names and
          dump.tensor_name not in self._tensor_handles and
          dump.tensor_name not in self._override_tensors and
          dump.tensor_name != target_name):
        self._dumped_intermediate_tensors[dump.tensor_name] = dump

  def _get_node_name(self, graph_element_name):
    return graph_element_name.split(":")[0]

  def _invalidate_transitively_outgoing_cache(self, source_element):
    """Invalidate the cached tensor handles by tracing output.

    This method is used to invalidate caches such as cached TensorHandles
    and intermediate tensor values when Variable mutation happens or when
    client overrides tensor values.

    Uses non-recursive implementation to avoid stack overflow on deep networks.

    Args:
      source_element: The source graph element (e.g., a Variable output slot)
        to trace the output from.
    """

    if not self._tensor_handles and not self._dumped_intermediate_tensors:
      return

    # First, use cached invalidation paths to eliminate some cached tensor
    # handles and intermediate tensors.
    to_delete_handles = []
    for handle_name in self._tensor_handles:
      if (handle_name in self._cached_invalidation_path and
          source_element in self._cached_invalidation_path[handle_name]):
        to_delete_handles.append(handle_name)
    for handle_name in to_delete_handles:
      del self._tensor_handles[handle_name]

    to_delete_intermediates = []
    for intm_tensor_name in self._dumped_intermediate_tensors:
      if (intm_tensor_name in self._cached_invalidation_path and
          source_element in self._cached_invalidation_path[intm_tensor_name]):
        to_delete_intermediates.append(intm_tensor_name)
    for intermediate in to_delete_intermediates:
      del self._dumped_intermediate_tensors[intermediate]

    if not self._tensor_handles and not self._dumped_intermediate_tensors:
      return

    stack = [source_element]
    done = set()

    while stack:
      curr_element = stack.pop()
      done.add(curr_element)

      if (curr_element in self._tensor_handles or
          curr_element in self._dumped_intermediate_tensors):
        # Cache the invalidation path for potential future use.
        if curr_element not in self._cached_invalidation_path:
          self._cached_invalidation_path[curr_element] = set([source_element])
        else:
          self._cached_invalidation_path[curr_element].add(source_element)

        if curr_element in self._tensor_handles:
          del self._tensor_handles[curr_element]
        else:
          del self._dumped_intermediate_tensors[curr_element]

      targets = self._output_targets.get(curr_element, [])
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

    self.restore_variable_values()
    return self._sess.run(self._fetches, feed_dict=self._client_feed_dict)

  def restore_variable_values(self):
    """Restore variables to the initial values.

    "Initial value" refers to the value when this NodeStepper instance was
    first constructed.
    """

    for var_name in self._dirty_variables:
      self._sess.run(self._variable_initializers[var_name],
                     feed_dict={
                         self._variable_initial_values[var_name]:
                             self._cached_variable_values[var_name]
                     })

  def handle_names(self):
    """Return names of the TensorHandles that the debugger is holding.

    Returns:
      (list of str) Name of the tensors for which TensorHandle is available.
    """

    return [name for name in self._tensor_handles]

  def handle_node_names(self):
    """Get list of names of the nodes for which handles are available.

    Returns:
      (set of str) List of names of the nodes.
    """

    return set([self._get_node_name(name) for name in self._tensor_handles])

  def intermediate_tensor_names(self):
    """Get list of the names of the Tensors for which dumps are available.

    Returns:
      (list of str) List of the names of the Tensors for which intermediate
        dumps are available.
    """

    return self._dumped_intermediate_tensors.keys()

  def last_updated(self):
    """Get the names of the variables updated in the last cont() call.

    Returns:
      A set of the variable names updated in the previous cont() call.
      If no cont() call has occurred before, returns None.
    """

    return self._last_updated

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

  def is_placeholder(self, graph_element_name):
    """Check whether a graph element is a Placeholder, by name.

    Args:
      graph_element_name: (str) Name of the tensor or op to be tested.

    Returns:
      (bool) Whether the graph element of the specified name is a Placeholder
        op or the output Tensor of a Placeholder op.

    Raises:
      ValueError: If graph_element_name is not in the transitive closure of the
        stepper instance.
    """

    node_name = self._get_node_name(graph_element_name)
    if node_name not in self.sorted_nodes():
      raise ValueError(
          "%s is not in the transitive closure of this NodeStepper "
          "instance" % graph_element_name)

    graph_element = self._sess.graph.as_graph_element(graph_element_name)
    if not isinstance(graph_element, ops.Operation):
      graph_element = graph_element.op
    return graph_element.type == "Placeholder"

  def placeholders(self):
    """Get the list of Placeholder Tensors in the transitive closure.

    Returns:
      (list of str) A list of Placeholder Tensors or ops in the transitive
        closure.
    """

    placeholders = []
    for item in self.sorted_nodes():
      if self.is_placeholder(item):
        placeholders.append(item)

    return placeholders

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

    if self.is_placeholder(tensor_name):
      if ":" not in tensor_name:
        tensor_name += ":0"
      return self._client_feed_dict[tensor_name]
    elif tensor_name in self._override_tensors:
      return self._override_tensors[tensor_name]
    elif tensor_name in self._tensor_handles:
      return self._tensor_handles[tensor_name].eval()
    elif tensor_name in self._dumped_intermediate_tensors:
      return self._dumped_intermediate_tensors[tensor_name].get_tensor()
    else:
      raise ValueError(
          "This stepper instance does not have access to the value of "
          "tensor \"%s\"" % tensor_name)

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
