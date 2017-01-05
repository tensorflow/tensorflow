<!-- This file is machine generated: DO NOT EDIT! -->

# TensorFlow Debugger
[TOC]

Public Python API of TensorFlow Debugger (tfdbg).

## Functions for adding debug watches

These functions help you modify `RunOptions` to specify which `Tensor`s are to
be watched when the TensorFlow graph is executed at runtime.

- - -

### `tf_debug.add_debug_tensor_watch(run_options, node_name, output_slot=0, debug_ops='DebugIdentity', debug_urls=None)` {#add_debug_tensor_watch}

Add watch on a `Tensor` to `RunOptions`.

N.B.: Under certain circumstances, the `Tensor` may not be actually watched
  (e.g., if the node of the `Tensor` is constant-folded during runtime).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`node_name`</b>: (`str`) name of the node to watch.
*  <b>`output_slot`</b>: (`int`) output slot index of the tensor from the watched node.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s). Can be a
    `list` of `str` or a single `str`. The latter case is equivalent to a
    `list` of `str` with only one element.
*  <b>`debug_urls`</b>: (`str` or `list` of `str`) URL(s) to send debug values to,
    e.g., `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.


- - -

### `tf_debug.watch_graph(run_options, graph, debug_ops='DebugIdentity', debug_urls=None, node_name_regex_whitelist=None, op_type_regex_whitelist=None)` {#watch_graph}

Add debug watches to `RunOptions` for a TensorFlow graph.

To watch all `Tensor`s on the graph, let both `node_name_regex_whitelist`
and `op_type_regex_whitelist` be the default (`None`).

N.B.: Under certain circumstances, not all specified `Tensor`s will be
  actually watched (e.g., nodes that are constant-folded during runtime will
  not be watched).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`graph`</b>: An instance of `ops.Graph`.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
*  <b>`debug_urls`</b>: URLs to send debug values to. Can be a list of strings,
    a single string, or None. The case of a single string is equivalent to
    a list consisting of a single string, e.g., `file:///tmp/tfdbg_dump_1`,
    `grpc://localhost:12345`.
*  <b>`node_name_regex_whitelist`</b>: Regular-expression whitelist for node_name,
    e.g., `"(weight_[0-9]+|bias_.*)"`
*  <b>`op_type_regex_whitelist`</b>: Regular-expression whitelist for the op type of
    nodes, e.g., `"(Variable|Add)"`.
    If both `node_name_regex_whitelist` and `op_type_regex_whitelist`
    are set, the two filtering operations will occur in a logical `AND`
    relation. In other words, a node will be included if and only if it
    hits both whitelists.


- - -

### `tf_debug.watch_graph_with_blacklists(run_options, graph, debug_ops='DebugIdentity', debug_urls=None, node_name_regex_blacklist=None, op_type_regex_blacklist=None)` {#watch_graph_with_blacklists}

Add debug tensor watches, blacklisting nodes and op types.

This is similar to `watch_graph()`, but the node names and op types are
blacklisted, instead of whitelisted.

N.B.: Under certain circumstances, not all specified `Tensor`s will be
  actually watched (e.g., nodes that are constant-folded during runtime will
  not be watched).

##### Args:


*  <b>`run_options`</b>: An instance of `config_pb2.RunOptions` to be modified.
*  <b>`graph`</b>: An instance of `ops.Graph`.
*  <b>`debug_ops`</b>: (`str` or `list` of `str`) name(s) of the debug op(s) to use.
*  <b>`debug_urls`</b>: URL(s) to send ebug values to, e.g.,
    `file:///tmp/tfdbg_dump_1`, `grpc://localhost:12345`.
*  <b>`node_name_regex_blacklist`</b>: Regular-expression blacklist for node_name.
    This should be a string, e.g., `"(weight_[0-9]+|bias_.*)"`.
*  <b>`op_type_regex_blacklist`</b>: Regular-expression blacklist for the op type of
    nodes, e.g., `"(Variable|Add)"`.
    If both node_name_regex_blacklist and op_type_regex_blacklist
    are set, the two filtering operations will occur in a logical `OR`
    relation. In other words, a node will be excluded if it hits either of
    the two blacklists; a node will be included if and only if it hits
    neither of the blacklists.




## Classes for debug-dump data and directories

These classes allow you to load and inspect tensor values dumped from
TensorFlow graphs during runtime.

- - -

### `class tf_debug.DebugTensorDatum` {#DebugTensorDatum}

A single tensor dumped by TensorFlow Debugger (tfdbg).

Contains metadata about the dumped tensor, including `timestamp`,
`node_name`, `output_slot`, `debug_op`, and path to the dump file
(`file_path`).

This type does not hold the generally space-expensive tensor value (numpy
array). Instead, it points to the file from which the tensor value can be
loaded (with the `get_tensor` method) if needed.
- - -

#### `tf_debug.DebugTensorDatum.__init__(dump_root, debug_dump_rel_path)` {#DebugTensorDatum.__init__}

`DebugTensorDatum` constructor.

##### Args:


*  <b>`dump_root`</b>: (`str`) Debug dump root directory.
*  <b>`debug_dump_rel_path`</b>: (`str`) Path to a debug dump file, relative to the
      `dump_root`. For example, suppose the debug dump root
      directory is `/tmp/tfdbg_1` and the dump file is at
      `/tmp/tfdbg_1/ns_1/node_a_0_DebugIdentity_123456789`, then
      the value of the debug_dump_rel_path should be
      `ns_1/node_a_0_DebugIdenity_1234456789`.

##### Raises:


*  <b>`ValueError`</b>: If the base file name of the dump file does not conform to
    the dump file naming pattern:
    `node_name`_`output_slot`_`debug_op`_`timestamp`


- - -

#### `tf_debug.DebugTensorDatum.__repr__()` {#DebugTensorDatum.__repr__}




- - -

#### `tf_debug.DebugTensorDatum.__str__()` {#DebugTensorDatum.__str__}




- - -

#### `tf_debug.DebugTensorDatum.debug_op` {#DebugTensorDatum.debug_op}

Name of the debug op.

##### Returns:

  (`str`) debug op name (e.g., `DebugIdentity`).


- - -

#### `tf_debug.DebugTensorDatum.dump_size_bytes` {#DebugTensorDatum.dump_size_bytes}

Size of the dump file.

Unit: byte.

##### Returns:

  If the dump file exists, size of the dump file, in bytes.
  If the dump file does not exist, None.


- - -

#### `tf_debug.DebugTensorDatum.file_path` {#DebugTensorDatum.file_path}

Path to the file which stores the value of the dumped tensor.


- - -

#### `tf_debug.DebugTensorDatum.get_tensor()` {#DebugTensorDatum.get_tensor}

Get tensor from the dump (`Event`) file.

##### Returns:

  The tensor loaded from the dump (`Event`) file.


- - -

#### `tf_debug.DebugTensorDatum.node_name` {#DebugTensorDatum.node_name}

Name of the node from which the tensor value was dumped.

##### Returns:

  (`str`) name of the node watched by the debug op.


- - -

#### `tf_debug.DebugTensorDatum.output_slot` {#DebugTensorDatum.output_slot}

Output slot index from which the tensor value was dumped.

##### Returns:

  (`int`) output slot index watched by the debug op.


- - -

#### `tf_debug.DebugTensorDatum.tensor_name` {#DebugTensorDatum.tensor_name}

Name of the tensor watched by the debug op.

##### Returns:

  (`str`) `Tensor` name, in the form of `node_name`:`output_slot`


- - -

#### `tf_debug.DebugTensorDatum.timestamp` {#DebugTensorDatum.timestamp}

Timestamp of when this tensor value was dumped.

##### Returns:

  (`int`) The timestamp in microseconds.


- - -

#### `tf_debug.DebugTensorDatum.watch_key` {#DebugTensorDatum.watch_key}

Watch key identities a debug watch on a tensor.

##### Returns:

  (`str`) A watch key, in the form of `tensor_name`:`debug_op`.



- - -

### `class tf_debug.DebugDumpDir` {#DebugDumpDir}

Data set from a debug-dump directory on filesystem.

An instance of `DebugDumpDir` contains all `DebugTensorDatum` instances
in a tfdbg dump root directory.
- - -

#### `tf_debug.DebugDumpDir.__init__(dump_root, partition_graphs=None, validate=True)` {#DebugDumpDir.__init__}

`DebugDumpDir` constructor.

##### Args:


*  <b>`dump_root`</b>: (`str`) path to the dump root directory.
*  <b>`partition_graphs`</b>: A repeated field of GraphDefs representing the
      partition graphs executed by the TensorFlow runtime.
*  <b>`validate`</b>: (`bool`) whether the dump files are to be validated against the
      partition graphs.

##### Raises:


*  <b>`IOError`</b>: If dump_root does not exist as a directory.


- - -

#### `tf_debug.DebugDumpDir.debug_watch_keys(node_name)` {#DebugDumpDir.debug_watch_keys}

Get all tensor watch keys of given node according to partition graphs.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node.

##### Returns:

  (`list` of `str`) all debug tensor watch keys. Returns an empty list if
    the node name does not correspond to any debug watch keys.

##### Raises:

  `LookupError`: If debug watch information has not been loaded from
    partition graphs yet.


- - -

#### `tf_debug.DebugDumpDir.devices()` {#DebugDumpDir.devices}

Get the list of devices.

##### Returns:

  (`list` of `str`) names of the devices.

##### Raises:


*  <b>`LookupError`</b>: If node inputs and control inputs have not been loaded
     from partition graphs yet.


- - -

#### `tf_debug.DebugDumpDir.dumped_tensor_data` {#DebugDumpDir.dumped_tensor_data}




- - -

#### `tf_debug.DebugDumpDir.find(predicate, first_n=0)` {#DebugDumpDir.find}

Find dumped tensor data by a certain predicate.

##### Args:


*  <b>`predicate`</b>: A callable that takes two input arguments:

    ```python
    def predicate(debug_tensor_datum, tensor):
      # returns a bool
    ```

    where `debug_tensor_datum` is an instance of `DebugTensorDatum`, which
    carries the metadata, such as the `Tensor`'s node name, output slot
    timestamp, debug op name, etc.; and `tensor` is the dumped tensor value
    as a `numpy.ndarray`.

*  <b>`first_n`</b>: (`int`) return only the first n `DebugTensotDatum` instances (in
    time order) for which the predicate returns True. To return all the
    `DebugTensotDatum` instances, let first_n be <= 0.

##### Returns:

  A list of all `DebugTensorDatum` objects in this `DebugDumpDir` object
   for which predicate returns True, sorted in ascending order of the
   timestamp.


- - -

#### `tf_debug.DebugDumpDir.get_dump_sizes_bytes(node_name, output_slot, debug_op)` {#DebugDumpDir.get_dump_sizes_bytes}

Get the sizes of the dump files for a debug-dumped tensor.

Unit of the file size: byte.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node that the tensor is produced by.
*  <b>`output_slot`</b>: (`int`) output slot index of tensor.
*  <b>`debug_op`</b>: (`str`) name of the debug op.

##### Returns:

  (`list` of `int`): list of dump file sizes in bytes.

##### Raises:


*  <b>`ValueError`</b>: If the tensor watch key does not exist in the debug dump data.


- - -

#### `tf_debug.DebugDumpDir.get_rel_timestamps(node_name, output_slot, debug_op)` {#DebugDumpDir.get_rel_timestamps}

Get the relative timestamp from for a debug-dumped tensor.

Relative timestamp means (absolute timestamp - `t0`), where `t0` is the
absolute timestamp of the first dumped tensor in the dump root. The tensor
may be dumped multiple times in the dump root directory, so a list of
relative timestamps (`numpy.ndarray`) is returned.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node that the tensor is produced by.
*  <b>`output_slot`</b>: (`int`) output slot index of tensor.
*  <b>`debug_op`</b>: (`str`) name of the debug op.

##### Returns:

  (`list` of `int`) list of relative timestamps.

##### Raises:


*  <b>`ValueError`</b>: If the tensor watch key does not exist in the debug dump data.


- - -

#### `tf_debug.DebugDumpDir.get_tensor_file_paths(node_name, output_slot, debug_op)` {#DebugDumpDir.get_tensor_file_paths}

Get the file paths from a debug-dumped tensor.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node that the tensor is produced by.
*  <b>`output_slot`</b>: (`int`) output slot index of tensor.
*  <b>`debug_op`</b>: (`str`) name of the debug op.

##### Returns:

  List of file path(s) loaded. This is a list because each debugged tensor
    may be dumped multiple times.

##### Raises:


*  <b>`ValueError`</b>: If the tensor does not exist in the debug-dump data.


- - -

#### `tf_debug.DebugDumpDir.get_tensors(node_name, output_slot, debug_op)` {#DebugDumpDir.get_tensors}

Get the tensor value from for a debug-dumped tensor.

The tensor may be dumped multiple times in the dump root directory, so a
list of tensors (`numpy.ndarray`) is returned.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node that the tensor is produced by.
*  <b>`output_slot`</b>: (`int`) output slot index of tensor.
*  <b>`debug_op`</b>: (`str`) name of the debug op.

##### Returns:

  List of tensors (`numpy.ndarray`) loaded from the debug-dump file(s).

##### Raises:


*  <b>`ValueError`</b>: If the tensor does not exist in the debug-dump data.


- - -

#### `tf_debug.DebugDumpDir.loaded_partition_graphs()` {#DebugDumpDir.loaded_partition_graphs}

Test whether partition graphs have been loaded.


- - -

#### `tf_debug.DebugDumpDir.node_attributes(node_name)` {#DebugDumpDir.node_attributes}

Get the attributes of a node.

##### Args:


*  <b>`node_name`</b>: Name of the node in question.

##### Returns:

  Attributes of the node.

##### Raises:


*  <b>`LookupError`</b>: If no partition graphs have been loaded.
*  <b>`ValueError`</b>: If no node named node_name exists.


- - -

#### `tf_debug.DebugDumpDir.node_device(node_name)` {#DebugDumpDir.node_device}

Get the device of a node.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node.

##### Returns:

  (`str`) name of the device on which the node is placed.

##### Raises:


*  <b>`LookupError`</b>: If node inputs and control inputs have not been loaded
     from partition graphs yet.
*  <b>`ValueError`</b>: If the node does not exist in partition graphs.


- - -

#### `tf_debug.DebugDumpDir.node_exists(node_name)` {#DebugDumpDir.node_exists}

Test if a node exists in the partition graphs.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node to be checked.

##### Returns:

  A boolean indicating whether the node exists.

##### Raises:


*  <b>`LookupError`</b>: If no partition graphs have been loaded yet.


- - -

#### `tf_debug.DebugDumpDir.node_inputs(node_name, is_control=False)` {#DebugDumpDir.node_inputs}

Get the inputs of given node according to partition graphs.

##### Args:


*  <b>`node_name`</b>: Name of the node.
*  <b>`is_control`</b>: (`bool`) Whether control inputs, rather than non-control
    inputs, are to be returned.

##### Returns:

  (`list` of `str`) inputs to the node, as a list of node names.

##### Raises:


*  <b>`LookupError`</b>: If node inputs and control inputs have not been loaded
     from partition graphs yet.
*  <b>`ValueError`</b>: If the node does not exist in partition graphs.


- - -

#### `tf_debug.DebugDumpDir.node_op_type(node_name)` {#DebugDumpDir.node_op_type}

Get the op type of given node.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node.

##### Returns:

  (`str`) op type of the node.

##### Raises:


*  <b>`LookupError`</b>: If node op types have not been loaded
     from partition graphs yet.
*  <b>`ValueError`</b>: If the node does not exist in partition graphs.


- - -

#### `tf_debug.DebugDumpDir.node_recipients(node_name, is_control=False)` {#DebugDumpDir.node_recipients}

Get recipient of the given node's output according to partition graphs.

##### Args:


*  <b>`node_name`</b>: (`str`) name of the node.
*  <b>`is_control`</b>: (`bool`) whether control outputs, rather than non-control
    outputs, are to be returned.

##### Returns:

  (`list` of `str`) all inputs to the node, as a list of node names.

##### Raises:


*  <b>`LookupError`</b>: If node inputs and control inputs have not been loaded
     from partition graphs yet.
*  <b>`ValueError`</b>: If the node does not exist in partition graphs.


- - -

#### `tf_debug.DebugDumpDir.node_traceback(element_name)` {#DebugDumpDir.node_traceback}

Try to retrieve the Python traceback of node's construction.

##### Args:


*  <b>`element_name`</b>: (`str`) Name of a graph element (node or tensor).

##### Returns:

  (list) The traceback list object as returned by the `extract_trace`
    method of Python's traceback module.

##### Raises:


*  <b>`LookupError`</b>: If Python graph is not available for traceback lookup.
*  <b>`KeyError`</b>: If the node cannot be found in the Python graph loaded.


- - -

#### `tf_debug.DebugDumpDir.nodes()` {#DebugDumpDir.nodes}

Get a list of all nodes from the partition graphs.

##### Returns:

  All nodes' names, as a list of str.

##### Raises:


*  <b>`LookupError`</b>: If no partition graphs have been loaded.


- - -

#### `tf_debug.DebugDumpDir.partition_graphs()` {#DebugDumpDir.partition_graphs}

Get the partition graphs.

##### Returns:

  Partition graphs as repeated fields of GraphDef.

##### Raises:


*  <b>`LookupError`</b>: If no partition graphs have been loaded.


- - -

#### `tf_debug.DebugDumpDir.set_python_graph(python_graph)` {#DebugDumpDir.set_python_graph}

Provide Python `Graph` object to the wrapper.

Unlike the partition graphs, which are protobuf `GraphDef` objects, `Graph`
is a Python object and carries additional information such as the traceback
of the construction of the nodes in the graph.

##### Args:


*  <b>`python_graph`</b>: (ops.Graph) The Python Graph object.


- - -

#### `tf_debug.DebugDumpDir.size` {#DebugDumpDir.size}

Total number of dumped tensors in the dump root directory.

##### Returns:

  (`int`) total number of dumped tensors in the dump root directory.


- - -

#### `tf_debug.DebugDumpDir.t0` {#DebugDumpDir.t0}

Absolute timestamp of the first dumped tensor.

##### Returns:

  (`int`) absolute timestamp of the first dumped tensor, in microseconds.


- - -

#### `tf_debug.DebugDumpDir.transitive_inputs(node_name, include_control=True)` {#DebugDumpDir.transitive_inputs}

Get the transitive inputs of given node according to partition graphs.

##### Args:


*  <b>`node_name`</b>: Name of the node
*  <b>`include_control`</b>: Include control inputs (True by default).

##### Returns:

  (`list` of `str`) all transitive inputs to the node, as a list of node
    names.

##### Raises:


*  <b>`LookupError`</b>: If node inputs and control inputs have not been loaded
     from partition graphs yet.
*  <b>`ValueError`</b>: If the node does not exist in partition graphs.


- - -

#### `tf_debug.DebugDumpDir.watch_key_to_data(debug_watch_key)` {#DebugDumpDir.watch_key_to_data}

Get all `DebugTensorDatum` instances corresponding to a debug watch key.

##### Args:


*  <b>`debug_watch_key`</b>: (`str`) debug watch key.

##### Returns:

  A list of `DebugTensorDatum` instances that correspond to the debug watch
  key. If the watch key does not exist, returns an empty list.

##### Raises:


*  <b>`ValueError`</b>: If the debug watch key does not exist.





## Functions for loading debug-dump data

- - -

### `tf_debug.load_tensor_from_event_file(event_file_path)` {#load_tensor_from_event_file}

Load a tensor from an event file.

Assumes that the event file contains a `Event` protobuf and the `Event`
protobuf contains a `Tensor` value.

##### Args:


*  <b>`event_file_path`</b>: (`str`) path to the event file.

##### Returns:

  The tensor value loaded from the event file, as a `numpy.ndarray`. For
  uninitialized tensors, returns None.




## Tensor-value predicates

Built-in tensor-filter predicates to support conditional breakpoint between
runs. See `DebugDumpDir.find()` for more details.

- - -

### `tf_debug.has_inf_or_nan(datum, tensor)` {#has_inf_or_nan}

A predicate for whether a tensor consists of any bad numerical values.

This predicate is common enough to merit definition in this module.
Bad numerical values include `nan`s and `inf`s.
The signature of this function follows the requirement of the method
`DebugDumpDir.find()`.

##### Args:


*  <b>`datum`</b>: (`DebugTensorDatum`) Datum metadata.
*  <b>`tensor`</b>: (`numpy.ndarray` or None) Value of the tensor. None represents
    an uninitialized tensor.

##### Returns:

  (`bool`) True if and only if tensor consists of any nan or inf values.




## Session wrapper class and `SessionRunHook` implementations

These classes allow you to

* wrap aroundTensorFlow `Session` objects to debug  plain TensorFlow models
  (see `LocalCLIDebugWrapperSession`), or
* generate `SessionRunHook` objects to debug `tf.contrib.learn` models (see
  `LocalCLIDebugHook`).

- - -

### `class tf_debug.LocalCLIDebugHook` {#LocalCLIDebugHook}

Command-line-interface debugger hook.

Can be used as a monitor/hook for tf.train.MonitoredSession.
- - -

#### `tf_debug.LocalCLIDebugHook.__enter__()` {#LocalCLIDebugHook.__enter__}




- - -

#### `tf_debug.LocalCLIDebugHook.__exit__(exec_type, exec_value, exec_tb)` {#LocalCLIDebugHook.__exit__}




- - -

#### `tf_debug.LocalCLIDebugHook.__init__()` {#LocalCLIDebugHook.__init__}

Create a local debugger command-line interface (CLI) hook.


- - -

#### `tf_debug.LocalCLIDebugHook.add_tensor_filter(filter_name, tensor_filter)` {#LocalCLIDebugHook.add_tensor_filter}

Add a tensor filter.

##### Args:


*  <b>`filter_name`</b>: (`str`) name of the filter.
*  <b>`tensor_filter`</b>: (`callable`) the filter callable. See the doc string of
    `DebugDumpDir.find()` for more details about its signature.


- - -

#### `tf_debug.LocalCLIDebugHook.after_create_session(session)` {#LocalCLIDebugHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf_debug.LocalCLIDebugHook.after_run(run_context, run_values)` {#LocalCLIDebugHook.after_run}




- - -

#### `tf_debug.LocalCLIDebugHook.before_run(run_context)` {#LocalCLIDebugHook.before_run}




- - -

#### `tf_debug.LocalCLIDebugHook.begin()` {#LocalCLIDebugHook.begin}




- - -

#### `tf_debug.LocalCLIDebugHook.close()` {#LocalCLIDebugHook.close}




- - -

#### `tf_debug.LocalCLIDebugHook.end(session)` {#LocalCLIDebugHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


- - -

#### `tf_debug.LocalCLIDebugHook.graph` {#LocalCLIDebugHook.graph}




- - -

#### `tf_debug.LocalCLIDebugHook.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#LocalCLIDebugHook.invoke_node_stepper}

Overrides method in base class to implement interactive node stepper.

##### Args:


*  <b>`node_stepper`</b>: (`stepper.NodeStepper`) The underlying NodeStepper API
    object.
*  <b>`restore_variable_values_on_exit`</b>: (`bool`) Whether any variables whose
    values have been altered during this node-stepper invocation should be
    restored to their old values when this invocation ends.

##### Returns:

  The same return values as the `Session.run()` call on the same fetches as
    the NodeStepper.


- - -

#### `tf_debug.LocalCLIDebugHook.on_run_end(request)` {#LocalCLIDebugHook.on_run_end}

Overrides on-run-end callback.

##### Actions taken:

  1) Load the debug dump.
  2) Bring up the Analyzer CLI.

##### Args:


*  <b>`request`</b>: An instance of OnSessionInitRequest.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugHook.on_run_start(request)` {#LocalCLIDebugHook.on_run_start}

Overrides on-run-start callback.

##### Invoke the CLI to let user choose what action to take:

  `run` / `invoke_stepper`.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of `OnSessionInitResponse`.

##### Raises:


*  <b>`RuntimeError`</b>: If user chooses to prematurely exit the debugger.


- - -

#### `tf_debug.LocalCLIDebugHook.on_session_init(request)` {#LocalCLIDebugHook.on_session_init}

Overrides on-session-init callback.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugHook.partial_run(handle, fetches, feed_dict=None)` {#LocalCLIDebugHook.partial_run}




- - -

#### `tf_debug.LocalCLIDebugHook.partial_run_setup(fetches, feeds=None)` {#LocalCLIDebugHook.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.LocalCLIDebugHook.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#LocalCLIDebugHook.run}

Wrapper around Session.run() that inserts tensor watch options.

##### Args:


*  <b>`fetches`</b>: Same as the `fetches` arg to regular `Session.run()`.
*  <b>`feed_dict`</b>: Same as the `feed_dict` arg to regular `Session.run()`.
*  <b>`options`</b>: Same as the `options` arg to regular `Session.run()`.
*  <b>`run_metadata`</b>: Same as the `run_metadata` arg to regular `Session.run()`.

##### Returns:

  Simply forwards the output of the wrapped `Session.run()` call.

##### Raises:


*  <b>`ValueError`</b>: On invalid `OnRunStartAction` value.


- - -

#### `tf_debug.LocalCLIDebugHook.sess_str` {#LocalCLIDebugHook.sess_str}




- - -

#### `tf_debug.LocalCLIDebugHook.session` {#LocalCLIDebugHook.session}





- - -

### `class tf_debug.LocalCLIDebugWrapperSession` {#LocalCLIDebugWrapperSession}

Concrete subclass of BaseDebugWrapperSession implementing a local CLI.

This class has all the methods that a `session.Session` object has, in order
to support debugging with minimal code changes. Invoking its `run()` method
will launch the command-line interface (CLI) of tfdbg.
- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__enter__()` {#LocalCLIDebugWrapperSession.__enter__}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__exit__(exec_type, exec_value, exec_tb)` {#LocalCLIDebugWrapperSession.__exit__}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__init__(sess, dump_root=None, log_usage=True)` {#LocalCLIDebugWrapperSession.__init__}

Constructor of LocalCLIDebugWrapperSession.

##### Args:


*  <b>`sess`</b>: The TensorFlow `Session` object being wrapped.
*  <b>`dump_root`</b>: (`str`) optional path to the dump root directory. Must be a
    directory that does not exist or an empty directory. If the directory
    does not exist, it will be created by the debugger core during debug
    `run()` calls and removed afterwards.
*  <b>`log_usage`</b>: (`bool`) whether the usage of this class is to be logged.

##### Raises:


*  <b>`ValueError`</b>: If dump_root is an existing and non-empty directory or if
    dump_root is a file.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.add_tensor_filter(filter_name, tensor_filter)` {#LocalCLIDebugWrapperSession.add_tensor_filter}

Add a tensor filter.

##### Args:


*  <b>`filter_name`</b>: (`str`) name of the filter.
*  <b>`tensor_filter`</b>: (`callable`) the filter callable. See the doc string of
    `DebugDumpDir.find()` for more details about its signature.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.close()` {#LocalCLIDebugWrapperSession.close}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.graph` {#LocalCLIDebugWrapperSession.graph}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#LocalCLIDebugWrapperSession.invoke_node_stepper}

Overrides method in base class to implement interactive node stepper.

##### Args:


*  <b>`node_stepper`</b>: (`stepper.NodeStepper`) The underlying NodeStepper API
    object.
*  <b>`restore_variable_values_on_exit`</b>: (`bool`) Whether any variables whose
    values have been altered during this node-stepper invocation should be
    restored to their old values when this invocation ends.

##### Returns:

  The same return values as the `Session.run()` call on the same fetches as
    the NodeStepper.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.on_run_end(request)` {#LocalCLIDebugWrapperSession.on_run_end}

Overrides on-run-end callback.

##### Actions taken:

  1) Load the debug dump.
  2) Bring up the Analyzer CLI.

##### Args:


*  <b>`request`</b>: An instance of OnSessionInitRequest.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.on_run_start(request)` {#LocalCLIDebugWrapperSession.on_run_start}

Overrides on-run-start callback.

##### Invoke the CLI to let user choose what action to take:

  `run` / `invoke_stepper`.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of `OnSessionInitResponse`.

##### Raises:


*  <b>`RuntimeError`</b>: If user chooses to prematurely exit the debugger.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.on_session_init(request)` {#LocalCLIDebugWrapperSession.on_session_init}

Overrides on-session-init callback.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.partial_run(handle, fetches, feed_dict=None)` {#LocalCLIDebugWrapperSession.partial_run}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.partial_run_setup(fetches, feeds=None)` {#LocalCLIDebugWrapperSession.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#LocalCLIDebugWrapperSession.run}

Wrapper around Session.run() that inserts tensor watch options.

##### Args:


*  <b>`fetches`</b>: Same as the `fetches` arg to regular `Session.run()`.
*  <b>`feed_dict`</b>: Same as the `feed_dict` arg to regular `Session.run()`.
*  <b>`options`</b>: Same as the `options` arg to regular `Session.run()`.
*  <b>`run_metadata`</b>: Same as the `run_metadata` arg to regular `Session.run()`.

##### Returns:

  Simply forwards the output of the wrapped `Session.run()` call.

##### Raises:


*  <b>`ValueError`</b>: On invalid `OnRunStartAction` value.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.sess_str` {#LocalCLIDebugWrapperSession.sess_str}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.session` {#LocalCLIDebugWrapperSession.session}





