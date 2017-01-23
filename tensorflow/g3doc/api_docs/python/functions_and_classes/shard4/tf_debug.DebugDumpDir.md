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

#### `tf_debug.DebugDumpDir.run_feed_keys_info` {#DebugDumpDir.run_feed_keys_info}

Get a str representation of the feed_dict used in the Session.run() call.

##### Returns:

  If the information is available, a `str` obtained from `repr(feed_dict)`.
  If the information is not available, `None`.


- - -

#### `tf_debug.DebugDumpDir.run_fetches_info` {#DebugDumpDir.run_fetches_info}

Get a str representation of the fetches used in the Session.run() call.

##### Returns:

  If the information is available, a `str` obtained from `repr(fetches)`.
  If the information is not available, `None`.


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


