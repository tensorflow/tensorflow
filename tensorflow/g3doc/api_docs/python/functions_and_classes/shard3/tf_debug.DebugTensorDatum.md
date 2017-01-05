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


