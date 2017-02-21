### `tf_debug.load_tensor_from_event_file(event_file_path)` {#load_tensor_from_event_file}

Load a tensor from an event file.

Assumes that the event file contains a `Event` protobuf and the `Event`
protobuf contains a `Tensor` value.

##### Args:


*  <b>`event_file_path`</b>: (`str`) path to the event file.

##### Returns:

  The tensor value loaded from the event file, as a `numpy.ndarray`. For
  uninitialized Tensors, returns `None`. For Tensors of data types that
  cannot be converted to `numpy.ndarray` (e.g., `tf.resource`), return
  `None`.

