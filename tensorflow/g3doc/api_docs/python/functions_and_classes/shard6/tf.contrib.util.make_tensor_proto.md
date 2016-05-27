### `tf.contrib.util.make_tensor_proto(values, dtype=None, shape=None)` {#make_tensor_proto}

Create a TensorProto.

##### Args:


*  <b>`values`</b>: Values to put in the TensorProto.
*  <b>`dtype`</b>: Optional tensor_pb2 DataType value.
*  <b>`shape`</b>: List of integers representing the dimensions of tensor.

##### Returns:

  A TensorProto. Depending on the type, it may contain data in the
  "tensor_content" attribute, which is not directly useful to Python programs.
  To access the values you should convert the proto back to a numpy ndarray
  with tensor_util.MakeNdarray(proto).

##### Raises:


*  <b>`TypeError`</b>: if unsupported types are provided.
*  <b>`ValueError`</b>: if arguments have inappropriate values.

make_tensor_proto accepts "values" of a python scalar, a python list, a
numpy ndarray, or a numpy scalar.

If "values" is a python scalar or a python list, make_tensor_proto
first convert it to numpy ndarray. If dtype is None, the
conversion tries its best to infer the right numpy data
type. Otherwise, the resulting numpy array has a compatible data
type with the given dtype.

In either case above, the numpy ndarray (either the caller provided
or the auto converted) must have the compatible type with dtype.

make_tensor_proto then converts the numpy array to a tensor proto.

If "shape" is None, the resulting tensor proto represents the numpy
array precisely.

Otherwise, "shape" specifies the tensor's shape and the numpy array
can not have more elements than what "shape" specifies.

