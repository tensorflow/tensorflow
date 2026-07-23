
  def testParseTensorMalformedVariantTensorListDoesNotCrash(self):
    """Regression test: malformed TensorList Variant must raise clean error.

    A TensorProto with dtype=DT_VARIANT wrapping a tensorflow::TensorList
    with empty metadata previously caused a SIGSEGV in PyErr_Occurred() via
    pybind11 exception translation instead of raising InvalidArgumentError.
    Confirmed in TF 2.21.0 in fresh isolated containers.

    Root cause: FromProtoField<Variant> in tensor.cc called buf->Unref()
    with data[0..i] holding partially-decoded Variant objects after
    DecodeUnaryVariant returned false, corrupting Python thread state.
    """
    from tensorflow.core.framework import tensor_pb2, types_pb2
    outer = tensor_pb2.TensorProto()
    outer.dtype = types_pb2.DT_VARIANT
    outer.tensor_shape.dim.add().size = 1
    v = outer.variant_val.add()
    v.type_name = "tensorflow::TensorList"
    leaf = v.tensors.add()
    leaf.dtype = types_pb2.DT_FLOAT
    leaf.tensor_shape.dim.add().size = 1
    serialized = outer.SerializeToString()
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(tf.io.parse_tensor(serialized, out_type=tf.variant))

