import logging
import os
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.core.framework import types_pb2 as _types_pb2
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import tensor_shape
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.framework import op_def_registry as _registry

# Load the native method
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_name = os.path.join(this_dir, '_avro_record_dataset.so')
reader_module = tf.load_op_library(lib_name)

_DEFAULT_READER_BUFFER_SIZE_BYTES = 256 * 1024  # 256 KB


def _create_avro_dataset_op_proto():
    """
    Creates the proto definition for the Avro dataset operator.

    :return: proto definition for the Avro dataset operator.
    """

    # Create a proto representation for the new op
    # op {
    #   name: "AvroRecordDataset"
    #   input_arg {
    #     name: "filenames"
    #     type: DT_STRING
    #   }
    #   input_arg {
    #     name: "schema"
    #     type: DT_STRING
    #   }
    #   input_arg {
    #     name: "buffer_size"
    #     type: DT_INT64
    #   }
    #   output_arg {
    #     name: "handle"
    #     type: DT_VARIANT
    #   }
    #   is_stateful: true
    # }

    # Create the operator definition
    avro_dataset_op = _op_def_pb2.OpDef()
    avro_dataset_op.name = "AvroRecordDataset"

    # Create and add the input arguments
    input_filenames = _op_def_pb2.OpDef.ArgDef()
    input_filenames.name = "filenames"
    input_filenames.type = _types_pb2.DT_STRING
    input_schema = _op_def_pb2.OpDef.ArgDef()
    input_schema.name = "schema"
    input_schema.type = _types_pb2.DT_STRING
    input_buffer_size = _op_def_pb2.OpDef.ArgDef()
    input_buffer_size.name = "buffer_size"
    input_buffer_size.type = _types_pb2.DT_INT64
    avro_dataset_op.input_arg.extend([input_filenames, input_schema, input_buffer_size])

    # Create and add the output argument
    output_handle = _op_def_pb2.OpDef.ArgDef()
    output_handle.name = "handle"
    output_handle.type = _types_pb2.DT_VARIANT
    avro_dataset_op.output_arg.extend([output_handle])

    # This method is stateful because it keeps a counter into rows/records
    avro_dataset_op.is_stateful = True

    return avro_dataset_op


def _create_op_def_library(op_proto):
    """
    Creates a customized operator definition library for the given op_proto.

    Notice that this method will check that a CC implementation for this method with the SAME signature has been
    registered with TensorFlow.  Make sure to load the native code before calling this method.

    :return: A operation definition library.
    """
    # Log the proto representation of the operator
    logging.info("Adding operator {} to the library.".format(op_proto))

    # Register the new op with TF and create an op library for it
    # Note: The native cc code will register the operator in _registered_ops, This is for checking
    registered_ops = _registry.get_registered_ops()
    if op_proto.name not in registered_ops:
        raise RuntimeError("Could not find native implementation of op '{}'.".format(op_proto.name))

    op_def_lib = _op_def_library.OpDefLibrary()
    ops_proto = _op_def_pb2.OpList()
    ops_proto.op.extend([op_proto])

    # Make sure the registered op matches this one by re-registering it
    _registry.register_op_list(ops_proto)  # Will fail if the natively registered op differs

    # Add the op to the customized operator library
    op_def_lib.add_op_list(ops_proto)

    return op_def_lib


# Load the customized operator library which contains only our avro dataset operator -- but that is enough here
_op_def_lib = _create_op_def_library(_create_avro_dataset_op_proto())


def _convert_optional_param_to_tensor(argument_name,
                                      argument_value,
                                      argument_default,
                                      argument_dtype=dtypes.int64):
    """
    Convert optional parameter to tensor.  This method has been copied from the original TensorFlow code:
    https://github.com/tensorflow/tensorflow/blob/v1.4.0/tensorflow/python/data/ops/readers.py

    :param argument_name: The argument name.
    :param argument_value: The value.
    :param argument_default: The default argument.
    :param argument_dtype: The type.
    :return:
    """
    if argument_value is not None:
        return ops.convert_to_tensor(
            argument_value, dtype=argument_dtype, name=argument_name)
    else:
        return constant_op.constant(
            argument_default, dtype=argument_dtype, name=argument_name)


def avro_record_dataset(filenames, schema, buffer_size, name=None):
    """Creates a dataset that emits the records from one or more AvroRecord files.

    Adopted from the generated file: gen_dataset_ops.py and therein the function tf_record_dataset in TensorFlow 1.4.

    Args:
      filenames: A `Tensor` of type `string`.
        A scalar or vector containing the name(s) of the file(s) to be
        read.
      schema: A `Tensor` of type `string` representing the schema used for schema resolution. If none is supplied
        defaults to the source file's schema (optional).
      buffer_size: A `Tensor` of type `int64`.
        A scalar representing the number of bytes to buffer.
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `variant`.
    """
    _ctx = _context.context()
    if _ctx.in_graph_mode():
        # Must call _apply_op_helper here and not apply_op because apply_op will add a non-compatible conversion
        _, _, _op = _op_def_lib._apply_op_helper(
            "AvroRecordDataset", filenames=filenames,
            schema=schema, buffer_size=buffer_size, name=name)
        _result = _op.outputs[:]
        _inputs_flat = _op.inputs
        _attrs = None
    else:
        filenames = ops.convert_to_tensor(filenames, dtypes.string)
        schema = ops.convert_to_tensor(schema, dtypes.string)
        buffer_size = ops.convert_to_tensor(buffer_size, dtypes.int64)
        _inputs_flat = [filenames, schema, buffer_size]
        _attrs = None
        _result = _execute.execute(b"AvroRecordDataset", 1, inputs=_inputs_flat,
                                   attrs=_attrs, ctx=_ctx, name=name)
    _execute.record_gradient(
        "AvroRecordDataset", _inputs_flat, _attrs, _result, name)
    _result, = _result
    return _result


class AvroRecordDataset(Dataset):
    """A `Dataset` comprising records from one or more Avro files."""

    def __init__(self, filenames, schema=None, buffer_size=None):
        """Creates a `AvroRecordDataset`.
        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          schema: (Optional.) A `tf.string` scalar for schema resolution.
          buffer_size: (Optional.) A `tf.int64` scalar representing the number of
            bytes in the read buffer. Must be larger >= 256.
        """
        super(AvroRecordDataset, self).__init__()

        # Force the type to string even if filenames is an empty list.
        self._filenames = ops.convert_to_tensor(filenames, dtypes.string, name="filenames")
        self._schema = _convert_optional_param_to_tensor(
            "schema",
            schema,
            argument_default="",
            argument_dtype=dtypes.string)
        self._buffer_size = _convert_optional_param_to_tensor(
            "buffer_size",
            buffer_size,
            argument_default=_DEFAULT_READER_BUFFER_SIZE_BYTES)

    def _as_variant_tensor(self):
        return avro_record_dataset(self._filenames, self._schema, self._buffer_size)

    @property
    def output_classes(self):
        return ops.Tensor

    @property
    def output_shapes(self):
        return tensor_shape.TensorShape([])

    @property
    def output_types(self):
        return dtypes.string
