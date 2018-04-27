import collections
import os
import re
import tensorflow as tf

from tensorflow.python.eager import context as _context
from tensorflow.python.eager import execute as _execute

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape

from tensorflow.core.framework import attr_value_pb2 as _attr_value_pb2
from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
from tensorflow.core.framework import types_pb2 as _types_pb2

from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parsing_ops import _features_to_raw_params, _prepend_none_dimension, VarLenFeature, \
    SparseFeature, FixedLenFeature, FixedLenSequenceFeature, _construct_sparse_tensors_for_sparse_features

# HACK: Change to tensorflow.contrib.avro when tensorflow/contrib/__init__.py
from .avro_record_dataset import _create_op_def_library

# Load the shared library
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_name = os.path.join(this_dir, '_parse_avro_record.so')  # Display the operators with lib_name.OP_LIST
parse_module = tf.load_op_library(lib_name)  # Load the library with dependent so's in '.'


def parse_avro_record(serialized, schema, features):
    """
    Parses serialized avro records into TensorFlow tensors. This method also handles a batch of serialized records and
    returns tensors with the batch added as first dimension.

    :param serialized: The serialized avro record(s) as TensorFlow string(s).

    :param schema: This is the schema that these records where serialized with. Schema resolution is not supported here
                   but in the AvroRecordReader.

    :param features: Is a map of keys that describe a single entry or sparse vector in the avro record and map that
                     entry to a tensor. The syntax is as follows:

                     features = {'my_meta_data/size': tf.FixedLenFeature([], tf.int64)}

                        Select the 'size' field from a record metadata that is in the field 'my_meta_data'. In this
                        example we assume that the size is encoded as a long in the Avro record for the metadata.


                     features = {'my_map_data/['source']/ip_addresses': tf.VarLenFeature([], tf.string)}

                        Select the 'ip_addresses' for the 'source' key in the map 'my_map_data'. Notice we assume that
                        IP addresses are encoded as strings.


                     features = {'my_friends/[1]/first_name': tf.FixedLenFeature([], tf.string)}

                        Select the 'first_name' for the second friend with index '1'. This assumes that all of your data
                        has a second friend. In addition, we assume that all friends have only one first name. For this
                        reason we chose a 'FixedLenFeature'.


                     features = {'my_friends/[*]/first_name': tf.VarLenFeature([], tf.string)}

                        Select all first_names in each row. For this example we use the wildcard '*' to indicate that
                        we want to select all 'first_name' entries from the array.


                     features = {'sparse_features': tf.SparseFeature(index_key='index', value_key='value',
                                dtype=tf.float32, size=10)}

                        We assume that sparse features contains an array with records that contain an 'index' field
                        that MUST BE LONG and an 'value' field with floats (single precision).

    :return: A map of with the same key as in features and that has the corresponding tensors as values.
    """

    # Code from https://github.com/tensorflow/tensorflow/blob/v1.4.1/tensorflow/python/ops/parsing_ops.py
    # For now I copied from these two methods 'parse_example' and '_parse_example_raw'
    # The TensorFlow source code could be refactored to fully integrate the avro parser and avoid copying code!

    if not features:
        raise ValueError("Missing: features was '{}'.".format(features))
    if not schema:
        raise ValueError("Missing: schema was '{}'".format(schema))

    features = _prepend_none_dimension(features)
    # ******************** START difference: This part is different from the originally copied code ********************
    features = _build_keys_for_sparse_features(features)
    # ******************** END difference: This part is different from the originally copied code **********************
    (sparse_keys, sparse_types, dense_keys, dense_types, dense_defaults, dense_shapes) = _features_to_raw_params(
        features, [VarLenFeature, SparseFeature, FixedLenFeature, FixedLenSequenceFeature])

    dense_defaults = collections.OrderedDict() if dense_defaults is None else dense_defaults
    sparse_keys = [] if sparse_keys is None else sparse_keys
    sparse_types = [] if sparse_types is None else sparse_types
    dense_keys = [] if dense_keys is None else dense_keys
    dense_types = [] if dense_types is None else dense_types
    dense_shapes = ([[]] * len(dense_keys) if dense_shapes is None else dense_shapes)

    num_dense = len(dense_keys)
    num_sparse = len(sparse_keys)

    if len(dense_shapes) != num_dense:
        raise ValueError("len(dense_shapes) != len(dense_keys): %d vs. %d"
                         % (len(dense_shapes), num_dense))
    if len(dense_types) != num_dense:
        raise ValueError("len(dense_types) != len(num_dense): %d vs. %d"
                         % (len(dense_types), num_dense))
    if len(sparse_types) != num_sparse:
        raise ValueError("len(sparse_types) != len(sparse_keys): %d vs. %d"
                         % (len(sparse_types), num_sparse))
    if num_dense + num_sparse == 0:
        raise ValueError("Must provide at least one sparse key or dense key")
    if not set(dense_keys).isdisjoint(set(sparse_keys)):
        raise ValueError(
            "Dense and sparse keys must not intersect; intersection: %s" %
            set(dense_keys).intersection(set(sparse_keys)))

    # Convert dense_shapes to TensorShape object.
    dense_shapes = [tensor_shape.as_shape(shape) for shape in dense_shapes]

    dense_defaults_vec = []
    for i, key in enumerate(dense_keys):
        default_value = dense_defaults.get(key)
        dense_shape = dense_shapes[i]
        # This part is used by the FixedLenSequenceFeature
        if dense_shape.ndims is not None and dense_shape.ndims > 0 and dense_shape[0].value is None:
            # Variable stride dense shape, the default value should be a scalar padding value
            if default_value is None:
                # ************* START difference: This part is different from the originally copied code ***************
                # Support default for other types
                if dense_types[i] == dtypes.string:
                    default_value = ""
                elif dense_types[i] == dtypes.bool:
                    default_value = False
                else:  # Should be numeric type
                    default_value = 0
                default_value = ops.convert_to_tensor(default_value, dtype=dense_types[i])
                # ************* END difference: This part is different from the originally copied code *****************
            else:
                # Reshape to a scalar to ensure user gets an error if they
                # provide a tensor that's not intended to be a padding value
                # (0 or 2+ elements).
                key_name = "padding_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
                default_value = ops.convert_to_tensor(
                    default_value, dtype=dense_types[i], name=key_name)
                default_value = array_ops.reshape(default_value, [])
        else:
            # This part is used by the FixedLenFeature
            if default_value is None:
                default_value = constant_op.constant([], dtype=dense_types[i])
            elif not isinstance(default_value, ops.Tensor):
                key_name = "key_" + re.sub("[^A-Za-z0-9_.\\-/]", "_", key)
                default_value = ops.convert_to_tensor(
                    default_value, dtype=dense_types[i], name=key_name)
                default_value = array_ops.reshape(default_value, dense_shape)

        dense_defaults_vec.append(default_value)

    # Finally, convert dense_shapes to TensorShapeProto
    dense_shapes = [shape.as_proto() for shape in dense_shapes]

    # ******************** START difference: This part is different from the originally copied code ********************
    outputs = _parse_avro_record(serialized=serialized,
                                 sparse_keys=sparse_keys,
                                 sparse_types=sparse_types,
                                 dense_defaults=dense_defaults_vec,
                                 dense_keys=dense_keys,
                                 dense_shapes=dense_shapes,
                                 schema=schema)
    # ********************** END difference: This part is different from the originally copied code ********************

    (sparse_indices, sparse_values, sparse_shapes, dense_values) = outputs

    sparse_tensors = [
        sparse_tensor.SparseTensor(ix, val, shape) for (ix, val, shape)
        in zip(sparse_indices, sparse_values, sparse_shapes)]

    return _construct_sparse_tensors_for_sparse_features(
        features, dict(zip(sparse_keys + dense_keys, sparse_tensors + dense_values)))


def _create_parse_avro_record_op_proto():
    """
    Creates the operator definition for the parse avro record function.

    Side note: Once we integrate this into the TensorFlow source code this would be automatically generated.

    :return: proto object.
    """
    # op {
    #   name: "ParseExample"
    #   input_arg {
    #     name: "serialized"
    #     type: DT_STRING
    #   }
    #   input_arg {
    #     name: "sparse_keys"
    #     type: DT_STRING
    #     number_attr: "Nsparse"
    #   }
    #   input_arg {
    #     name: "dense_keys"
    #     type: DT_STRING
    #     number_attr: "Ndense"
    #   }
    #   input_arg {
    #     name: "dense_defaults"
    #     type_list_attr: "Tdense"
    #   }
    #   output_arg {
    #     name: "sparse_indices"
    #     type: DT_INT64
    #     number_attr: "Nsparse"
    #   }
    #   output_arg {
    #     name: "sparse_values"
    #     type_list_attr: "sparse_types"
    #   }
    #   output_arg {
    #     name: "sparse_shapes"
    #     type: DT_INT64
    #     number_attr: "Nsparse"
    #   }
    #   output_arg {
    #     name: "dense_values"
    #     type_list_attr: "Tdense"
    #   }
    #   attr {
    #     name: "Nsparse"
    #     type: "int"
    #     has_minimum: true
    #   }
    #   attr {
    #     name: "Ndense"
    #     type: "int"
    #     has_minimum: true
    #   }
    #   attr {
    #     name: "sparse_types"
    #     type: "list(type)"
    #     has_minimum: true
    #     allowed_values {
    #       list {
    #         type: DT_DOUBLE
    #         type: DT_FLOAT
    #         type: DT_INT64
    #         type: DT_INT32
    #         type: DT_STRING
    #         type: DT_BOOL
    #       }
    #     }
    #   }
    #   attr {
    #     name: "Tdense"
    #     type: "list(type)"
    #     has_minimum: true
    #     allowed_values {
    #       list {
    #         type: DT_DOUBLE
    #         type: DT_FLOAT
    #         type: DT_INT64
    #         type: DT_INT32
    #         type: DT_STRING
    #         type: DT_BOOL
    #       }
    #     }
    #   }
    #   attr {
    #     name: "dense_shapes"
    #     type: "list(shape)"
    #     has_minimum: true
    #   }
    #   attr {
    #     name: "schema"
    #     type: DT_STRING
    #   }
    # }

    # tensorflow/core/framework/op_def.proto  Helps to know the proto definitions for operators

    # Create the operator definition
    parse_avro_op = _op_def_pb2.OpDef(name="ParseAvroRecord")

    # Create input arguments
    inputs = [
        _op_def_pb2.OpDef.ArgDef(name="serialized", type=_types_pb2.DT_STRING),
        _op_def_pb2.OpDef.ArgDef(name="sparse_keys", type=_types_pb2.DT_STRING, number_attr="Nsparse"),
        _op_def_pb2.OpDef.ArgDef(name="dense_keys", type=_types_pb2.DT_STRING, number_attr="Ndense"),
        _op_def_pb2.OpDef.ArgDef(name="dense_defaults", type_list_attr="Tdense")
    ]

    # Create output arguments
    outputs = [
        _op_def_pb2.OpDef.ArgDef(name="sparse_indices", type=_types_pb2.DT_INT64, number_attr="Nsparse"),
        _op_def_pb2.OpDef.ArgDef(name="sparse_values", type_list_attr="sparse_types"),
        _op_def_pb2.OpDef.ArgDef(name="sparse_shapes", type=_types_pb2.DT_INT64, number_attr="Nsparse"),
        _op_def_pb2.OpDef.ArgDef(name="dense_values", type_list_attr="Tdense")
    ]

    # Create attributes
    values = _attr_value_pb2.AttrValue(list=_attr_value_pb2.AttrValue.ListValue(
        type=[_types_pb2.DT_FLOAT, _types_pb2.DT_DOUBLE, _types_pb2.DT_INT64, _types_pb2.DT_INT32, _types_pb2.DT_STRING,
              _types_pb2.DT_BOOL]))
    attributes = [
        _op_def_pb2.OpDef.AttrDef(name="Nsparse", type="int", has_minimum=True),
        _op_def_pb2.OpDef.AttrDef(name="Ndense", type="int", has_minimum=True),
        _op_def_pb2.OpDef.AttrDef(name="sparse_types", type="list(type)", has_minimum=True, allowed_values=values),
        _op_def_pb2.OpDef.AttrDef(name="Tdense", type="list(type)", has_minimum=True, allowed_values=values),
        _op_def_pb2.OpDef.AttrDef(name="dense_shapes", type="list(shape)", has_minimum=True),
        _op_def_pb2.OpDef.AttrDef(name="schema", type="string")
    ]

    # Add input, outputs, attributes
    parse_avro_op.input_arg.extend(inputs)
    parse_avro_op.output_arg.extend(outputs)
    parse_avro_op.attr.extend(attributes)

    return parse_avro_op


# Load the customized operator library which contains only our avro dataset operator -- but that is enough here
_op_def_lib = _create_op_def_library(_create_parse_avro_record_op_proto())

# Code from tensorflow/python/ops/gen_parsing_op.py and the the function '_parse_example'
__parse_avro_outputs = ["sparse_indices", "sparse_values", "sparse_shapes", "dense_values"]
_ParseAvroOutput = collections.namedtuple("ParseExample", __parse_avro_outputs)


def _parse_avro_record(serialized, sparse_keys, dense_keys, dense_defaults, sparse_types, dense_shapes, schema,
                       name=None):
    """
    Side note: Once we integrate this into TensorFlow this function will be automatically generated.

    ATTENTION: This is not part of the public interface!

    Args:
    serialized: A `Tensor` of type `string`.
      A vector containing a batch of binary serialized Example protos.
    names: A `Tensor` of type `string`.
      A vector containing the names of the serialized protos.
      May contain, for example, table key (descriptive) names for the
      corresponding serialized protos.  These are purely useful for debugging
      purposes, and the presence of values here has no effect on the output.
      May also be an empty vector if no names are available.
      If non-empty, this vector must be the same length as "serialized".
    sparse_keys: A list of `Tensor` objects with type `string`.
      A list of Nsparse string Tensors (scalars).
      The keys expected in the Examples' features associated with sparse values.
    dense_keys: A list of `Tensor` objects with type `string`.
      A list of Ndense string Tensors (scalars).
      The keys expected in the Examples' features associated with dense values.
    dense_defaults: A list of `Tensor` objects with types from: `float32`, `int64`, `string`.
      A list of Ndense Tensors (some may be empty).
      dense_defaults[j] provides default values
      when the example's feature_map lacks dense_key[j].  If an empty Tensor is
      provided for dense_defaults[j], then the Feature dense_keys[j] is required.
      The input type is inferred from dense_defaults[j], even when it's empty.
      If dense_defaults[j] is not empty, and dense_shapes[j] is fully defined,
      then the shape of dense_defaults[j] must match that of dense_shapes[j].
      If dense_shapes[j] has an undefined major dimension (variable strides dense
      feature), dense_defaults[j] must contain a single element:
      the padding element.
    sparse_types: A list of `tf.DTypes` from: `tf.float32, tf.int64, tf.string`.
      A list of Nsparse types; the data types of data in each Feature
      given in sparse_keys.
      Currently the ParseExample supports DT_FLOAT (FloatList),
      DT_INT64 (Int64List), and DT_STRING (BytesList).
    dense_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      A list of Ndense shapes; the shapes of data in each Feature
      given in dense_keys.
      The number of elements in the Feature corresponding to dense_key[j]
      must always equal dense_shapes[j].NumEntries().
      If dense_shapes[j] == (D0, D1, ..., DN) then the shape of output
      Tensor dense_values[j] will be (|serialized|, D0, D1, ..., DN):
      The dense outputs are just the inputs row-stacked by batch.
      This works for dense_shapes[j] = (-1, D1, ..., DN).  In this case
      the shape of the output Tensor dense_values[j] will be
      (|serialized|, M, D1, .., DN), where M is the maximum number of blocks
      of elements of length D1 * .... * DN, across all minibatch entries
      in the input.  Any minibatch entry with less than M blocks of elements of
      length D1 * ... * DN will be padded with the corresponding default_value
      scalar element along the second dimension.
    name: A name for the operation (optional).

    Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shapes, dense_values).

    sparse_indices: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    sparse_values: A list of `Tensor` objects of type `sparse_types`.
    sparse_shapes: A list with the same length as `sparse_keys` of `Tensor` objects with type `int64`.
    dense_values: A list of `Tensor` objects. Has the same type as `dense_defaults`.
    """
    if not isinstance(sparse_keys, (list, tuple)):
        raise TypeError(
            "Expected list for 'sparse_keys' argument to "
            "'parse_example' Op, not %r." % sparse_keys)
    _attr_Nsparse = len(sparse_keys)
    if not isinstance(dense_keys, (list, tuple)):
        raise TypeError(
            "Expected list for 'dense_keys' argument to "
            "'parse_example' Op, not %r." % dense_keys)
    _attr_Ndense = len(dense_keys)
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError(
            "Expected list for 'sparse_types' argument to "
            "'parse_example' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, "sparse_types") for _t in sparse_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError(
            "Expected list for 'dense_shapes' argument to "
            "'parse_example' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, "dense_shapes") for _s in dense_shapes]
    _ctx = _context.context()
    if _ctx.in_graph_mode():
        _, _, _op = _op_def_lib._apply_op_helper(
            "ParseAvroRecord", serialized=serialized,
            sparse_keys=sparse_keys, dense_keys=dense_keys,
            dense_defaults=dense_defaults, sparse_types=sparse_types,
            dense_shapes=dense_shapes, schema=schema, name=None)
        _result = _op.outputs[:]
        _inputs_flat = _op.inputs
        _attrs = ("Nsparse", _op.get_attr("Nsparse"), "Ndense",
                  _op.get_attr("Ndense"), "sparse_types",
                  _op.get_attr("sparse_types"), "Tdense", _op.get_attr("Tdense"),
                  "dense_shapes", _op.get_attr("dense_shapes"))
    else:
        _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, _ctx)
        _attr_Tdense = [_t.as_datatype_enum for _t in _attr_Tdense]
        serialized = ops.convert_to_tensor(serialized, dtypes.string)
        sparse_keys = ops.convert_n_to_tensor(sparse_keys, dtypes.string)
        dense_keys = ops.convert_n_to_tensor(dense_keys, dtypes.string)
        _inputs_flat = [serialized] + list(sparse_keys) + list(dense_keys) + list(dense_defaults) + [schema]
        _attrs = ("Nsparse", _attr_Nsparse, "Ndense", _attr_Ndense,
                  "sparse_types", sparse_types, "Tdense", _attr_Tdense,
                  "dense_shapes", dense_shapes)
        _result = _execute.execute(b"ParseAvroRecord", _attr_Nsparse +
                                   len(sparse_types) + _attr_Nsparse +
                                   len(dense_defaults), inputs=_inputs_flat,
                                   attrs=_attrs, ctx=_ctx, name=name)
    _execute.record_gradient(
        "ParseAvroRecord", _inputs_flat, _attrs, _result, name)
    _result = [_result[:_attr_Nsparse]] + _result[_attr_Nsparse:]
    _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
    _result = _result[:2] + [_result[2:2 + _attr_Nsparse]] + _result[2 + _attr_Nsparse:]
    _result = _result[:3] + [_result[3:]]
    _result = _ParseAvroOutput._make(_result)
    return _result


def _build_keys_for_sparse_features(features):
    """
    Builds the fully qualified names for keys of sparse features.

    :param features:  A map of features with keys to TensorFlow features.

    :return: A map of features where for the sparse feature the 'index_key' and the 'value_key' have been expanded
             properly for the parser in the native code.
    """
    if features:
        # NOTE: We iterate over sorted keys to keep things deterministic.
        for key in sorted(features.keys()):
            feature = features[key]
            if isinstance(feature, SparseFeature):
                features[key] = SparseFeature(index_key=key + '/[*]/' + feature.index_key,
                                              value_key=key + '/[*]/' + feature.value_key,
                                              dtype=feature.dtype,
                                              size=feature.size,
                                              already_sorted=feature.already_sorted)
    return features
