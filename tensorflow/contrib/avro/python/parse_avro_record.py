import collections
import os
import re

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape

from tensorflow.python.framework import load_library
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.parsing_ops import _features_to_raw_params, \
    _prepend_none_dimension, VarLenFeature, SparseFeature, FixedLenFeature, \
    FixedLenSequenceFeature, _construct_sparse_tensors_for_sparse_features
from tensorflow.contrib.avro.ops.gen_parse_avro_record import \
    parse_avro_record as _parse_avro_record

# Load the shared library
this_dir = os.path.dirname(os.path.abspath(__file__))
lib_name = os.path.join(this_dir, '_parse_avro_record.so')  # Display the operators with lib_name.OP_LIST
parse_module = load_library.load_op_library(lib_name)  # Load the library with dependent so's in '.'


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
