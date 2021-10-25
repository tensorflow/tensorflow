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
"""Extract parse_example op configuration to a proto."""

from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def extract_example_parser_configuration(parse_example_op, sess):
  """Returns an ExampleParserConfig proto.

  Args:
    parse_example_op: A ParseExample or ParseExampleV2 `Operation`
    sess: A tf.compat.v1.Session needed to obtain some configuration values.
  Returns:
    A ExampleParserConfig proto.

  Raises:
    ValueError: If attributes are inconsistent.
  """
  if parse_example_op.type == "ParseExample":
    return _extract_from_parse_example(parse_example_op, sess)
  elif parse_example_op.type == "ParseExampleV2":
    return _extract_from_parse_example_v2(parse_example_op, sess)
  else:
    raise ValueError(
        "Found unexpected type when parsing example. Expected `ParseExample` "
        f"object. Received type: {parse_example_op.type}")


def _extract_from_parse_example(parse_example_op, sess):
  """Extract ExampleParserConfig from ParseExample op."""
  config = example_parser_configuration_pb2.ExampleParserConfiguration()

  num_sparse = parse_example_op.get_attr("Nsparse")
  num_dense = parse_example_op.get_attr("Ndense")
  total_features = num_dense + num_sparse

  sparse_types = parse_example_op.get_attr("sparse_types")
  dense_types = parse_example_op.get_attr("Tdense")
  dense_shapes = parse_example_op.get_attr("dense_shapes")

  if len(sparse_types) != num_sparse:
    raise ValueError("len(sparse_types) attribute does not match "
                     "Nsparse attribute (%d vs %d)" %
                     (len(sparse_types), num_sparse))

  if len(dense_types) != num_dense:
    raise ValueError("len(dense_types) attribute does not match "
                     "Ndense attribute (%d vs %d)" %
                     (len(dense_types), num_dense))

  if len(dense_shapes) != num_dense:
    raise ValueError("len(dense_shapes) attribute does not match "
                     "Ndense attribute (%d vs %d)" %
                     (len(dense_shapes), num_dense))

  # Skip over the serialized input, and the names input.
  fetch_list = parse_example_op.inputs[2:]

  # Fetch total_features key names and num_dense default values.
  if len(fetch_list) != (total_features + num_dense):
    raise ValueError("len(fetch_list) does not match total features + "
                     "num_dense (%d vs %d)" %
                     (len(fetch_list), (total_features + num_dense)))

  fetched = sess.run(fetch_list)

  if len(fetched) != len(fetch_list):
    raise ValueError("len(fetched) does not match len(fetch_list) "
                     "(%d vs %d)" % (len(fetched), len(fetch_list)))

  # Fetch indices.
  sparse_keys_start = 0
  dense_keys_start = sparse_keys_start + num_sparse
  dense_def_start = dense_keys_start + num_dense

  # Output tensor indices.
  sparse_indices_start = 0
  sparse_values_start = num_sparse
  sparse_shapes_start = sparse_values_start + num_sparse
  dense_values_start = sparse_shapes_start + num_sparse

  # Dense features.
  for i in range(num_dense):
    key = fetched[dense_keys_start + i]
    feature_config = config.feature_map[key]
    # Convert the default value numpy array fetched from the session run
    # into a TensorProto.
    fixed_config = feature_config.fixed_len_feature

    fixed_config.default_value.CopyFrom(
        tensor_util.make_tensor_proto(fetched[dense_def_start + i]))
    # Convert the shape from the attributes
    # into a TensorShapeProto.
    fixed_config.shape.CopyFrom(
        tensor_shape.TensorShape(dense_shapes[i]).as_proto())

    fixed_config.dtype = dense_types[i].as_datatype_enum
    # Get the output tensor name.
    fixed_config.values_output_tensor_name = parse_example_op.outputs[
        dense_values_start + i].name

  # Sparse features.
  for i in range(num_sparse):
    key = fetched[sparse_keys_start + i]
    feature_config = config.feature_map[key]
    var_len_feature = feature_config.var_len_feature
    var_len_feature.dtype = sparse_types[i].as_datatype_enum
    var_len_feature.indices_output_tensor_name = parse_example_op.outputs[
        sparse_indices_start + i].name
    var_len_feature.values_output_tensor_name = parse_example_op.outputs[
        sparse_values_start + i].name
    var_len_feature.shapes_output_tensor_name = parse_example_op.outputs[
        sparse_shapes_start + i].name

  return config


def _extract_from_parse_example_v2(parse_example_op, sess):
  """Extract ExampleParserConfig from ParseExampleV2 op."""
  config = example_parser_configuration_pb2.ExampleParserConfiguration()

  dense_types = parse_example_op.get_attr("Tdense")
  num_sparse = parse_example_op.get_attr("num_sparse")
  sparse_types = parse_example_op.get_attr("sparse_types")
  ragged_value_types = parse_example_op.get_attr("ragged_value_types")
  ragged_split_types = parse_example_op.get_attr("ragged_split_types")
  dense_shapes = parse_example_op.get_attr("dense_shapes")

  num_dense = len(dense_types)
  num_ragged = len(ragged_value_types)
  assert len(ragged_value_types) == len(ragged_split_types)
  assert len(parse_example_op.inputs) == 5 + num_dense

  # Skip over the serialized input, and the names input.
  fetched = sess.run(parse_example_op.inputs[2:])
  sparse_keys = fetched[0].tolist()
  dense_keys = fetched[1].tolist()
  ragged_keys = fetched[2].tolist()
  dense_defaults = fetched[3:]
  assert len(sparse_keys) == num_sparse
  assert len(dense_keys) == num_dense
  assert len(ragged_keys) == num_ragged

  # Output tensor indices.
  sparse_indices_start = 0
  sparse_values_start = num_sparse
  sparse_shapes_start = sparse_values_start + num_sparse
  dense_values_start = sparse_shapes_start + num_sparse
  ragged_values_start = dense_values_start + num_dense
  ragged_row_splits_start = ragged_values_start + num_ragged

  # Dense features.
  for i in range(num_dense):
    key = dense_keys[i]
    feature_config = config.feature_map[key]
    # Convert the default value numpy array fetched from the session run
    # into a TensorProto.
    fixed_config = feature_config.fixed_len_feature

    fixed_config.default_value.CopyFrom(
        tensor_util.make_tensor_proto(dense_defaults[i]))
    # Convert the shape from the attributes
    # into a TensorShapeProto.
    fixed_config.shape.CopyFrom(
        tensor_shape.TensorShape(dense_shapes[i]).as_proto())

    fixed_config.dtype = dense_types[i].as_datatype_enum
    # Get the output tensor name.
    fixed_config.values_output_tensor_name = parse_example_op.outputs[
        dense_values_start + i].name

  # Sparse features.
  for i in range(num_sparse):
    key = sparse_keys[i]
    feature_config = config.feature_map[key]
    var_len_feature = feature_config.var_len_feature
    var_len_feature.dtype = sparse_types[i].as_datatype_enum
    var_len_feature.indices_output_tensor_name = parse_example_op.outputs[
        sparse_indices_start + i].name
    var_len_feature.values_output_tensor_name = parse_example_op.outputs[
        sparse_values_start + i].name
    var_len_feature.shapes_output_tensor_name = parse_example_op.outputs[
        sparse_shapes_start + i].name

  if num_ragged != 0:
    del ragged_values_start  # unused
    del ragged_row_splits_start  # unused
    raise ValueError("Ragged features are not yet supported by "
                     "example_parser_configuration.proto")

  return config
