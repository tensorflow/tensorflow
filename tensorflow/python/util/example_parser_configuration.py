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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util


def extract_example_parser_configuration(parse_example_op, sess):
  """Returns an ExampleParserConfig proto.

  Args:
    parse_example_op: A ParseExample `Operation`
    sess: A tf.Session needed to obtain some configuration values.
  Returns:
    A ExampleParserConfig proto.

  Raises:
    ValueError: If attributes are inconsistent.
  """
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
    raise ValueError("len(fetch_list) does not match total features + num_dense"
                     "(%d vs %d" % (len(fetch_list),
                                    (total_features + num_dense)))

  fetched = sess.run(fetch_list)

  if len(fetched) != len(fetch_list):
    raise ValueError("len(fetched) does not match len(fetch_list)"
                     "(%d vs %d" % (len(fetched), len(fetch_list)))

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

    fixed_config.dtype = dense_types[i]
    # Get the output tensor name.
    fixed_config.values_output_tensor_name = parse_example_op.outputs[
        dense_values_start + i].name

  # Sparse features.
  for i in range(num_sparse):
    key = fetched[sparse_keys_start + i]
    feature_config = config.feature_map[key]
    var_len_feature = feature_config.var_len_feature
    var_len_feature.dtype = sparse_types[i]
    var_len_feature.indices_output_tensor_name = parse_example_op.outputs[
        sparse_indices_start + i].name
    var_len_feature.values_output_tensor_name = parse_example_op.outputs[
        sparse_values_start + i].name
    var_len_feature.shapes_output_tensor_name = parse_example_op.outputs[
        sparse_shapes_start + i].name

  return config
