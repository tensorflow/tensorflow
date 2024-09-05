/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/example/example_parser_configuration.h"

#include <vector>

#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

Status FindNodeIndexByName(const tensorflow::GraphDef& graph,
                           const string& node_name, int* node_idx) {
  for (int i = 0; i < graph.node_size(); ++i) {
    const auto& node = graph.node(i);
    if (node.name() == node_name) {
      *node_idx = i;
      return absl::OkStatus();
    }
  }
  return errors::InvalidArgument(node_name, " not found in GraphDef");
}

Status ExtractExampleParserConfiguration(
    const tensorflow::GraphDef& graph, const string& node_name,
    tensorflow::Session* session,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features) {
  int node_idx;
  TF_RETURN_IF_ERROR(FindNodeIndexByName(graph, node_name, &node_idx));

  const auto& node = graph.node(node_idx);
  if (node.op() != "ParseExample") {
    return errors::InvalidArgument(node_name, " node is not a ParseExample op");
  }

  auto& attr_map = node.attr();
  auto num_sparse = attr_map.at("Nsparse").i();
  auto num_dense = attr_map.at("Ndense").i();
  fixed_len_features->resize(num_dense);
  var_len_features->resize(num_sparse);

  auto tdense = attr_map.at("Tdense");
  auto dense_shapes = attr_map.at("dense_shapes");
  auto sparse_types = attr_map.at("sparse_types");

  // Consistency check attributes.
  if (tdense.list().type_size() != num_dense) {
    return errors::InvalidArgument("Node attr Tdense has ",
                                   tdense.list().type_size(),
                                   " elements != Ndense attr: ", num_dense);
  }

  if (dense_shapes.list().shape_size() != num_dense) {
    return errors::InvalidArgument("Node attr dense_shapes has ",
                                   dense_shapes.list().shape_size(),
                                   " elements != Ndense attr: ", num_dense);
  }

  if (sparse_types.list().type_size() != num_sparse) {
    return errors::InvalidArgument("Node attr sparse_types has ",
                                   sparse_types.list().type_size(),
                                   " elements != NSparse attr: ", num_sparse);
  }

  for (int i = 0; i < tdense.list().type_size(); ++i) {
    (*fixed_len_features)[i].dtype = tdense.list().type(i);
    // Convert TensorShapeProto to TensorShape.
    (*fixed_len_features)[i].shape = TensorShape(dense_shapes.list().shape(i));
  }

  for (int i = 0; i < sparse_types.list().type_size(); ++i) {
    (*var_len_features)[i].dtype = sparse_types.list().type(i);
  }

  // We must fetch the configuration input tensors to the ParseExample op.
  // Skipping index = 0, which is the serialized proto input.
  std::vector<string> fetch_names(node.input_size() - 1);
  for (int i = 1; i < node.input_size(); ++i) {
    fetch_names[i - 1] = node.input(i);
  }

  std::vector<Tensor> op_input_tensors;

  TF_RETURN_IF_ERROR(session->Run({},               // no_inputs,
                                  fetch_names, {},  // no target_node_names,
                                  &op_input_tensors));

  // The input tensors are laid out sequentially in a flat manner.
  // Here are the various start offsets.
  int sparse_keys_start = 1;
  int dense_keys_start = sparse_keys_start + num_sparse;
  int dense_defaults_start = dense_keys_start + num_dense;

  for (int i = 0; i < num_sparse; ++i) {
    int input_idx = sparse_keys_start + i;
    (*var_len_features)[i].key =
        op_input_tensors[input_idx].scalar<tstring>()();
  }

  for (int i = 0; i < num_dense; ++i) {
    FixedLenFeature& config = (*fixed_len_features)[i];
    int dense_keys_offset = dense_keys_start + i;
    config.key = op_input_tensors[dense_keys_offset].scalar<tstring>()();

    int defaults_offset = dense_defaults_start + i;
    config.default_value = op_input_tensors[defaults_offset];
  }

  // The output tensors are laid out sequentially in a flat manner.
  // Here are the various start offsets.
  int sparse_indices_output_start = 0;
  int sparse_values_output_start = sparse_indices_output_start + num_sparse;
  int sparse_shapes_output_start = sparse_values_output_start + num_sparse;
  int dense_values_output_start = sparse_shapes_output_start + num_sparse;

  string node_output_prefix = strings::StrCat(node_name, ":");

  for (int i = 0; i < num_sparse; ++i) {
    VarLenFeature& config = (*var_len_features)[i];

    int indices_offset = sparse_indices_output_start + i;
    config.indices_output_tensor_name =
        strings::StrCat(node_output_prefix, indices_offset);

    int values_offset = sparse_values_output_start + i;
    config.values_output_tensor_name =
        strings::StrCat(node_output_prefix, values_offset);

    int shapes_offset = sparse_shapes_output_start + i;
    config.shapes_output_tensor_name =
        strings::StrCat(node_output_prefix, shapes_offset);
  }

  for (int i = 0; i < num_dense; ++i) {
    int output_idx = dense_values_output_start + i;
    (*fixed_len_features)[i].values_output_tensor_name =
        strings::StrCat(node_output_prefix, output_idx);
  }
  return absl::OkStatus();
}

Status ExampleParserConfigurationProtoToFeatureVectors(
    const ExampleParserConfiguration& config_proto,
    std::vector<FixedLenFeature>* fixed_len_features,
    std::vector<VarLenFeature>* var_len_features) {
  const auto& feature_map = config_proto.feature_map();
  for (auto it = feature_map.cbegin(); it != feature_map.cend(); ++it) {
    string key = it->first;
    const auto& config = it->second;
    if (config.has_fixed_len_feature()) {
      const auto& fixed_config = config.fixed_len_feature();
      FixedLenFeature f;
      f.key = key;
      f.dtype = fixed_config.dtype();
      f.shape = TensorShape(fixed_config.shape());
      Tensor default_value(f.dtype, f.shape);
      if (!default_value.FromProto(fixed_config.default_value())) {
        return errors::InvalidArgument(
            "Invalid default_value in config proto ",
            fixed_config.default_value().DebugString());
      }
      f.default_value = default_value;
      f.values_output_tensor_name = fixed_config.values_output_tensor_name();
      fixed_len_features->push_back(f);
    } else {
      const auto& var_len_config = config.var_len_feature();
      VarLenFeature v;
      v.key = key;
      v.dtype = var_len_config.dtype();
      v.values_output_tensor_name = var_len_config.values_output_tensor_name();
      v.indices_output_tensor_name =
          var_len_config.indices_output_tensor_name();
      v.shapes_output_tensor_name = var_len_config.shapes_output_tensor_name();
      var_len_features->push_back(v);
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
