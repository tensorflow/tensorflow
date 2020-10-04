/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_svdf.h"

#include <ctype.h>
#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "google/protobuf/map.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/toco_port.h"
#include "tensorflow/lite/toco/tooling_util.h"

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/logging.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

namespace toco {

namespace {

// Receives a vector of cluster nodes and returns only those which are array
// partitions (of type 'Const' and have the pattern 'part_<.*>' in their name.
// Since these nodes are connected to a Concatenate node, it makes sure the
// axis value input of the Concatenate operator is 0.
void FilterPartitionedConstNodes(
    const std::string& const_pattern,
    const std::vector<const NodeDef*>& cluster_nodes,
    std::vector<const NodeDef*>* const_node_parts) {
  for (const NodeDef* node : cluster_nodes) {
    std::string node_name_to_upper = node->name();
    std::transform(node_name_to_upper.begin(), node_name_to_upper.end(),
                   node_name_to_upper.begin(), ::toupper);
    if (StrContains(node->name(), const_pattern) && node->op() == "Const") {
      if (StrContains(node_name_to_upper, "/PART_")) {
        const_node_parts->push_back(node);
      } else if (StrContains(node->name(), "AXIS") &&
                 StrContains(node->name(), "CONCAT")) {
        // For now only supporting Concatenate on Axix 0
        const auto& value_attr = node->attr().at("value");
        const tensorflow::TensorProto& tensor = value_attr.tensor();
        CHECK_EQ(tensor.int_val(0), 0);
      }
    }
  }
  sort(const_node_parts->begin(), const_node_parts->end(),
       [](const NodeDef* a, const NodeDef* b) {
         return (a->name().compare(b->name()) < 0 &&
                 (a->name().size() < b->name().size()));
       });
}

}  // namespace

// SvdfCluster methods

int SvdfCluster::InferFilterRank() {
  for (const NodeDef* node : nodes_) {
    if (StrContains(node->name(), "Reshape/shape")) {
      const auto& value_attr = node->attr().at("value");
      const tensorflow::TensorProto& tensor = value_attr.tensor();
      std::vector<int32> shape_values(
          tensor.tensor_content().size() / sizeof(int), 0);
      port::CopyToBuffer(tensor.tensor_content(),
                         reinterpret_cast<char*>(shape_values.data()));
      CHECK_EQ(shape_values.size(), 3);
      // shape_value array is arranged as:
      // [num_units, rank, -1]
      CHECK_EQ(shape_values[2], -1);
      return shape_values[1];
    }
  }
  return -1;
}

void SvdfCluster::CreateNodes() {
  for (const std::string& const_pattern : const_node_patterns_) {
    CreateConstNode(const_pattern);
  }
  std::unique_ptr<tensorflow::NodeDef> svdf_node(new NodeDef);
  svdf_node->set_op("Svdf");
  svdf_node->set_name(name_);
  svdf_node->set_device(device_);

  // Add the main input.
  svdf_node->add_input(inputs_[0]);

  // Add the rest of the inputs to Svdf cell: weights and bias.
  CHECK(new_nodes_.size() == 3 || new_nodes_.size() == 2);
  std::string* weights_feature_input = svdf_node->add_input();
  std::string* weights_time_input = svdf_node->add_input();
  std::string* bias_input;
  if (new_nodes_.size() == 3) {
    bias_input = svdf_node->add_input();
  }
  for (const std::unique_ptr<tensorflow::NodeDef>& node : new_nodes_) {
    const std::string node_name = node->name();
    if (StrContains(node_name, "SVDF_weights_feature")) {
      *weights_feature_input = node_name;
    } else if (StrContains(node_name, "SVDF_weights_time")) {
      *weights_time_input = node_name;
    } else if (StrContains(node_name, "SVDF_bias")) {
      CHECK(bias_input) << "Bias input cannot be provided when there are only "
                           "two Const input nodes!";
      *bias_input = node_name;
    } else {
      // Unexpected input for Svdf op.
      LOG(FATAL) << "Unexpected input node for SVDF op! Accepted inputs are: "
                    "weights_feature, weights_time and bias.";
    }
  }
  const int rank = InferFilterRank();
  CHECK_GT(rank, 0);

  // Add Svdf activation and rank.
  std::string activation_function =
      StrContains(outputs_[0], "Relu") ? "Relu" : "None";
  (*svdf_node->mutable_attr())["ActivationFunction"].set_s(activation_function);
  (*svdf_node->mutable_attr())["Rank"].set_i(rank);

  // Finally add it to the list of the newly created nodes.
  new_nodes_.push_back(std::move(svdf_node));
}

void SvdfCluster::CreateConstNode(const std::string& const_pattern) {
  // Find the nodes with pattern like: "const_pattern"/part_xxx of type Const.
  std::vector<const NodeDef*> const_node_parts;
  FilterPartitionedConstNodes(const_pattern, nodes_, &const_node_parts);

  if (const_node_parts.empty()) return;

  bool transpose_tensor_value =
      StrContains(const_pattern, "SVDF_weights_feature");

  // Merge them if necessary.
  std::unique_ptr<tensorflow::NodeDef> merged_node(new NodeDef);
  MaybeMergeConstNodes(const_node_parts, transpose_tensor_value, merged_node);
  new_nodes_.push_back(std::move(merged_node));
}

void SvdfCluster::MaybeMergeConstNodes(
    const std::vector<const NodeDef*>& const_node_parts,
    bool transpose_tensor_value,
    const std::unique_ptr<tensorflow::NodeDef>& merged_node) {
  merged_node->set_name(const_node_parts[0]->name());
  merged_node->set_op("Const");
  merged_node->set_device(const_node_parts[0]->device());
  (*merged_node->mutable_attr())["dtype"].set_type(
      const_node_parts[0]->attr().at("dtype").type());

  // Figuring out Value attribute for the merged node.
  // Assuming the partitioning is done on Axis 0.
  // The attributes which are inferred:
  // * Shape and dimensions
  // * Float content values

  // Inferring shape and dimension
  int dim0_size = 0;
  int dim1_size = 1;
  tensorflow::TensorProto* allocated_tensor =
      (*merged_node->mutable_attr())["value"].mutable_tensor();
  tensorflow::TensorShapeProto* allocated_tensor_shape =
      allocated_tensor->mutable_tensor_shape();
  auto tensor_shape_dim0 = allocated_tensor_shape->add_dim();
  int allocated_content_flat_size = 0;
  for (size_t i = 0; i < const_node_parts.size(); i++) {
    const auto& value_attr = const_node_parts[i]->attr().at("value");
    const tensorflow::TensorProto& tensor = value_attr.tensor();
    if (i == 0) {
      allocated_tensor->set_dtype(tensor.dtype());
    } else {
      CHECK_EQ(allocated_tensor->dtype(), tensor.dtype());
    }
    allocated_content_flat_size += tensor.tensor_content().size();
    CHECK(tensor.has_tensor_shape());
    const tensorflow::TensorShapeProto shape = tensor.tensor_shape();
    dim0_size += shape.dim(0).size();
    for (int d = 1; d < shape.dim_size(); d++) {
      if (i == 0) {
        allocated_tensor_shape->add_dim()->set_size(shape.dim(d).size());
        allocated_tensor_shape->set_unknown_rank(shape.unknown_rank());
        dim1_size *= shape.dim(d).size();
      } else {
        CHECK_EQ(shape.dim(d).size(), allocated_tensor_shape->dim(d).size());
        CHECK_EQ(allocated_tensor_shape->unknown_rank(), shape.unknown_rank());
      }
    }
  }

  // Copying the float content from each array partition.
  std::unique_ptr<char[]> allocated_content(
      new char[allocated_content_flat_size]);
  char* content_ptr = allocated_content.get();
  for (size_t i = 0; i < const_node_parts.size(); i++) {
    const auto& value_attr = const_node_parts[i]->attr().at("value");
    const tensorflow::TensorProto& tensor = value_attr.tensor();
    port::CopyToBuffer(tensor.tensor_content(), content_ptr);
    content_ptr += tensor.tensor_content().size();
  }

  // Transpose the tensor if needed.
  if (transpose_tensor_value) {
    // We use dimension 0 to show the row size for the tensor.
    // We use multiplication of the rest of dimension size to for the col size
    // of the tensor.
    std::unique_ptr<float[]> transposed_tensor(
        new float[dim0_size * dim1_size]);
    Transpose2DTensor(reinterpret_cast<float*>(allocated_content.get()),
                      dim0_size, dim1_size, transposed_tensor.get());
    allocated_tensor_shape->clear_dim();
    allocated_tensor_shape->add_dim()->set_size(dim1_size);
    allocated_tensor_shape->add_dim()->set_size(dim0_size);

    // Set the tensor attributes.
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(transposed_tensor.get()),
                    allocated_content_flat_size));
  } else {
    tensor_shape_dim0->set_size(dim0_size);

    // Set the tensor attributes.
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(allocated_content.get()),
                    allocated_content_flat_size));
  }
}

// SvdfClusterFactory methods

std::unique_ptr<Cluster> SvdfClusterFactory::CreateCluster(
    const NodeDef& node, const GraphDef& graph_def) const {
  std::vector<std::string> node_patterns = {"SVDF_weights_feature",
                                            "SVDF_weights_time", "SVDF_bias"};

  std::string node_name_to_upper = node.name();
  std::transform(node_name_to_upper.begin(), node_name_to_upper.end(),
                 node_name_to_upper.begin(), ::toupper);
  std::unique_ptr<SvdfCluster> cluster = nullptr;
  if (node_name_to_upper.find("SVDF", 0) != std::string::npos) {
    size_t weights_pos = node.name().find(node_patterns[0]);
    if (weights_pos != std::string::npos) {
      // Assuming the node name has a pattern like:
      // "SOMESTRING1/CELLNAME/SEARCH_PATTERN/SOMESTRING2", we use
      // CELLNAME as the cluster name.
      size_t cell_pos = node.name().rfind("/", weights_pos - 2) + 1;
      std::string cell_name =
          node.name().substr(cell_pos, weights_pos - cell_pos - 1);
      cluster = std::unique_ptr<SvdfCluster>(new SvdfCluster);
      cluster->SetName(cell_name);
      cluster->SetDevice(node.device());
      cluster->SetGraphDefInfo(&graph_def);
      CHECK(cluster->FindClusterInputsAndOutputs());

      for (const std::string& const_pattern : node_patterns) {
        cluster->AddConstNodePattern(const_pattern);
      }
    }
  }
  return std::move(cluster);
}

}  // end namespace toco
