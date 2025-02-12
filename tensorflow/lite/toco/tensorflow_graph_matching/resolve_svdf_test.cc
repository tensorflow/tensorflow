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

#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/cluster_utils.h"
#include "tensorflow/lite/toco/tensorflow_graph_matching/resolve_cluster.h"
#include "tensorflow/lite/toco/toco_port.h"

using tensorflow::GraphDef;
using tensorflow::NodeDef;

namespace toco {

class ResolveSvdfTest : public ::testing::Test {
 public:
  ResolveSvdfTest() {
    AddNewNode("Input1", "Const", {});
    AddNewNode("Svdf1/SVDF_weights_feature/part_0", "Const", {},
               {0.1, 0.2, 0.3});
    AddNewNode("Svdf1/SVDF_weights_feature/part_0/read", "Identity",
               {"Svdf1/SVDF_weights_feature/part_0"});
    AddNewNode("Svdf1/SVDF_weights_time/part_0", "Const", {}, {0.1, 0.2, 0.3});
    AddNewNode("Svdf1/SVDF_weights_time/part_0/read", "Identity",
               {"Svdf1/SVDF_weights_time/part_0"});

    AddNewNode("Svdf1/f1", "SVDF_F1",
               {"Input1", "Svdf1/SVDF_weights_feature/part_0/read"});
    AddNewNode("Svdf1/f2", "SVDF_F2",
               {"Svdf1/SVDF_weights_time/part_0/read", "Svdf1/f1"});
    AddNewNode("Svdf1/Relu", "Relu", {"Svdf1/f2"});
    AddShapeNode("Svdf1/Reshape/shape", {10, 1, -1});
    AddNewNode("Output1", "Const", {"Svdf1/Relu"});

    AddNewNode("Input2", "Const", {});
    AddNewNode("Svdf2/SVDF_weights_feature/part_0", "Const", {},
               {0.1, 0.2, 0.3});
    AddNewNode("Svdf2/SVDF_weights_feature/part_0/read", "Identity",
               {"Svdf2/SVDF_weights_feature/part_0"});
    AddNewNode("Svdf2/SVDF_weights_time/part_0", "Const", {}, {0.1, 0.2, 0.3});
    AddNewNode("Svdf2/SVDF_weights_time/part_0/read", "Identity",
               {"Svdf2/SVDF_weights_time/part_0"});

    AddNewNode("Svdf2/f1", "SVDF_F1",
               {"Input1", "Svdf2/SVDF_weights_feature/part_0/read"});
    AddNewNode("Svdf2/f2", "SVDF_F2",
               {"Svdf2/SVDF_weights_time/part_0/read", "Svdf2/f1"});
    AddNewNode("Svdf2/Relu", "Relu", {"Svdf2/f2"});
    AddShapeNode("Svdf2/Reshape/shape", {10, 2, -1});
    AddNewNode("Output2", "Const", {"Svdf2/Relu"});
  }

  ~ResolveSvdfTest() override {}

 protected:
  void AddNewNode(const std::string& name, const std::string& op,
                  const std::vector<std::string>& inputs) {
    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op(op);
    node->set_device("");
    for (int i = 0; i < inputs.size(); i++) {
      node->add_input();
      node->set_input(i, inputs[i]);
    }
  }

  void AddNewNode(const std::string& name, const std::string& op,
                  const std::vector<std::string>& inputs,
                  const std::vector<float>& values) {
    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op(op);
    node->set_device("");
    for (int i = 0; i < inputs.size(); i++) {
      node->add_input();
      node->set_input(i, inputs[i]);
    }
    // Add the float vector as an attribute to the node.
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_FLOAT);
    tensorflow::TensorProto* allocated_tensor = new tensorflow::TensorProto;
    tensorflow::TensorShapeProto* allocated_tensor_shape =
        new tensorflow::TensorShapeProto;
    auto tensor_shape_dim0 = allocated_tensor_shape->add_dim();
    tensor_shape_dim0->set_size(values.size());
    allocated_tensor->set_allocated_tensor_shape(allocated_tensor_shape);
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(float)));
    (*node->mutable_attr())["value"].set_allocated_tensor(allocated_tensor);
  }

  void AddShapeNode(const std::string& name, const std::vector<int>& values) {
    NodeDef* node = graph_.add_node();
    node->set_name(name);
    node->set_op("Const");
    node->set_device("");
    // Add the float vector as an attribute to the node.
    (*node->mutable_attr())["dtype"].set_type(tensorflow::DT_INT32);
    tensorflow::TensorProto* allocated_tensor = new tensorflow::TensorProto;
    tensorflow::TensorShapeProto* allocated_tensor_shape =
        new tensorflow::TensorShapeProto;
    auto tensor_shape_dim0 = allocated_tensor_shape->add_dim();
    tensor_shape_dim0->set_size(values.size());
    allocated_tensor->set_allocated_tensor_shape(allocated_tensor_shape);
    allocated_tensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(values.data()),
                    values.size() * sizeof(int)));
    (*node->mutable_attr())["value"].set_allocated_tensor(allocated_tensor);
  }

  GraphDef graph_;
  SvdfClusterFactory svdf_cluster_factory_;
  std::vector<std::unique_ptr<Cluster>> clusters_;
};

TEST_F(ResolveSvdfTest, TestTranspose2DTensor) {
  static float matrix[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
  static float expected_transposed_matrix[] = {1., 5., 9.,  2., 6., 10.,
                                               3., 7., 11., 4., 8., 12.};
  float* transposed_matrix = new float[12];
  Transpose2DTensor(matrix, 3, 4, transposed_matrix);

  std::vector<float> actual;
  actual.insert(
      actual.end(), transposed_matrix,
      transposed_matrix + sizeof(expected_transposed_matrix) / sizeof(float));
  std::vector<float> expected;
  expected.insert(expected.end(), expected_transposed_matrix,
                  expected_transposed_matrix +
                      sizeof(expected_transposed_matrix) / sizeof(float));
  delete[] transposed_matrix;
}

TEST_F(ResolveSvdfTest, TestResolveSvdfFlow) {
  std::unordered_map<std::string, bool> is_node_in_cluster;
  for (const NodeDef& node : graph_.node()) {
    is_node_in_cluster[node.name()] = false;
  }

  std::vector<std::string> cluster_names;
  CHECK(FindCluster(svdf_cluster_factory_, graph_, &is_node_in_cluster,
                    &clusters_));

  for (const std::unique_ptr<Cluster>& cluster : clusters_) {
    cluster_names.push_back(cluster->GetName());
    cluster->CreateNodes();
  }

  EXPECT_THAT(cluster_names,
              testing::UnorderedElementsAreArray({"Svdf1", "Svdf2"}));

  std::vector<std::string> new_node_names;
  std::vector<float> content_array(3);
  for (const std::unique_ptr<Cluster>& cluster : clusters_) {
    // After CreateNodes in each cluster we have three nodes: Svdf,
    // weights_feature and weights_time.
    CHECK_EQ(cluster->GetNewNodes().size(), 3);
    for (const std::unique_ptr<tensorflow::NodeDef>& node :
         cluster->GetNewNodes()) {
      new_node_names.push_back(node->name());
      if (node->op() == "Const") {
        CHECK_EQ(node->attr().at("dtype").type(), tensorflow::DT_FLOAT);
        toco::port::CopyToBuffer(
            node->attr().at("value").tensor().tensor_content(),
            reinterpret_cast<char*>(content_array.data()));
        EXPECT_THAT(content_array,
                    testing::UnorderedElementsAreArray({0.1, 0.2, 0.3}));
      } else {
        // Checking the Svdf node attributes (rank and activation type) are
        // correct.
        if (node->name() == "Svdf1") {
          CHECK_EQ(node->attr().at("Rank").i(), 1);
        } else if (node->name() == "Svdf2") {
          CHECK_EQ(node->attr().at("Rank").i(), 2);
        }
        CHECK_EQ(node->attr().at("ActivationFunction").s(), "Relu");
      }
    }
  }
  EXPECT_THAT(new_node_names, testing::UnorderedElementsAreArray(
                                  {"Svdf2/SVDF_weights_feature/part_0",
                                   "Svdf2/SVDF_weights_time/part_0", "Svdf2",
                                   "Svdf1/SVDF_weights_feature/part_0",
                                   "Svdf1/SVDF_weights_time/part_0", "Svdf1"}));
}

}  // end namespace toco
