/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/regularization/util.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow::graph_regularization {

namespace {

GraphDef CreateTestGraph() {
  GraphDef graph_def;
  NodeDef* node = graph_def.add_node();
  node->set_name("name1");
  node->set_op("op1");
  node = graph_def.add_node();
  node->set_name("name2");
  node->set_op("op2");
  return graph_def;
}

TEST(UtilTest, TestGetSuffixUID) { EXPECT_EQ(*GetSuffixUID("foo_32"), 32); }

TEST(UtilTest, TestGetSuffixUID64Bit) {
  EXPECT_EQ(*GetSuffixUID("foo_2209431640"), 2209431640);
}

TEST(UtilTest, TestGetSuffixUIDInvalid) {
  EXPECT_FALSE(GetSuffixUID("foo").ok());
}

TEST(FingerprintingTest, TestComputeHash) {
  GraphDef graph_def = CreateTestGraph();
  EXPECT_EQ(ComputeHash(graph_def), 4870331646167591885);
}
}  // namespace
}  // namespace tensorflow::graph_regularization
