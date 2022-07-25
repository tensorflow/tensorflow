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

#include "tensorflow/cc/saved_model/fingerprinting.h"

#include <string>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow::fingerprinting {

namespace {

GraphDef CreateTestProto() {
  GraphDef graph_def;
  NodeDef* node = graph_def.add_node();
  node->set_name("name1");
  node->set_op("op1");
  node = graph_def.add_node();
  node->set_name("name2");
  node->set_op("op2");
  return graph_def;
}

TEST(FingerprintingTest, TestComputeHash) {
  GraphDef graph_def = CreateTestProto();
  EXPECT_EQ(ComputeHash(graph_def), 4870331646167591885);
}

TEST(FingerprintingTest, TestCreateFingerprint) {
  // Read a SavedModel from disk.
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "VarsAndArithmeticObjectGraph", "saved_model.pb");
  std::string serialized_saved_model;
  TF_EXPECT_OK(
      ReadFileToString(Env::Default(), export_dir, &serialized_saved_model));

  SavedModel saved_model_pb;
  saved_model_pb.ParseFromString(serialized_saved_model);
  MetaGraphDef metagraph = saved_model_pb.meta_graphs(0);

  FingerprintDef fingerprint_def = CreateFingerprintDef(metagraph);
  EXPECT_GT(fingerprint_def.graph_def_hash(), 0);
}

}  // namespace
}  // namespace tensorflow::fingerprinting
