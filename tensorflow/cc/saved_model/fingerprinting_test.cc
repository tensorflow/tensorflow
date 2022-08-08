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
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
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

StatusOr<SavedModel> ReadSavedModel(absl::string_view file_path) {
  std::string serialized_saved_model;
  auto status = ReadFileToString(Env::Default(), std::string(file_path),
                                 &serialized_saved_model);
  if (!status.ok()) {
    return status;
  }
  SavedModel saved_model_pb;
  saved_model_pb.ParseFromString(serialized_saved_model);
  return saved_model_pb;
}

TEST(FingerprintingTest, TestComputeHash) {
  GraphDef graph_def = CreateTestProto();
  EXPECT_EQ(ComputeHash(graph_def), 4870331646167591885);
}

TEST(FingerprintingTest, TestCreateFingerprint) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "VarsAndArithmeticObjectGraph", "saved_model.pb");

  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  FingerprintDef fingerprint_def =
      CreateFingerprintDef(saved_model_pb.meta_graphs(0));

  EXPECT_GT(fingerprint_def.graph_def_checksum(), 0);
  EXPECT_EQ(fingerprint_def.signature_def_hash(), 5693392539583495303);
  EXPECT_EQ(fingerprint_def.saved_object_graph_hash(), 3678101440349108924);
}

// Test that canonicalization returns the same hash for two models saved by
// calling `tf.saved_model.save` twice in a row in the same program.
TEST(FingerprintingTest, TestCanonicalizeGraphDeforModelSavedTwice) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "bert1", "saved_model.pb");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));

  MetaGraphDef* metagraph = saved_model_pb.mutable_meta_graphs(0);
  GraphDef* graph_def = metagraph->mutable_graph_def();
  CanonicalizeGraphDef(*graph_def);
  uint64 hash1 = ComputeHash(*graph_def);

  const std::string export_dir2 =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "bert2", "saved_model.pb");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb2,
                          ReadSavedModel(export_dir2));
  const MetaGraphDef& metagraph2 = saved_model_pb2.meta_graphs(0);
  GraphDef graph_def2 = metagraph2.graph_def();
  CanonicalizeGraphDef(graph_def2);
  uint64 hash2 = ComputeHash(graph_def2);

  EXPECT_EQ(hash1, hash2);
}

// Compare the fingerprints of two models saved by calling
// `tf.saved_model.save` twice in a row in the same program.
TEST(FingerprintingTest, TestCompareFingerprintForTwoModelSavedTwice) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "bert1", "saved_model.pb");

  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  FingerprintDef fingerprint_def =
      CreateFingerprintDef(saved_model_pb.meta_graphs(0));

  const std::string export_dir2 =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "bert2", "saved_model.pb");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb2,
                          ReadSavedModel(export_dir2));
  FingerprintDef fingerprint_def2 =
      CreateFingerprintDef(saved_model_pb2.meta_graphs(0));

  EXPECT_EQ(fingerprint_def.graph_def_program_hash(),
            fingerprint_def2.graph_def_program_hash());
  EXPECT_EQ(fingerprint_def.signature_def_hash(),
            fingerprint_def2.signature_def_hash());
  EXPECT_EQ(fingerprint_def.saved_object_graph_hash(),
            fingerprint_def2.saved_object_graph_hash());
}

TEST(FingerprintingTest, TestFingerprintComputationDoesNotMutateModel) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "bert1", "saved_model.pb");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  FingerprintDef fingerprint_def =
      CreateFingerprintDef(saved_model_pb.meta_graphs(0));
  FingerprintDef fingerprint_def2 =
      CreateFingerprintDef(saved_model_pb.meta_graphs(0));

  EXPECT_EQ(fingerprint_def.graph_def_checksum(),
            fingerprint_def2.graph_def_checksum());
}

}  // namespace
}  // namespace tensorflow::fingerprinting
