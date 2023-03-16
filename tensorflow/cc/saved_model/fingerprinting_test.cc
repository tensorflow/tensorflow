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
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/fingerprint.pb.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"

namespace tensorflow::saved_model::fingerprinting {

namespace {

StatusOr<SavedModel> ReadSavedModel(absl::string_view file_dir) {
  std::string file_path = io::JoinPath(file_dir, "saved_model.pb");
  std::string serialized_saved_model;
  auto status =
      ReadFileToString(Env::Default(), file_path, &serialized_saved_model);
  if (!status.ok()) {
    return status;
  }
  SavedModel saved_model_pb;
  saved_model_pb.ParseFromString(serialized_saved_model);
  return saved_model_pb;
}


TEST(FingerprintingTest, TestCreateFingerprint) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "VarsAndArithmeticObjectGraph");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def,
                          CreateFingerprintDef(saved_model_pb, export_dir));

  EXPECT_GT(fingerprint_def.saved_model_checksum(), 0);
  EXPECT_EQ(fingerprint_def.graph_def_program_hash(), 10127142238652115842U);
  EXPECT_EQ(fingerprint_def.signature_def_hash(), 5693392539583495303);
  EXPECT_EQ(fingerprint_def.saved_object_graph_hash(), 3678101440349108924);
  // TODO(b/242348400): The checkpoint hash is non-deterministic, so we cannot
  // check its value here.
  EXPECT_GT(fingerprint_def.checkpoint_hash(), 0);
}

// Compare the fingerprints of two models saved by calling
// `tf.saved_model.save` twice in a row in the same program.
TEST(FingerprintingTest, TestCompareFingerprintForTwoModelSavedTwice) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "bert1");

  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def,
                          CreateFingerprintDef(saved_model_pb, export_dir));

  const std::string export_dir2 = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "bert2");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb2,
                          ReadSavedModel(export_dir2));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def2,
                          CreateFingerprintDef(saved_model_pb2, export_dir2));

  EXPECT_EQ(fingerprint_def.graph_def_program_hash(),
            fingerprint_def2.graph_def_program_hash());
  EXPECT_EQ(fingerprint_def.signature_def_hash(),
            fingerprint_def2.signature_def_hash());
  EXPECT_EQ(fingerprint_def.saved_object_graph_hash(),
            fingerprint_def2.saved_object_graph_hash());
}

TEST(FingerprintingTest, TestFingerprintComputationDoesNotMutateModel) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "bert1");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def,
                          CreateFingerprintDef(saved_model_pb, export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def2,
                          CreateFingerprintDef(saved_model_pb, export_dir));

  EXPECT_EQ(fingerprint_def.saved_model_checksum(),
            fingerprint_def2.saved_model_checksum());
}

TEST(FingerprintingTest, TestFingerprintHasVersion) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "bert1");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def,
                          CreateFingerprintDef(saved_model_pb, export_dir));
  EXPECT_EQ(fingerprint_def.version().producer(), 1);
}

TEST(FingerprintingTest, TestHashCheckpointForModelWithNoVariables) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "bert1");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_def,
                          CreateFingerprintDef(saved_model_pb, export_dir));
  EXPECT_EQ(fingerprint_def.checkpoint_hash(), 0);
}

TEST(FingerprintingTest, TestReadValidFingerprint) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(), "cc/saved_model/testdata",
                   "VarsAndArithmeticObjectGraph");
  TF_ASSERT_OK_AND_ASSIGN(FingerprintDef fingerprint_pb,
                          ReadSavedModelFingerprint(export_dir));
  EXPECT_EQ(fingerprint_pb.saved_model_checksum(), 15788619162413586750u);
}

TEST(FingerprintingTest, TestReadNonexistentFingerprint) {
  const std::string export_dir = io::JoinPath(
      testing::TensorFlowSrcRoot(), "cc/saved_model/testdata", "AssetModule");
  EXPECT_FALSE(ReadSavedModelFingerprint(export_dir).ok());
}

}  // namespace
}  // namespace tensorflow::saved_model::fingerprinting
