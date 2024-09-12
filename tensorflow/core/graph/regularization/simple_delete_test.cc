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

#include "tensorflow/core/graph/regularization/simple_delete.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/graph/regularization/util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/saved_model.pb.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::graph_regularization {

namespace {

absl::StatusOr<SavedModel> ReadSavedModel(absl::string_view file_dir) {
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

// Test that SimpleDelete algorithm returns the same hash for two models saved
// by calling `tf.saved_model.save` twice in a row in the same program.
TEST(SimpleDeleteTest, TestSimpleDeleteModelSavedTwice) {
  const std::string export_dir =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/graph/regularization/testdata", "bert1");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb,
                          ReadSavedModel(export_dir));

  MetaGraphDef* metagraph = saved_model_pb.mutable_meta_graphs(0);
  GraphDef* graph_def = metagraph->mutable_graph_def();
  SimpleDelete(*graph_def);
  uint64 hash1 = ComputeHash(*graph_def);

  const std::string export_dir2 =
      io::JoinPath(testing::TensorFlowSrcRoot(),
                   "core/graph/regularization/testdata", "bert2");
  TF_ASSERT_OK_AND_ASSIGN(SavedModel saved_model_pb2,
                          ReadSavedModel(export_dir2));
  const MetaGraphDef& metagraph2 = saved_model_pb2.meta_graphs(0);
  GraphDef graph_def2 = metagraph2.graph_def();
  SimpleDelete(graph_def2);
  uint64 hash2 = ComputeHash(graph_def2);

  EXPECT_EQ(hash1, hash2);
}

}  // namespace
}  // namespace tensorflow::graph_regularization
