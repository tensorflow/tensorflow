/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/debug/node_io_dump_rewriter.h"

#include <dirent.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr absl::string_view kDumpSubDirName = "node-io-dump";

const Node* FindNode(const Graph* graph, absl::string_view node_name) {
  for (Node* node : graph->nodes()) {
    if (node->name() == node_name) return node;
  }
  return nullptr;
}

const Node* GetInputNode(const Node* node, size_t index) {
  const Node* input_node;
  CHECK(node->input_node(index, &input_node).ok());
  return input_node;
}

const Node* GetOutputNode(const Node* node, size_t index) {
  for (const Edge* edge : node->out_edges()) {
    if (edge->src_output() == index) return edge->dst();
  }
  return nullptr;
}

absl::StatusOr<std::vector<std::string>> GetFilenames(
    absl::string_view dump_dir) {
  // Read the step directory names.
  auto dump_sub_dir = absl::StrCat(dump_dir, "/", kDumpSubDirName);
  DIR* dir = opendir(dump_sub_dir.data());
  if (dir == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("can't open directory: ", dump_sub_dir));
  }
  std::vector<std::string> step_dirs;
  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (entry->d_type != DT_DIR) {
      return absl::InternalError(absl::StrCat(
          "Found non-directory entry under dump_sub_dir: ", entry->d_name));
    }
    step_dirs.push_back(absl::StrCat(dump_sub_dir, "/", entry->d_name));
  }
  closedir(dir);
  CHECK_EQ(step_dirs.size(), 1);
  // Read the filenames.
  dir = opendir(step_dirs[0].data());
  if (dir == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("can't open directory: ", step_dirs[0]));
  }
  std::vector<std::string> filenames;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }
    if (entry->d_type == DT_DIR) {
      return absl::InternalError(absl::StrCat(
          "Found directory entry under step_dir: ", entry->d_name));
    }
    filenames.push_back(entry->d_name);
  }
  closedir(dir);
  return filenames;
}

TEST(NodeIoDumpRewriterTest, OnGraph) {
  // Construct a graph.
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Scope scope = Scope::NewRootScope().WithDevice("/device:CPU:0");
  auto input_a = ops::Placeholder(scope.WithOpName("input_a"), DT_INT32);
  auto input_b = ops::Placeholder(scope.WithOpName("input_b"), DT_INT32);
  auto add = ops::Add(scope.WithOpName("add"), input_a, input_b);
  auto output = ops::Identity(scope.WithOpName("output"), add);
  TF_ASSERT_OK(scope.ToGraph(graph.get()));
  // Insert dump ops.
  const std::string dump_dir = ::testing::UniqueTestDirectory();
  TF_ASSERT_OK(InsertDumpOps(*graph, {"add"}, dump_dir));
  // Check the inserted dump ops.
  auto* node = FindNode(graph.get(), "add");
  EXPECT_EQ(node->num_inputs(), 2);
  EXPECT_EQ(GetInputNode(node, 0)->name(), "input_a/0/debug_identity");
  EXPECT_EQ(GetInputNode(node, 1)->name(), "input_b/0/debug_identity");
  EXPECT_EQ(node->num_outputs(), 1);
  EXPECT_EQ(GetOutputNode(node, 0)->name(), "add/0/debug_identity");
}

TEST(NodeIoDumpRewriterTest, OnSavedModelV1) {
  // Read meta_graph_def.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");
  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));
  // Insert dump ops.
  const std::string dump_dir = ::testing::UniqueTestDirectory();
  TF_ASSERT_OK(InsertDumpOps(meta_graph_def, {"Add"}, dump_dir));
  // Load and run saved model.
  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  SavedModel::Options options(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = false;
  TF_ASSERT_OK_AND_ASSIGN(
      auto saved_model,
      SavedModelImpl::LoadSavedModel(options, meta_graph_def, saved_model_dir));
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run({}, "another_toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1]),
              ::testing::ElementsAreArray({12}));
  // Check the dump files.
  ASSERT_OK_AND_ASSIGN(auto filenames, GetFilenames(dump_dir));
  ASSERT_EQ(filenames.size(), 3);
  EXPECT_TRUE(absl::StartsWith(filenames[0], "Add:out:0_"));
  EXPECT_TRUE(absl::StartsWith(filenames[1], "Add:in:0_"));
  EXPECT_TRUE(absl::StartsWith(filenames[2], "Add:in:1_"));
}

TEST(NodeIoDumpRewriterTest, OnSavedModelV2) {
  // Read meta_graph_def.
  std::string saved_model_dir = GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v2");
  MetaGraphDef meta_graph_def;
  TF_ASSERT_OK(ReadMetaGraphDefFromSavedModel(saved_model_dir, {"serve"},
                                              &meta_graph_def));
  // Insert dump ops.
  const std::string dump_dir = ::testing::UniqueTestDirectory();
  TF_ASSERT_OK(InsertDumpOps(meta_graph_def, {"result"}, dump_dir));
  // Load and run saved model.
  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  SavedModel::Options options(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = false;
  TF_ASSERT_OK_AND_ASSIGN(
      auto saved_model,
      SavedModelImpl::LoadSavedModel(options, meta_graph_def, saved_model_dir));
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
  // Check the dump files.
  ASSERT_OK_AND_ASSIGN(auto filenames, GetFilenames(dump_dir));
  ASSERT_EQ(filenames.size(), 3);
  EXPECT_TRUE(absl::StartsWith(filenames[0], "result:out:0_"));
  EXPECT_TRUE(absl::StartsWith(filenames[1], "result:in:1_"));
  EXPECT_TRUE(absl::StartsWith(filenames[2], "result:in:0_"));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
