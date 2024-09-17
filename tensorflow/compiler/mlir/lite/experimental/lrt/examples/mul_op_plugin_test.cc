// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/test_data/test_data_util.h"

namespace {

struct LrtCompilerPluginDeleter {
  void operator()(LrtCompilerPlugin plugin) {
    if (plugin != nullptr) {
      PluginDestroy(plugin);
    }
  }
};

using UniqueLrtCompilerPlugin =
    std::unique_ptr<LrtCompilerPluginT, LrtCompilerPluginDeleter>;

UniqueLrtCompilerPlugin GetDummyPlugin() {
  LrtCompilerPlugin dummy_plugin;
  LRT_CHECK_STATUS_OK(PluginInit(&dummy_plugin));
  CHECK_NE(dummy_plugin, nullptr);
  return UniqueLrtCompilerPlugin(dummy_plugin);
}

TEST(TestDummyPlugin, ConstructAndDestroy) {
  auto plugin = GetDummyPlugin();
  ASSERT_STREQ(PluginGetNamespace(plugin.get()), "mul_op_plugin");
}

TEST(TestCallDummyPlugin, PartitionSimpleMultiAdd) {
  auto plugin = GetDummyPlugin();
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  LrtOpListT selected_ops;
  ASSERT_STATUS_OK(
      PluginPartitionModel(plugin.get(), model.get(), &selected_ops));

  ASSERT_EQ(selected_ops.ops.size(), 2);
  ASSERT_EQ(selected_ops.ops[0]->op_code, kLrtOpCodeTflMul);
  ASSERT_EQ(selected_ops.ops[1]->op_code, kLrtOpCodeTflMul);
}

TEST(TestCallDummyPlugin, CompileMulSubgraph) {
  auto plugin = GetDummyPlugin();
  auto model = LoadTestFileModel("mul_simple.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));

  LrtCompiledPartition compiled;
  ASSERT_STATUS_OK(PluginCompilePartition(plugin.get(), subgraph, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_STATUS_OK(PluginCompiledPartitionGetByteCode(compiled, &byte_code,
                                                      &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);

  ASSERT_EQ(byte_code_string, "partition_with_2_muls");

  PluginCompiledPartitionDestroy(compiled);
}

}  // namespace
