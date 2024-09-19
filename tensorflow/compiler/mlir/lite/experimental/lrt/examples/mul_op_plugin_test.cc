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
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_op_code.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/test_data/test_data_util.h"

namespace {

UniqueLrtCompilerPlugin GetDummyPlugin() {
  LrtCompilerPlugin dummy_plugin;
  LRT_CHECK_STATUS_OK(LrtPluginInit(&dummy_plugin));
  CHECK_NE(dummy_plugin, nullptr);
  return UniqueLrtCompilerPlugin(dummy_plugin);
}

TEST(TestDummyPlugin, GetConfigInfo) {
  ASSERT_STREQ(LrtPluginSocManufacturer(), "ExampleSocManufacturer");

  auto plugin = GetDummyPlugin();

  ASSERT_EQ(1, LrtPluginNumSupportedSocModels(plugin.get()));

  const char* config_id;
  ASSERT_STATUS_OK(
      LrtPluginGetSupportedSocModelId(plugin.get(), 0, &config_id));
  ASSERT_STREQ(config_id, "DummyMulOp");
}

TEST(TestCallDummyPlugin, PartitionSimpleMultiAdd) {
  auto plugin = GetDummyPlugin();
  auto model = LoadTestFileModel("simple_multi_op.tflite");

  LrtOpListT selected_ops;
  ASSERT_STATUS_OK(
      LrtPluginPartitionModel(plugin.get(), model.get(), &selected_ops));

  ASSERT_EQ(selected_ops.ops.size(), 2);
  ASSERT_EQ(selected_ops.ops[0]->op_code, kLrtOpCodeTflMul);
  ASSERT_EQ(selected_ops.ops[1]->op_code, kLrtOpCodeTflMul);
}

TEST(TestCallDummyPlugin, CompileMulSubgraph) {
  auto plugin = GetDummyPlugin();
  auto model = LoadTestFileModel("mul_simple.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));

  LrtCompiledResult compiled;
  ASSERT_STATUS_OK(LrtPluginCompile(plugin.get(), &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_STATUS_OK(
      LrtCompiledResultGetByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ASSERT_EQ(byte_code_string, "Partition_0_with_2_muls:");

  const void* op_data;
  size_t op_data_size;

  ASSERT_STATUS_OK(
      LrtCompiledResultGetCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ASSERT_EQ(op_data_string, "Partition_0");

  LrtCompiledResultDestroy(compiled);
}

}  // namespace
