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
#include <string>

#include <gtest/gtest.h>
#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

UniqueLrtCompilerPlugin GetQnnPlugin() {
  LrtCompilerPlugin qnn_plugin;
  LRT_CHECK_STATUS_OK(LrtPluginInit(&qnn_plugin));
  ABSL_CHECK_NE(qnn_plugin, nullptr);
  return UniqueLrtCompilerPlugin(qnn_plugin);
}

TEST(TestQnnPlugin, GetConfigInfo) {
  EXPECT_STREQ(LrtPluginSocManufacturer(), "Qualcomm");

  auto plugin = GetQnnPlugin();

  ASSERT_GE(LrtPluginNumSupportedSocModels(plugin.get()), 1);

  const char* config_id;
  LRT_CHECK_STATUS_OK(
      LrtPluginGetSupportedSocModel(plugin.get(), 0, &config_id));
  EXPECT_STREQ(config_id, "V68");
}

TEST(TestQnnPlugin, PartitionMulOps) {
  auto plugin = GetQnnPlugin();
  auto model = lrt::testing::LoadTestFileModel("one_mul.tflite");

  LrtOpListT selected_ops;
  ASSERT_STATUS_OK(
      LrtPluginPartitionModel(plugin.get(), model.get(), &selected_ops));

  EXPECT_EQ(selected_ops.ops.size(), 1);
}

TEST(TestQnnPlugin, CompileMulSubgraph) {
  auto plugin = GetQnnPlugin();
  auto model = lrt::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));

  LrtCompiledResult compiled;
  ASSERT_STATUS_OK(
      LrtPluginCompile(plugin.get(), "V75", &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_STATUS_OK(
      LrtCompiledResultGetByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  ASSERT_STATUS_OK(
      LrtCompiledResultGetCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LrtCompiledResultDestroy(compiled);
}

}  // namespace
