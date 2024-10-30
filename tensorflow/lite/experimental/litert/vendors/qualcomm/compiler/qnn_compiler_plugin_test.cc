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
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/core/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

namespace {
static constexpr absl::string_view kOpTpl = "simple_%s_op.tflite";
// clang-format off
const auto kSupportedOps =
    testing::Values(
                    "add",
                    "div",
                    "mul",
                    "rsqrt",
                    "slice",
                    "sub",
                    "tanh",
                    "reshape",
                    "batch_matmul"
                    );
// clang-format on

UniqueLiteRtCompilerPlugin GetQnnPlugin() {
  LiteRtCompilerPlugin qnn_plugin;
  LITERT_CHECK_STATUS_OK(LiteRtPluginInit(&qnn_plugin));
  ABSL_CHECK_NE(qnn_plugin, nullptr);
  return UniqueLiteRtCompilerPlugin(qnn_plugin);
}

TEST(TestQnnPlugin, GetConfigInfo) {
  EXPECT_STREQ(LiteRtPluginSocManufacturer(), "Qualcomm");

  auto plugin = GetQnnPlugin();

  ASSERT_GE(LiteRtPluginNumSupportedSocModels(plugin.get()), 1);

  const char* config_id;
  LITERT_CHECK_STATUS_OK(
      LiteRtPluginGetSupportedSocModel(plugin.get(), 0, &config_id));
  EXPECT_STREQ(config_id, "V68");
}

TEST(TestQnnPlugin, PartitionMulOps) {
  auto plugin = GetQnnPlugin();
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");

  LiteRtOpListT selected_op_list;
  ASSERT_STATUS_OK(
      LiteRtPluginPartitionModel(plugin.get(), model.get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Vec();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0]->op_code, kLiteRtOpCodeTflMul);
}

TEST(TestQnnPlugin, CompileMulSubgraph) {
  auto plugin = GetQnnPlugin();
  auto model = litert::testing::LoadTestFileModel("one_mul.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));

  LiteRtCompiledResult compiled;
  ASSERT_STATUS_OK(
      LiteRtPluginCompile(plugin.get(), "V75", &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_STATUS_OK(
      LiteRtCompiledResultGetByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  ASSERT_STATUS_OK(
      LiteRtCompiledResultGetCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtCompiledResultDestroy(compiled);
}

class QnnPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPluginOpCompatibilityTest, SupportedOpsTest) {
  auto plugin = GetQnnPlugin();
  auto model =
      litert::testing::LoadTestFileModel(absl::StrFormat(kOpTpl, GetParam()));

  ASSERT_RESULT_OK_ASSIGN(auto subgraph,
                          ::graph_tools::GetSubgraph(model.get()));

  LiteRtCompiledResult compiled;
  ASSERT_STATUS_OK(
      LiteRtPluginCompile(plugin.get(), "V75", &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  ASSERT_STATUS_OK(
      LiteRtCompiledResultGetByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  ASSERT_STATUS_OK(
      LiteRtCompiledResultGetCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtCompiledResultDestroy(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpCompatibilityTest,
                         kSupportedOps);

}  // namespace
