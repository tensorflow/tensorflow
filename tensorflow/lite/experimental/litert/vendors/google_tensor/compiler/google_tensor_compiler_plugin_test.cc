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

#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_compiler_plugin.h"

namespace litert {
namespace {

TEST(TestGoogleTensorPlugin, GetConfigInfo) {
  ASSERT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "GoogleTensor");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 1);

  const char* soc_model_name;
  LITERT_ASSERT_OK(LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0,
                                                            &soc_model_name));
  ASSERT_STREQ(soc_model_name, "P25");
}

TEST(TestCallGoogleTensorPlugin, PartitionSimpleMultiAdd) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("simple_multi_op.tflite");

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), model.Subgraph(0)->Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Vec();

  ASSERT_EQ(selected_ops.size(), 2);
  ASSERT_EQ(selected_ops[0]->OpCode(), kLiteRtOpCodeTflMul);
  ASSERT_EQ(selected_ops[1]->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestCallGoogleTensorPlugin, CompileMulSubgraph) {
  GTEST_SKIP() << "Skipping this test until the compiler wrapper is updated";
  /* DISABLES CODE */
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("mul_simple.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "P25", model.Get(), &compiled));

  LiteRtDestroyCompiledResult(compiled);
}  // Todo(abhirs): activate this test once the compiler wrapper is updated

}  // namespace
}  // namespace litert
