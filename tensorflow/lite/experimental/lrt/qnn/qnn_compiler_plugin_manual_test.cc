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
#include <iostream>
#include <string>

#include "absl/log/absl_check.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/test_data/test_data_util.h"

typedef void (*TestFunc)();

namespace {

UniqueLrtCompilerPlugin GetQnnPlugin() {
  LrtCompilerPlugin qnn_plugin;
  LRT_CHECK_STATUS_OK(LrtPluginInit(&qnn_plugin));
  ABSL_CHECK_NE(qnn_plugin, nullptr);
  return UniqueLrtCompilerPlugin(qnn_plugin);
}

void TestQnnPlugin_GetConfigInfo() {
  ABSL_CHECK_STREQ(LrtPluginSocManufacturer(), "QNN");

  auto plugin = GetQnnPlugin();

  ABSL_CHECK_EQ(1, LrtPluginNumSupportedSocModels(plugin.get()));

  const char* config_id;
  LRT_CHECK_STATUS_OK(
      LrtPluginGetSupportedSocModelId(plugin.get(), 0, &config_id));
  ABSL_CHECK_STREQ(config_id, "HTP_Reference");
}

void TestQnnPluginPartition_PartitionMulOps() {
  auto plugin = GetQnnPlugin();
  auto model = LoadTestFileModel("one_mul.tflite");

  LrtOpListT selected_ops;
  LRT_CHECK_STATUS_OK(
      LrtPluginPartitionModel(plugin.get(), model.get(), &selected_ops));

  ABSL_CHECK_EQ(selected_ops.ops.size(), 1);
}

void TestQnnPluginCompile_CompileMulSubgraph() {
  auto plugin = GetQnnPlugin();
  auto model = LoadTestFileModel("one_mul.tflite");

  auto result = ::graph_tools::GetSubgraph(model.get());
  ABSL_CHECK(result.HasValue());
  auto subgraph = result.Value();

  LrtCompiledResult compiled;
  LRT_CHECK_STATUS_OK(LrtPluginCompile(plugin.get(), &subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LRT_CHECK_STATUS_OK(
      LrtCompiledResultGetByteCode(compiled, &byte_code, &byte_code_size));

  std::string byte_code_string(reinterpret_cast<const char*>(byte_code),
                               byte_code_size);
  ABSL_CHECK(!byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  LRT_CHECK_STATUS_OK(
      LrtCompiledResultGetCallInfo(compiled, 0, &op_data, &op_data_size));

  std::string op_data_string(reinterpret_cast<const char*>(op_data),
                             op_data_size);
  ABSL_CHECK_EQ("Unimplemented_QNN_Graph", op_data_string);

  LrtCompiledResultDestroy(compiled);
}

}  // namespace

void ExecuteSuite() {
  static const TestFunc suite[] = {TestQnnPlugin_GetConfigInfo,
                                   TestQnnPluginPartition_PartitionMulOps,
                                   TestQnnPluginCompile_CompileMulSubgraph};

  std::cerr << "RUNNING SUITE\n";
  for (const auto& t : suite) {
    t();
  }
  std::cerr << "SUCCESS\n";
}

int main(int argc, char* argv[]) {
  ExecuteSuite();
  return 0;
}
