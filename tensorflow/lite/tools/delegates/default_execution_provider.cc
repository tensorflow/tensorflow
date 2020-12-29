/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

// This class actually doesn't provide any TFLite delegate instances, it simply
// provides common params and flags that are common to all actual delegate
// providers.
class DefaultExecutionProvider : public DelegateProvider {
 public:
  DefaultExecutionProvider() {
    default_params_.AddParam("num_threads", ToolParam::Create<int32_t>(1));
    default_params_.AddParam("max_delegated_partitions",
                             ToolParam::Create<int32_t>(0));
    default_params_.AddParam("min_nodes_per_partition",
                             ToolParam::Create<int32_t>(0));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;
  void LogParams(const ToolParams& params, bool verbose) const final;
  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::string GetName() const final { return "Default-NoDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DefaultExecutionProvider);

std::vector<Flag> DefaultExecutionProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<int32_t>("num_threads", params,
                          "number of threads used for inference on CPU."),
      CreateFlag<int32_t>("max_delegated_partitions", params,
                          "Max number of partitions to be delegated."),
      CreateFlag<int32_t>(
          "min_nodes_per_partition", params,
          "The minimal number of TFLite graph nodes of a partition that has to "
          "be reached for it to be delegated.A negative value or 0 means to "
          "use the default choice of each delegate.")};
  return flags;
}

void DefaultExecutionProvider::LogParams(const ToolParams& params,
                                         bool verbose) const {
  LOG_TOOL_PARAM(params, int32_t, "num_threads",
                 "#threads used for CPU inference", verbose);
  LOG_TOOL_PARAM(params, int32_t, "max_delegated_partitions",
                 "Max number of delegated partitions", verbose);
  LOG_TOOL_PARAM(params, int32_t, "min_nodes_per_partition",
                 "Min nodes per partition", verbose);
}

TfLiteDelegatePtr DefaultExecutionProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
}

}  // namespace tools
}  // namespace tflite
