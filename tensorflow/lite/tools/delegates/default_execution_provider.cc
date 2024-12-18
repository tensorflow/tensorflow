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
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"

namespace tflite {
namespace tools {

// This class actually doesn't provide any TFLite delegate instances, it simply
// provides common params and flags that are common to all actual delegate
// providers.
class DefaultExecutionProvider : public DelegateProvider {
 public:
  DefaultExecutionProvider() {
    default_params_.AddParam("help", ToolParam::Create<bool>(false));
    default_params_.AddParam("num_threads", ToolParam::Create<int32_t>(-1));
    default_params_.AddParam("max_delegated_partitions",
                             ToolParam::Create<int32_t>(0));
    default_params_.AddParam("min_nodes_per_partition",
                             ToolParam::Create<int32_t>(0));
    default_params_.AddParam("delegate_serialize_dir",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("delegate_serialize_token",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam("first_delegate_node_index",
                             ToolParam::Create<int32_t>(0));
    default_params_.AddParam(
        "last_delegate_node_index",
        ToolParam::Create<int32_t>(std::numeric_limits<int32_t>::max()));
    default_params_.AddParam("gpu_invoke_loop_times",
                             ToolParam::Create<int32_t>(-1));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;
  void LogParams(const ToolParams& params, bool verbose) const final;
  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "Default-NoDelegate"; }
};
REGISTER_DELEGATE_PROVIDER(DefaultExecutionProvider);

std::vector<Flag> DefaultExecutionProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("help", params,
                       "Print out all supported flags if true."),
      CreateFlag<int32_t>("num_threads", params,
                          "number of threads used for inference on CPU."),
      CreateFlag<int32_t>("max_delegated_partitions", params,
                          "Max number of partitions to be delegated."),
      CreateFlag<int32_t>(
          "min_nodes_per_partition", params,
          "The minimal number of TFLite graph nodes of a partition that has to "
          "be reached for it to be delegated.A negative value or 0 means to "
          "use the default choice of each delegate."),
      CreateFlag<int32_t>(
          "first_delegate_node_index", params,
          "The index of the first node that could be delegated. Used only when "
          "TFLITE_DEBUG_DELEGATE is defined. Default is 0."),
      CreateFlag<int32_t>(
          "last_delegate_node_index", params,
          "The index of the last node that could be delegated. Used only when "
          "TFLITE_DEBUG_DELEGATE is defined. Default is INT_MAX."),
      CreateFlag<int32_t>(
          "gpu_invoke_loop_times", params,
          "Number of GPU delegate invoke loop iterations. Used only when "
          "TFLITE_GPU_ENABLE_INVOKE_LOOP is defined. Default is 1."),
      CreateFlag<std::string>(
          "delegate_serialize_dir", params,
          "Directory to be used by delegates for serializing any model data. "
          "This allows the delegate to save data into this directory to reduce "
          "init time after the first run. Currently supported by NNAPI "
          "delegate with specific backends on Android. Note that "
          "delegate_serialize_token is also required to enable this feature."),
      CreateFlag<std::string>(
          "delegate_serialize_token", params,
          "Model-specific token acting as a namespace for delegate "
          "serialization. Unique tokens ensure that the delegate doesn't read "
          "inapplicable/invalid data. Note that delegate_serialize_dir is also "
          "required to enable this feature."),
  };
  return flags;
}

void DefaultExecutionProvider::LogParams(const ToolParams& params,
                                         bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "help", "print out all supported flags",
                 verbose);
  LOG_TOOL_PARAM(params, int32_t, "num_threads",
                 "#threads used for CPU inference", verbose);
  LOG_TOOL_PARAM(params, int32_t, "max_delegated_partitions",
                 "Max number of delegated partitions", verbose);
  LOG_TOOL_PARAM(params, int32_t, "min_nodes_per_partition",
                 "Min nodes per partition", verbose);
  LOG_TOOL_PARAM(params, int32_t, "first_delegate_node_index",
                 "Index of the first node that could be delegated", verbose);
  LOG_TOOL_PARAM(params, int32_t, "last_delegate_node_index",
                 "Index of the last node that could be delegated", verbose);
  LOG_TOOL_PARAM(params, int32_t, "gpu_invoke_loop_times",
                 "Number of GPU delegate invoke loop iterations", verbose);
  LOG_TOOL_PARAM(params, std::string, "delegate_serialize_dir",
                 "Directory for delegate serialization", verbose);
  LOG_TOOL_PARAM(params, std::string, "delegate_serialize_token",
                 "Model-specific token/key for delegate serialization.",
                 verbose);
}

TfLiteDelegatePtr DefaultExecutionProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  return CreateNullDelegate();
}

std::pair<TfLiteDelegatePtr, int>
DefaultExecutionProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), 0);
}

}  // namespace tools
}  // namespace tflite
