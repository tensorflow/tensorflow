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
#include <utility>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#if defined(__APPLE__)
#include "TargetConditionals.h"
#if (TARGET_OS_IPHONE && !TARGET_IPHONE_SIMULATOR) || \
    (TARGET_OS_OSX && TARGET_CPU_ARM64)
// Only enable coreml delegate when using a real iPhone device or Apple Silicon.
#define REAL_IPHONE_DEVICE
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#endif
#endif

namespace tflite {
namespace tools {

class CoreMlDelegateProvider : public DelegateProvider {
 public:
  CoreMlDelegateProvider() {
#if defined(REAL_IPHONE_DEVICE)
    default_params_.AddParam("use_coreml", ToolParam::Create<bool>(false));
    default_params_.AddParam("coreml_version", ToolParam::Create<int>(0));
#endif
  }
  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "COREML"; }
};
REGISTER_DELEGATE_PROVIDER(CoreMlDelegateProvider);

std::vector<Flag> CoreMlDelegateProvider::CreateFlags(
    ToolParams* params) const {
#if defined(REAL_IPHONE_DEVICE)
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_coreml", params, "use Core ML"),
      CreateFlag<int>("coreml_version", params,
                      "Target Core ML version for model conversion. "
                      "The default value is 0 and it means using the newest "
                      "version that's available on the device."),
  };
  return flags;
#else
  return {};
#endif
}

void CoreMlDelegateProvider::LogParams(const ToolParams& params,
                                       bool verbose) const {
#if defined(REAL_IPHONE_DEVICE)
  LOG_TOOL_PARAM(params, bool, "use_coreml", "Use CoreML", verbose);
  LOG_TOOL_PARAM(params, int, "coreml_version", "CoreML version", verbose);
#endif
}

TfLiteDelegatePtr CoreMlDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr delegate = CreateNullDelegate();

#if defined(REAL_IPHONE_DEVICE)
  if (params.Get<bool>("use_coreml")) {
    TfLiteCoreMlDelegateOptions coreml_opts = {
        .enabled_devices = TfLiteCoreMlDelegateAllDevices};
    coreml_opts.coreml_version = params.Get<int>("coreml_version");
    coreml_opts.max_delegated_partitions =
        params.Get<int>("max_delegated_partitions");
    coreml_opts.min_nodes_per_partition =
        params.Get<int>("min_nodes_per_partition");
#ifdef TFLITE_DEBUG_DELEGATE
    coreml_opts.first_delegate_node_index =
        params.Get<int>("first_delegate_node_index");
    coreml_opts.last_delegate_node_index =
        params.Get<int>("last_delegate_node_index");
#endif  // TFLITE_DEBUG_DELEGATE
    delegate = TfLiteDelegatePtr(TfLiteCoreMlDelegateCreate(&coreml_opts),
                                 &TfLiteCoreMlDelegateDelete);
    if (!delegate) {
      TFLITE_LOG(WARN)
          << "CoreML acceleration is unsupported on this platform.";
    }
  }
#endif

  return delegate;
}

std::pair<TfLiteDelegatePtr, int>
CoreMlDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  int rank = 0;
#if defined(REAL_IPHONE_DEVICE)
  rank = params.GetPosition<bool>("use_coreml");
#endif
  return std::make_pair(std::move(ptr), rank);
}

}  // namespace tools
}  // namespace tflite
