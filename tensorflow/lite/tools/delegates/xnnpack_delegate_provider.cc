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

namespace tflite {
namespace tools {

class XnnpackDelegateProvider : public DelegateProvider {
 public:
  XnnpackDelegateProvider() {
    default_params_.AddParam("use_xnnpack", ToolParam::Create<bool>(false));
    default_params_.AddParam("xnnpack_force_fp16",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("xnnpack_experimental_weight_cache_file_path",
                             ToolParam::Create<std::string>(""));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "XNNPACK"; }
};
REGISTER_DELEGATE_PROVIDER(XnnpackDelegateProvider);

std::vector<Flag> XnnpackDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_xnnpack", params,
                       "explicitly apply the XNNPACK delegate. Note the "
                       "XNNPACK delegate could "
                       "be implicitly applied by the TF Lite runtime "
                       "regardless the value of "
                       "this parameter. To disable this implicit application, "
                       "set the value to "
                       "false explicitly."),
      CreateFlag<bool>("xnnpack_force_fp16", params,
                       "enforce float16 inference."),
      CreateFlag<std::string>("xnnpack_experimental_weight_cache_file_path",
                              params, "enable file-backed weight caching."),
  };
  return flags;
}

void XnnpackDelegateProvider::LogParams(const ToolParams& params,
                                        bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_xnnpack", "Use xnnpack", verbose);
  LOG_TOOL_PARAM(params, bool, "xnnpack_force_fp16", "xnnpack_force_fp16",
                 verbose);
  LOG_TOOL_PARAM(params, std::string,
                 "xnnpack_experimental_weight_cache_file_path",
                 "xnnpack_experimental_weight_cache_file_path", verbose);
}

TfLiteDelegatePtr XnnpackDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_xnnpack")) {
    return evaluation::CreateXNNPACKDelegate(
        params.Get<int32_t>("num_threads"),
        params.Get<bool>("xnnpack_force_fp16"),
        params.Get<std::string>("xnnpack_experimental_weight_cache_file_path")
            .c_str());
  }
  return CreateNullDelegate();
}

std::pair<TfLiteDelegatePtr, int>
XnnpackDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_xnnpack"));
}

}  // namespace tools
}  // namespace tflite
