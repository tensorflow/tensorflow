/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"
#include "tensorflow/lite/tools/tool_params.h"

namespace tflite {
namespace tools {

class YnnpackDelegateProvider : public DelegateProvider {
 public:
  YnnpackDelegateProvider() {
    default_params_.AddParam("use_ynnpack", ToolParam::Create<bool>(false));
    default_params_.AddParam("ynnpack_static_shape",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("ynnpack_fast_math",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("ynnpack_consistent_arithmetic",
                             ToolParam::Create<bool>(false));
    default_params_.AddParam("ynnpack_no_excess_precision",
                             ToolParam::Create<bool>(false));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "YNNPACK"; }
};
REGISTER_DELEGATE_PROVIDER(YnnpackDelegateProvider);

std::vector<Flag> YnnpackDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<bool>("use_ynnpack", params,
                       "explicitly apply the YNNPACK delegate."),
      CreateFlag<bool>(
          "ynnpack_static_shape", params,
          "make input shapes static instead of dynamic. May improve invoke "
          "performance at the cost of much more expensive reshape."),
      CreateFlag<bool>("ynnpack_fast_math", params,
                       "enable YNN_FLAG_FAST_MATH."),
      CreateFlag<bool>("ynnpack_consistent_arithmetic", params,
                       "enable YNN_FLAG_CONSISTENT_ARITHMETIC."),
      CreateFlag<bool>("ynnpack_no_excess_precision", params,
                       "enable YNN_FLAG_NO_EXCESS_PRECISION."),
  };
  return flags;
}

void YnnpackDelegateProvider::LogParams(const ToolParams& params,
                                        bool verbose) const {
  LOG_TOOL_PARAM(params, bool, "use_ynnpack", "Use ynnpack", verbose);
  LOG_TOOL_PARAM(params, bool, "ynnpack_static_shape", "YNNPACK static shape",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "ynnpack_fast_math", "YNNPACK fast math",
                 verbose);
  LOG_TOOL_PARAM(params, bool, "ynnpack_consistent_arithmetic",
                 "YNNPACK consistent arithmetic", verbose);
  LOG_TOOL_PARAM(params, bool, "ynnpack_no_excess_precision",
                 "YNNPACK no excess precision", verbose);
}

TfLiteDelegatePtr YnnpackDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  if (params.Get<bool>("use_ynnpack")) {
    auto opts = TfLiteYNNPackDelegateOptionsDefault();
    opts.num_threads = params.Get<int32_t>("num_threads");
    // Note that we don't want to use the thread pool for num_threads == 1.
    if (opts.num_threads <= 1) opts.num_threads = 0;

    opts.static_shape = params.Get<bool>("ynnpack_static_shape");
    opts.fast_math = params.Get<bool>("ynnpack_fast_math");
    opts.consistent_arithmetic =
        params.Get<bool>("ynnpack_consistent_arithmetic");
    opts.no_excess_precision = params.Get<bool>("ynnpack_no_excess_precision");

    return TfLiteDelegatePtr(reinterpret_cast<TfLiteOpaqueDelegate*>(
                                 TfLiteYNNPackDelegateCreate(&opts)),
                             [](TfLiteOpaqueDelegate* delegate) {
                               TfLiteYNNPackDelegateDelete(
                                   reinterpret_cast<TfLiteDelegate*>(delegate));
                             });
  }
  return CreateNullDelegate();
}

std::pair<TfLiteDelegatePtr, int>
YnnpackDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr),
                        params.GetPosition<bool>("use_ynnpack"));
}

}  // namespace tools
}  // namespace tflite
