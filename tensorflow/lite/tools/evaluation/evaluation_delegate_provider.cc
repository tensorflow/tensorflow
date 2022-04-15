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

#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"

#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {
namespace {
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";
constexpr char kHexagonDelegate[] = "hexagon";
constexpr char kXnnpackDelegate[] = "xnnpack";
}  // namespace

TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val) {
  if (val == kNnapiDelegate) return TfliteInferenceParams::NNAPI;
  if (val == kGpuDelegate) return TfliteInferenceParams::GPU;
  if (val == kHexagonDelegate) return TfliteInferenceParams::HEXAGON;
  if (val == kXnnpackDelegate) return TfliteInferenceParams::XNNPACK;
  return TfliteInferenceParams::NONE;
}

TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg) {
  const auto type = params.delegate();
  switch (type) {
    case TfliteInferenceParams::NNAPI: {
      auto p = CreateNNAPIDelegate();
      if (!p && error_msg) *error_msg = "NNAPI not supported";
      return p;
    }
    case TfliteInferenceParams::GPU: {
      auto p = CreateGPUDelegate();
      if (!p && error_msg) *error_msg = "GPU delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::HEXAGON: {
      auto p = CreateHexagonDelegate(/*library_directory_path=*/"",
                                     /*profiling=*/false);
      if (!p && error_msg) {
        *error_msg =
            "Hexagon delegate is not supported on the platform or required "
            "libraries are missing.";
      }
      return p;
    }
    case TfliteInferenceParams::XNNPACK: {
      auto p = CreateXNNPACKDelegate(params.num_threads());
      if (!p && error_msg) *error_msg = "XNNPACK delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::NONE:
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
    default:
      if (error_msg) {
        *error_msg = "Creation of delegate type: " +
                     TfliteInferenceParams::Delegate_Name(type) +
                     " not supported yet.";
      }
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
}

DelegateProviders::DelegateProviders()
    : delegate_list_util_(&params_),
      delegates_map_([=]() -> std::unordered_map<std::string, int> {
        std::unordered_map<std::string, int> delegates_map;
        const auto& providers = delegate_list_util_.providers();
        for (int i = 0; i < providers.size(); ++i) {
          delegates_map[providers[i]->GetName()] = i;
        }
        return delegates_map;
      }()) {
  delegate_list_util_.AddAllDelegateParams();
}

std::vector<Flag> DelegateProviders::GetFlags() {
  std::vector<Flag> flags;
  delegate_list_util_.AppendCmdlineFlags(flags);
  return flags;
}

bool DelegateProviders::InitFromCmdlineArgs(int* argc, const char** argv) {
  std::vector<Flag> flags = GetFlags();
  bool parse_result = Flags::Parse(argc, argv, flags);
  if (!parse_result || params_.Get<bool>("help")) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
    // Returning false intentionally when "--help=true" is specified so that
    // the caller could check the return value to decide stopping the execution.
    parse_result = false;
  }
  return parse_result;
}

TfLiteDelegatePtr DelegateProviders::CreateDelegate(
    const std::string& name) const {
  const auto it = delegates_map_.find(name);
  if (it == delegates_map_.end()) {
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
  const auto& providers = delegate_list_util_.providers();
  return providers[it->second]->CreateTfLiteDelegate(params_);
}

tools::ToolParams DelegateProviders::GetAllParams(
    const TfliteInferenceParams& params) const {
  tools::ToolParams tool_params;
  tool_params.Merge(params_, /*overwrite*/ false);

  if (params.has_num_threads()) {
    tool_params.Set<int32_t>("num_threads", params.num_threads());
  }

  const auto type = params.delegate();
  switch (type) {
    case TfliteInferenceParams::NNAPI:
      if (tool_params.HasParam("use_nnapi")) {
        tool_params.Set<bool>("use_nnapi", true);
      }
      break;
    case TfliteInferenceParams::GPU:
      if (tool_params.HasParam("use_gpu")) {
        tool_params.Set<bool>("use_gpu", true);
      }
      break;
    case TfliteInferenceParams::HEXAGON:
      if (tool_params.HasParam("use_hexagon")) {
        tool_params.Set<bool>("use_hexagon", true);
      }
      break;
    case TfliteInferenceParams::XNNPACK:
      if (tool_params.HasParam("use_xnnpack")) {
        tool_params.Set<bool>("use_xnnpack", true);
      }
      break;
    default:
      break;
  }
  return tool_params;
}

}  // namespace evaluation
}  // namespace tflite
