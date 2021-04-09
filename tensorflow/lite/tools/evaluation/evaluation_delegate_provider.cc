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
    : delegates_list_(tools::GetRegisteredDelegateProviders()),
      delegates_map_([=]() -> std::unordered_map<std::string, int> {
        std::unordered_map<std::string, int> delegates_map;
        for (int i = 0; i < delegates_list_.size(); ++i) {
          delegates_map[delegates_list_[i]->GetName()] = i;
        }
        return delegates_map;
      }()) {
  for (const auto& one : delegates_list_) {
    params_.Merge(one->DefaultParams());
  }
}

std::vector<Flag> DelegateProviders::GetFlags() {
  std::vector<Flag> flags;
  for (const auto& one : delegates_list_) {
    auto one_flags = one->CreateFlags(&params_);
    flags.insert(flags.end(), one_flags.begin(), one_flags.end());
  }
  return flags;
}

bool DelegateProviders::InitFromCmdlineArgs(int* argc, const char** argv) {
  std::vector<Flag> flags = GetFlags();
  const bool parse_result = Flags::Parse(argc, argv, flags);
  if (!parse_result) {
    std::string usage = Flags::Usage(argv[0], flags);
    TFLITE_LOG(ERROR) << usage;
  }
  return parse_result;
}

TfLiteDelegatePtr DelegateProviders::CreateDelegate(
    const std::string& name) const {
  const auto it = delegates_map_.find(name);
  if (it == delegates_map_.end()) {
    return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
  return delegates_list_[it->second]->CreateTfLiteDelegate(params_);
}

std::vector<TfLiteDelegatePtr> DelegateProviders::CreateAllDelegates(
    const tools::ToolParams& params) const {
  std::vector<TfLiteDelegatePtr> delegates;
  for (const auto& one : delegates_list_) {
    auto ptr = one->CreateTfLiteDelegate(params);
    // It's possible that a delegate of certain type won't be created as
    // user-specified benchmark params tells not to.
    if (ptr == nullptr) continue;
    delegates.emplace_back(std::move(ptr));
    TFLITE_LOG(INFO) << one->GetName() << " delegate is created.";
  }
  return delegates;
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
