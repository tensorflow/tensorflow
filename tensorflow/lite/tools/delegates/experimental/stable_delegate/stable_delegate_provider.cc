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
#include <vector>

#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#if !defined(_WIN32)
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/delegate_plugin.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#endif  // !defined(_WIN32)

namespace tflite {
namespace tools {

class StableAbiDelegateProvider : public DelegateProvider {
 public:
  StableAbiDelegateProvider() {
    default_params_.AddParam("stable_delegate_path",
                             ToolParam::Create<std::string>(""));
    default_params_.AddParam(
        "stable_delegate_plugin_symbol",
        ToolParam::Create<std::string>(
            delegates::utils::kTfLiteStableDelegateSymbol));
  }

  std::vector<Flag> CreateFlags(ToolParams* params) const final;

  void LogParams(const ToolParams& params, bool verbose) const final;

  TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;
  std::pair<TfLiteDelegatePtr, int> CreateRankedTfLiteDelegate(
      const ToolParams& params) const final;

  std::string GetName() const final { return "STABLE_DELEGATE"; }
};
REGISTER_DELEGATE_PROVIDER(StableAbiDelegateProvider);

std::vector<Flag> StableAbiDelegateProvider::CreateFlags(
    ToolParams* params) const {
  std::vector<Flag> flags = {
      CreateFlag<std::string>("stable_delegate_path", params,
                              "The library path for the delegate."),
      CreateFlag<std::string>(
          "stable_delegate_plugin_symbol", params,
          "The name of the delegate plugin symbol in the shared library. "
          "(default='TFL_TheStableDelegate')")};
  return flags;
}

void StableAbiDelegateProvider::LogParams(const ToolParams& params,
                                          bool verbose) const {
  if (params.Get<std::string>("stable_delegate_path").empty()) return;

  LOG_TOOL_PARAM(params, std::string, "stable_delegate_path", "Delegate path",
                 verbose);
  LOG_TOOL_PARAM(params, std::string, "stable_delegate_plugin_symbol",
                 "Delegate plugin symbol", verbose);
}

TfLiteDelegatePtr StableAbiDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
  TfLiteDelegatePtr null_delegate = CreateNullDelegate();
#if !defined(_WIN32)
  std::string lib_path = params.Get<std::string>("stable_delegate_path");
  std::string stable_delegate_plugin_symbol =
      params.Get<std::string>("stable_delegate_plugin_symbol");
  if (lib_path.empty()) {
    // Stable ABI delegate is not used if "stable_delegate_path" is not
    // provided.
    return null_delegate;
  }
  if (stable_delegate_plugin_symbol.empty()) {
    TFLITE_LOG(ERROR) << "Delegate plugin symbol ("
                      << stable_delegate_plugin_symbol
                      << ") must not be empty.";
    return null_delegate;
  }
  auto stable_delegate_pointer =
      delegates::utils::LoadDelegateFromSharedLibrary(
          lib_path, stable_delegate_plugin_symbol);
  if (!stable_delegate_pointer) {
    TFLITE_LOG(ERROR)
        << "Failed to load stable ABI delegate pointer from stable ABI "
           "delegate binary ("
        << lib_path << ") with delegate plugin symbol ("
        << stable_delegate_plugin_symbol << ").";
    return null_delegate;
  }

  // TODO(b/250886376): Allow passing TFLiteSettings via JSON formatted string
  // arguments.
  flatbuffers::FlatBufferBuilder flatbuffer_builder;
  tflite::TFLiteSettingsBuilder tflite_settings_builder(flatbuffer_builder);
  flatbuffers::Offset<tflite::TFLiteSettings> tflite_settings =
      tflite_settings_builder.Finish();
  flatbuffer_builder.Finish(tflite_settings);
  auto delegate_plugin = stable_delegate_pointer->delegate_plugin;
  TfLiteOpaqueDelegate* delegate =
      delegate_plugin->create(flatbuffer_builder.GetBufferPointer());
  void (*delegate_deleter)(TfLiteOpaqueDelegate*) = delegate_plugin->destroy;
  return TfLiteDelegatePtr(delegate, delegate_deleter);
#else   // !defined(_WIN32)
  return null_delegate;
#endif  // !defined(_WIN32)
}

std::pair<TfLiteDelegatePtr, int>
StableAbiDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(
      std::move(ptr), params.GetPosition<std::string>("stable_delegate_path"));
}

}  // namespace tools
}  // namespace tflite
