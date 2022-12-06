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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/core/shims/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#if !defined(_WIN32)
#include "tensorflow/lite/core/shims/c/experimental/acceleration/configuration/delegate_plugin.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#endif  // !defined(_WIN32)

namespace tflite {
namespace tools {

// The stable delegate provider is disabled on Windows as the delegate shared
// library loader doesn't support Windows platform.
#if !defined(_WIN32)
namespace {
TfLiteDelegatePtr CreateStableDelegate(const std::string& settings_file_path) {
  TfLiteDelegatePtr null_delegate = CreateNullDelegate();
  if (settings_file_path.empty()) {
    TFLITE_LOG(ERROR) << "Invalid delegate settings path.";
    return null_delegate;
  }
  delegates::utils::TfLiteSettingsJsonParser parser;
  const TFLiteSettings* tflite_settings = parser.Parse(settings_file_path);
  if (!tflite_settings || !tflite_settings->stable_delegate_loader_settings() ||
      !tflite_settings->stable_delegate_loader_settings()->delegate_path()) {
    TFLITE_LOG(ERROR) << "Invalid TFLiteSettings for the stable delegate.";
    return null_delegate;
  }
  std::string delegate_path = tflite_settings->stable_delegate_loader_settings()
                                  ->delegate_path()
                                  ->str();
  auto stable_delegate_pointer =
      delegates::utils::LoadDelegateFromSharedLibrary(delegate_path);
  if (!stable_delegate_pointer || !stable_delegate_pointer->delegate_plugin) {
    TFLITE_LOG(ERROR)
        << "Failed to load stable ABI delegate pointer from stable ABI "
           "delegate binary ("
        << delegate_path << ".";
    return null_delegate;
  }
  const TfLiteOpaqueDelegatePlugin* delegate_plugin =
      stable_delegate_pointer->delegate_plugin;
  return TfLiteDelegatePtr(delegate_plugin->create(tflite_settings),
                           delegate_plugin->destroy);
}
}  // namespace
#endif  // !defined(_WIN32)

class StableAbiDelegateProvider : public DelegateProvider {
 public:
  StableAbiDelegateProvider() {
    default_params_.AddParam("stable_delegate_settings_file",
                             ToolParam::Create<std::string>(""));
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
      CreateFlag<std::string>("stable_delegate_settings_file", params,
                              "The path to the delegate settings JSON file.")};
  return flags;
}

void StableAbiDelegateProvider::LogParams(const ToolParams& params,
                                          bool verbose) const {
  if (params.Get<std::string>("stable_delegate_settings_file").empty()) return;

  LOG_TOOL_PARAM(params, std::string, "stable_delegate_settings_file",
                 "Delegate settings file path", verbose);
}

TfLiteDelegatePtr StableAbiDelegateProvider::CreateTfLiteDelegate(
    const ToolParams& params) const {
#if !defined(_WIN32)
  std::string stable_delegate_settings_file =
      params.Get<std::string>("stable_delegate_settings_file");
  return CreateStableDelegate(stable_delegate_settings_file);
#else   // !defined(_WIN32)
  return CreateNullDelegate();
#endif  // !defined(_WIN32)
}

std::pair<TfLiteDelegatePtr, int>
StableAbiDelegateProvider::CreateRankedTfLiteDelegate(
    const ToolParams& params) const {
  auto ptr = CreateTfLiteDelegate(params);
  return std::make_pair(std::move(ptr), params.GetPosition<std::string>(
                                            "stable_delegate_settings_file"));
}

}  // namespace tools
}  // namespace tflite
