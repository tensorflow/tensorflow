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
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#if !defined(_WIN32)
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/delegate_loader.h"
#include "tensorflow/lite/delegates/utils/experimental/stable_delegate/tflite_settings_json_parser.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/delegate_plugin.h"
#include "tensorflow/lite/experimental/acceleration/configuration/c/stable_delegate.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#endif  // !defined(_WIN32)

namespace tflite {
namespace tools {

// The stable delegate provider is disabled on Windows as the delegate shared
// library loader doesn't support Windows platform.
#if !defined(_WIN32)
namespace {

// Parses the JSON settings, loads the appropriate stable delegate plugin,
// and uses the stable delegate plugin to create a stable delegate.
// This assumes that the settings in the given json_settings_file_path
// will not to change during the lifetime of the program.
TfLiteDelegatePtr CreateStableDelegate(
    const std::string& json_settings_file_path);

// Class that encapulates the stable delegate cache management.
class StableDelegatePluginLoader {
 public:
  // Returns a singleton instance of this class.
  static StableDelegatePluginLoader& GetInstance() {
    static StableDelegatePluginLoader* const instance =
        new StableDelegatePluginLoader;
    return *instance;
  }

  // As per ::tflite::tools::CreateStableDelegate, above.
  TfLiteDelegatePtr CreateStableDelegate(
      const std::string& json_settings_file_path);

 private:
  struct CacheEntry {
    const TfLiteStableDelegate* stable_delegate = nullptr;
    delegates::utils::TfLiteSettingsJsonParser parser;  // Owns parsed_settings.
    const TFLiteSettings* parsed_settings = nullptr;
  };

  StableDelegatePluginLoader() = default;
  const CacheEntry* LoadStableDelegatePlugin(
      const std::string& json_settings_file_path);

  std::map<std::string /*settings_file_path*/, CacheEntry> cache_;
};

const StableDelegatePluginLoader::CacheEntry*
StableDelegatePluginLoader::LoadStableDelegatePlugin(
    const std::string& json_settings_file_path) {
  auto it = cache_.find(json_settings_file_path);
  if (it != cache_.end()) {
    return &it->second;
  }
  CacheEntry result;
  const TFLiteSettings* tflite_settings =
      result.parser.Parse(json_settings_file_path);
  result.parsed_settings = tflite_settings;
  if (!tflite_settings || !tflite_settings->stable_delegate_loader_settings() ||
      !tflite_settings->stable_delegate_loader_settings()->delegate_path()) {
    TFLITE_LOG(ERROR) << "Invalid TFLiteSettings for the stable delegate.";
    result.stable_delegate = nullptr;
  } else {
    std::string delegate_path =
        tflite_settings->stable_delegate_loader_settings()
            ->delegate_path()
            ->str();
    result.stable_delegate =
        delegates::utils::LoadDelegateFromSharedLibrary(delegate_path);
    if (!result.stable_delegate || !result.stable_delegate->delegate_plugin) {
      TFLITE_LOG(ERROR) << "Failed to load stable ABI delegate from stable ABI "
                           "delegate binary ("
                        << delegate_path << ").";
    }
  }
  auto it2 = cache_.emplace(json_settings_file_path, std::move(result)).first;
  return &it2->second;
}

TfLiteDelegatePtr CreateStableDelegate(
    const std::string& json_settings_file_path) {
  return StableDelegatePluginLoader::GetInstance().CreateStableDelegate(
      json_settings_file_path);
}

TfLiteDelegatePtr StableDelegatePluginLoader::CreateStableDelegate(
    const std::string& json_settings_file_path) {
  if (json_settings_file_path.empty()) {
    return CreateNullDelegate();
  }
  const CacheEntry* entry =
      StableDelegatePluginLoader::GetInstance().LoadStableDelegatePlugin(
          json_settings_file_path);
  if (!entry || !entry->stable_delegate ||
      !entry->stable_delegate->delegate_plugin) {
    return CreateNullDelegate();
  }
  const TfLiteOpaqueDelegatePlugin* delegate_plugin =
      entry->stable_delegate->delegate_plugin;
  return TfLiteDelegatePtr(delegate_plugin->create(entry->parsed_settings),
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
