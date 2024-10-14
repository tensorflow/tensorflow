// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/plugin_manager.h"

#include <glob.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin_api.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"

namespace lrt::internal {

const LrtPluginApi& LrtPluginManager::Api() { return plugin_api_; }

LrtCompilerPlugin LrtPluginManager::PluginHandle() { return plugin_handle_; }

LrtCompiledResult LrtPluginManager::CompiledResultHandle() {
  return compiled_result_handle_;
}

void LrtPluginManager::DumpLibInfo() const { ::lrt::DumpLibInfo(lib_handle_); }

void LrtPluginManager::DumpPluginInfo() const {
  LITE_RT_LOG(LRT_INFO, "CompilerPlugin: %s : %d",
              plugin_api_.soc_manufacturer(),
              plugin_api_.num_supported_models(plugin_handle_));
}

LrtStatus LrtPluginManager::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths,
    std::vector<LrtPluginManager>& loaded_plugins) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    LRT_RETURN_STATUS_IF_NOT_OK(
        FindLrtSharedLibs(search_path, plugin_lib_paths));
  }

  for (const auto& lib_path : plugin_lib_paths) {
    LITE_RT_LOG(LRT_INFO, "Loading plugin at: %s", lib_path.c_str());

    void* lib_handle;
    if (OpenLib(lib_path, &lib_handle) != kLrtStatusOk) {
      LITE_RT_LOG(LRT_WARNING, "Failed to load plugin at: %s",
                  lib_path.c_str());
      continue;
    }

    LrtPluginApi api;
    if (ResolvePluginApi(lib_handle, api) != kLrtStatusOk) {
      LITE_RT_LOG(LRT_WARNING, "Failed to resolve plugin api at: %s",
                  lib_path.c_str());
      continue;
    }

    LrtCompilerPlugin plugin_handle;
    if (api.init(&plugin_handle) != kLrtStatusOk) {
      LITE_RT_LOG(LRT_WARNING, "Failed to initialize plugin at: %s",
                  lib_path.c_str());
      if (CloseLib(lib_handle) != kLrtStatusOk) {
        LITE_RT_LOG(LRT_WARNING, "Failed to close loaded library at: %s",
                    lib_path.c_str());
      }
      continue;
    }

    auto& new_plugin_manager = loaded_plugins.emplace_back();
    new_plugin_manager.plugin_api_ = api;
    new_plugin_manager.lib_handle_ = lib_handle;
    new_plugin_manager.plugin_handle_ = plugin_handle;
  }

  return kLrtStatusOk;
}

LrtPluginManager::LrtPluginManager(LrtPluginManager&& other)
    : lib_handle_(other.lib_handle_),
      plugin_api_(std::move(other.plugin_api_)),
      plugin_handle_(other.plugin_handle_),
      compiled_result_handle_(other.compiled_result_handle_) {
  other.plugin_api_ = LrtPluginApi();
  other.lib_handle_ = nullptr;
  other.plugin_handle_ = nullptr;
  other.compiled_result_handle_ = nullptr;
}

LrtPluginManager& LrtPluginManager::operator=(LrtPluginManager&& other) {
  if (this != &other) {
    lib_handle_ = other.lib_handle_;
    other.lib_handle_ = nullptr;

    plugin_api_ = std::move(other.plugin_api_);
    other.plugin_api_ = LrtPluginApi();

    plugin_handle_ = other.plugin_handle_;
    other.plugin_handle_ = nullptr;
  }
  return *this;
}

LrtPluginManager::~LrtPluginManager() {
  if (compiled_result_handle_ != nullptr) {
    Api().compiled_result_destroy(CompiledResultHandle());
  }
  if (plugin_handle_ != nullptr) {
    Api().destroy(PluginHandle());
  }
  if (lib_handle_ != nullptr) {
    if (kLrtStatusOk != CloseLib(lib_handle_)) {
      LITE_RT_LOG(LRT_WARNING, "%s", "Failed to close shared library\n");
    }
  }
}

}  // namespace lrt::internal
