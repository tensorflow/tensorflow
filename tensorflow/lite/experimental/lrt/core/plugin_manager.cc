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
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin_api.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"

namespace lrt::internal {

const LrtPluginApi* LrtPluginManager::Api() {
  return IsLoaded() ? &plugin_api_ : nullptr;
}

bool LrtPluginManager::IsLoaded() const { return lib_handle_ != nullptr; }

void LrtPluginManager::DumpLibInfo() const {
  if (!IsLoaded()) {
    LITE_RT_LOG(LRT_INFO, "%s", "Lib is not loaded\n");
    return;
  }
  ::lrt::DumpLibInfo(lib_handle_);
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

    auto& new_plugin_manager = loaded_plugins.emplace_back();
    new_plugin_manager.plugin_api_ = api;
    new_plugin_manager.lib_handle_ = lib_handle;
  }

  return kLrtStatusOk;
}

LrtStatus LrtPluginManager::FreeLib() {
  if (IsLoaded()) {
    LRT_RETURN_STATUS_IF_NOT_OK(CloseLib(lib_handle_));
    plugin_api_ = LrtPluginApi();
    lib_handle_ = nullptr;
  }
  return kLrtStatusOk;
}

}  // namespace lrt::internal
