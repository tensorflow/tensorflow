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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_PLUGIN_MANAGER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_PLUGIN_MANAGER_H_

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/core/compiler_plugin_api.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"

namespace lrt::internal {

// Syntatic sugar around dynamically loaded LrtCompilerPlugin libraries.
class LrtPluginManager {
 public:
  // Search for shared library files with prefix "libLrtPlugin" in the
  // directories passed through "lib_search_paths". Populates "loaded_plugins"
  // with resolved plugin apis for each found library that can be succesfully
  // loaded. Additionally initializes the compiler plugin instances
  // and stores handle.
  static LrtStatus LoadPlugins(
      absl::Span<const absl::string_view> lib_search_paths,
      std::vector<LrtPluginManager>& loaded_plugins);

  // Dump information about the loaded shared library like library dependencies.
  // See "dlinfo".
  void DumpLibInfo() const;

  // Dump static information about a compiler plugin. Does nothing if it
  // isn't loaded.
  void DumpPluginInfo() const;

  // Resolved function pointers to a dynamically loaded `LrtCompilerPlugin`
  // instance. Lifetimes of all such funtions are tied to the underlying
  // shared library handle.
  const LrtPluginApi& Api();

  // Handle to current LrtCompilerPlugin instances instatiated through
  // the Api.
  LrtCompilerPlugin PluginHandle();

  // Handle to current LrtCompiledResult instances instatiated through
  // the Api.
  LrtCompiledResult CompiledResultHandle();

  LrtPluginManager() = default;

  LrtPluginManager(LrtPluginManager&& other);
  LrtPluginManager& operator=(LrtPluginManager&& other);

  LrtPluginManager(const LrtPluginManager& other) = delete;
  LrtPluginManager& operator=(const LrtPluginManager& other) = delete;

  // Destroys any living `LrtCompilerPlugin`. Does not free library.
  ~LrtPluginManager();

 private:
  void* lib_handle_ = nullptr;
  LrtPluginApi plugin_api_ = {};
  LrtCompilerPlugin plugin_handle_ = nullptr;
  LrtCompiledResult compiled_result_handle_ = nullptr;
};

}  // namespace lrt::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_PLUGIN_MANAGER_H_
