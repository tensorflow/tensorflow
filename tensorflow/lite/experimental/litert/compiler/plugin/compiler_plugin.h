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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_PLUGIN_H_

#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

class CompiledResult {
  friend class CompilerPlugin;
  // Get the single module of compiled byte code. This contains the
  // compilation result for all entry points.
  Expected<BufferRef<uint8_t>> ByteCode() const;

  // Get information regarding the "ith" entry points in the compiled module.
  // There will be oe entry point for each subgraph compiled for.
  Expected<std::string> CallInfo(LiteRtParamIndex call_idx) const;

  // Get the number of entry points in the compiled module. This will be equal
  // to the number of subgraphs passed to the compilation step.
  Expected<LiteRtParamIndex> NumCalls() const;

  explicit CompiledResult(const LiteRtCompilerPluginApi& allocating_plugin_api)
      : allocating_plugin_api_(allocating_plugin_api) {}

  CompiledResult(CompiledResult&& other) = default;
  CompiledResult& operator=(CompiledResult&& other) = default;
  CompiledResult(const CompiledResult& other) = delete;
  CompiledResult& operator=(const CompiledResult& other) = delete;

  ~CompiledResult();

  LiteRtCompilerPluginApi allocating_plugin_api_;
  LiteRtCompiledResult compiled_result_handle_ = nullptr;
};

// Syntatic sugar around dynamically loaded LiteRtCompilerPlugin libraries.
// TODO turn this into a general C++ wraper for the whole compiler plugin api.
class CompilerPlugin {
 public:
  // Get the compiler plugin's API version.
  Expected<LiteRtApiVersion> ApiVersion() const;

  // Get the manufacturer associated with this plugin. NOTE: SocManufacturer
  // string returned by the underlying plugin are expected to have static
  // lifetime.
  absl::string_view SocManufacturer() const {
    return plugin_api_.get_compiler_plugin_soc_manufacturer();
  }

  // Get list of unique soc models targetable by this plugin.
  const SmallVec<std::string>& SocModels() const { return soc_models_; }

  // Selects ops for the plugin to compile.
  Expected<std::vector<LiteRtOp>> Partition(const Subgraph& subgraph);

  // Compile given LiteRtSubgraphs. Write compiled byte code to the given
  // stream. For each given subgraph, write opaque data about the corresponding
  // entry point to the given "call_info_out". Parameter "soc_model" is optional
  // and can be set to specify the target SoC; for on-device compilation it
  // should be left unspecified so as to let the underlying logic pick the
  // architecture that matches the SoC on the user device.
  LiteRtStatus Compile(std::optional<absl::string_view> soc_model,
                       const std::vector<LiteRtSubgraph>& partitions,
                       std::ostream& byte_code_out,
                       std::vector<std::string>& call_info_out);

  // Search for shared library files with prefix "libLiteRtCompilerPlugin" in
  // the directories passed through "lib_search_paths". Populates
  // "loaded_plugins" with resolved plugin apis for each found library that can
  // be succesfully loaded. Additionally initializes the compiler plugin
  // instances and stores handle.
  static Expected<SmallVec<CompilerPlugin>> LoadPlugins(
      absl::Span<const absl::string_view> lib_search_paths);

  // Search for shared library files with prefix "libLiteRtCompilerPlugin" in
  // the directories passed through "lib_search_paths" and return a compiler
  // plugin instance for a given manufactured, if one is found.
  static Expected<CompilerPlugin> LoadPlugin(
      absl::Span<const absl::string_view> lib_search_paths,
      absl::string_view soc_manufacturer);

  CompilerPlugin(CompilerPlugin&& other);
  CompilerPlugin& operator=(CompilerPlugin&& other);
  CompilerPlugin(const CompilerPlugin& other) = delete;
  CompilerPlugin& operator=(const CompilerPlugin& other) = delete;

  // Destroys any living `LiteRtCompilerPlugin` and frees reference
  // to dynamically loaded library.
  ~CompilerPlugin();

 private:
  static Expected<CompilerPlugin> LoadPlugin(absl::string_view lib_path);
  CompilerPlugin() = default;

  SmallVec<std::string> soc_models_;
  void* lib_handle_ = nullptr;
  LiteRtCompilerPluginApi plugin_api_ = {};
  LiteRtCompilerPlugin plugin_handle_ = nullptr;

  // Internal LiteRtCompiledResult wrapper.

  CompiledResult MakeResult() const { return CompiledResult(plugin_api_); }
};

// Applies the plugin's "partition" and "compile" steps to the given model.
// Returns the serialized model with NPU code appended to the back.  Parameter
// "soc_model" is optional and can be set to specify the target SoC; for
// on-device compilation it should be left unspecified so as to let the
// underlying logic pick the architecture that matches the SoC on the user
// device
Expected<OwningBufferRef<uint8_t>> ApplyPlugin(
    CompilerPlugin& compiler_plugin, Model& model,
    std::optional<absl::string_view> soc_model = std::nullopt);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
