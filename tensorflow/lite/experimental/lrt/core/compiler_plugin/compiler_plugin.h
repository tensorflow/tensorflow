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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_

#include <forward_list>
#include <memory>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

class CompiledResult {
  friend class CompilerPlugin;
  struct BytesT {
    const char* data;
    size_t size;

    std::string String() const;
  };

  // Get the single module of compiled byte code. This contains the
  // compilation result for all entry points.
  LiteRtResult<BytesT> ByteCode() const;

  // Get information regarding the "ith" entry points in the compiled module.
  // There will be oe entry point for each subgraph compiled for.
  LiteRtResult<std::string> CallInfo(LiteRtParamIndex call_idx) const;

  // Get the number of entry points in the compiled module. This will be equal
  // to the number of subgraphs passed to the compilation step.
  LiteRtResult<LiteRtParamIndex> NumCalls() const;

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
  using VecT = std::vector<CompilerPlugin>;
  using ResultT = LiteRtResult<CompilerPlugin>;
  using ResultVecT = LiteRtResult<VecT>;

  // Get the manufacturer associated with this plugin. NOTE: SocManufacturer
  // string returned by the underlying plugin are expected to have static
  // lifetime.
  absl::string_view SocManufacturer() const {
    return plugin_api_.soc_manufacturer();
  }

  // Get list of unique soc models targetable by this plugin.
  const std::vector<std::string>& SocModels() const { return soc_models_; }

  // Selects ops for the plugin to compile.
  LiteRtResult<std::vector<LiteRtOp>> PartitionModel(const LiteRtModelT& model);

  // Compile given LiteRtSubgraphs for target "soc_model". Write compiled byte
  // code to the given stream. For each given subgraph, write opaque data about
  // the corresponding entry point to the given "call_info_out".
  LiteRtStatus Compile(absl::string_view soc_model,
                       const std::vector<LiteRtSubgraph>& partitions,
                       std::ostream& byte_code_out,
                       std::vector<std::string>& call_info_out);

  // Search for shared library files with prefix "libLiteRtPlugin" in the
  // directories passed through "lib_search_paths". Populates "loaded_plugins"
  // with resolved plugin apis for each found library that can be succesfully
  // loaded. Additionally initializes the compiler plugin instances
  // and stores handle.
  static ResultVecT LoadPlugins(
      absl::Span<const absl::string_view> lib_search_paths);

  CompilerPlugin(CompilerPlugin&& other);
  CompilerPlugin& operator=(CompilerPlugin&& other);
  CompilerPlugin(const CompilerPlugin& other) = delete;
  CompilerPlugin& operator=(const CompilerPlugin& other) = delete;

  // Destroys any living `LiteRtCompilerPlugin` and frees reference
  // to dynamically loaded library.
  ~CompilerPlugin();

 private:
  static ResultT LoadPlugin(absl::string_view lib_path);
  CompilerPlugin() = default;

  std::vector<std::string> soc_models_;
  void* lib_handle_ = nullptr;
  LiteRtCompilerPluginApi plugin_api_ = {};
  LiteRtCompilerPlugin plugin_handle_ = nullptr;

  // Internal LiteRtCompiledResult wrapper.

  CompiledResult MakeResult() const { return CompiledResult(plugin_api_); }
};

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
