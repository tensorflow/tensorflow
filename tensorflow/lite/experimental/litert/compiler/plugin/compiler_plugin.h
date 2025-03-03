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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_flags.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin_api.h"

// C++ wrappers and high-level functions for managing compiler plugins
// and applying them to models.

namespace litert::internal {

// Name and index of byte code.
using CallInfo = std::pair<absl::string_view, LiteRtParamIndex>;

// Wraps vendor compiled result. Must be outlived by the CompilerPlugin
// the generated it.
class CompiledResult {
 public:
  friend class CompilerPlugin;

  // Number of byte code modules compiled by the plugin.
  Expected<LiteRtParamIndex> NumByteCodeModules() const;

  // Get the single module of compiled byte code. This contains the
  // compilation result for all entry points.
  Expected<BufferRef<uint8_t>> ByteCode(
      LiteRtParamIndex byte_code_idx = 0) const;

  // Get information regarding the "ith" entry points in the compiled module.
  // There will be oe entry point for each subgraph compiled for.
  Expected<CallInfo> CallInfo(LiteRtParamIndex call_idx) const;

  // Get the number of entry points in the compiled module. This will be equal
  // to the number of subgraphs passed to the compilation step.
  Expected<LiteRtParamIndex> NumCalls() const;

  explicit CompiledResult(const LiteRtCompilerPluginApi& parent)
      : parent_(parent) {}

  CompiledResult(CompiledResult&& other);
  CompiledResult& operator=(CompiledResult&& other);
  CompiledResult(const CompiledResult& other) = delete;
  CompiledResult& operator=(const CompiledResult& other) = delete;

  ~CompiledResult();

 private:
  LiteRtCompilerPluginApi parent_;
  LiteRtCompiledResult compiled_result_handle_ = nullptr;
};

// Wraps vendor compiler plugin.
class CompilerPlugin {
 public:
  std::string DebugString() const;

  // Get the compiler plugin's API version.
  Expected<LiteRtApiVersion> ApiVersion() const;

  // Get the supported HW accelerators (e.g., GPU, NPU).
  Expected<LiteRtHwAccelerators> SupportedHardware() const;

  // Get the manufacturer associated with this plugin. NOTE: SocManufacturer
  // string returned by the underlying plugin are expected to have static
  // lifetime.
  absl::string_view SocManufacturer() const {
    return plugin_api_.get_compiler_plugin_soc_manufacturer();
  }

  // Get list of unique soc models targetable by this plugin.
  const std::vector<std::string>& SocModels() const { return soc_models_; }

  // Selects ops for the plugin to compile.
  Expected<std::vector<LiteRtOpWithPartitionIndex>> Partition(
      const Subgraph& subgraph);

  // Compile given LiteRtSubgraphs. Result object must be outlived by
  // this CompilerPlugin.
  Expected<CompiledResult> Compile(LiteRtModel partitions,
                                   absl::string_view soc_model = "");

  // Search for shared library files with prefix "libLiteRtCompilerPlugin" in
  // the directories passed through "lib_search_paths". Populates
  // "loaded_plugins" with resolved plugin apis for each found library that can
  // be succesfully loaded. Additionally initializes the compiler plugin
  // instances and stores handle.
  static Expected<std::vector<CompilerPlugin>> LoadPlugins(
      absl::Span<const absl::string_view> lib_search_paths);

  // Set compiler flags within the plugin.
  LiteRtStatus SetFlags(const CompilerFlags& flags) {
    return flags.SetPluginFlags(plugin_handle_, plugin_api_.set_flags);
  }

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

  std::vector<std::string> soc_models_;
  void* lib_handle_ = nullptr;
  LiteRtCompilerPluginApi plugin_api_ = {};
  LiteRtCompilerPlugin plugin_handle_ = nullptr;

  // Internal LiteRtCompiledResult wrapper.

  CompiledResult MakeResult() const { return CompiledResult(plugin_api_); }
};

// Higher level functions for applying plugin to graph.
//===---------------------------------------------------------------------------

// Dispatch op references and their subgraph to be compiled.
using PartitionResult =
    std::pair<std::vector<LiteRtOp>, typename LiteRtSubgraphT::Alloc>;

// Applies just the partition phase of the plugin on the model. Returns
// references newly allocated subgraphs removed from input and their
// corresponding dispatch ops in the input.
Expected<PartitionResult> PartitionModel(CompilerPlugin& compiler_plugin,
                                         LiteRtModelT& model);

// Same as "PartitionModel" choose partitions directly based on the selected
// ops. Selected ops may contain any ops in the the main subgraph of the model.
// This function will seperate them into DAGs and slice the model accordingly.
Expected<PartitionResult> PartitionModelDirect(
    std::vector<LiteRtOpWithPartitionIndex> selected_ops, LiteRtModelT& model);

// Applies both the partition and compile steps to the model. Generated
// byte_code will be internalized within the model for later serialization.
Expected<void> ApplyPlugin(CompilerPlugin& compiler_plugin, LiteRtModelT& model,
                           absl::string_view soc_model = "");

// Applies the compilation step to the model given a pre-determined partition.
Expected<void> ApplyPluginWithPartition(CompilerPlugin& compiler_plugin,
                                        LiteRtModelT& model,
                                        PartitionResult partitions,
                                        absl::string_view soc_model = "");

// Apply all available plugins providing the selected HW accelerators to the
// given model, modify the model accordingly, and return (1) the number of
// compiler plugins successfully applied, (2) a new flatbuffer backing the
// modified model, (3) a string listing the compiler plugins that were
// successfully applied, and (4) a string listing the compiler plugins that
// failed to apply with an associated error message.
struct ApplyPluginsResult {
  size_t num_applied_plugins;
  OwningBufferRef<uint8_t> new_flatbuffer;
  std::string success_message;
  std::string error_message;
};

Expected<ApplyPluginsResult> ApplyPlugins(
    LiteRtEnvironment environment, LiteRtModel model,
    LiteRtHwAcceleratorSet selected_hw_accelerators);

}  // namespace litert::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_COMPILER_PLUGIN_COMPILER_PLUGIN_H_
