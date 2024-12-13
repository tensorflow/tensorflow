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

#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/compiler/plugin/algo.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/model/ir_allocator.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

//
// CompiledResult
//

Expected<BufferRef<uint8_t>> CompiledResult::ByteCode() const {
  const void* data;
  size_t size;
  LITERT_EXPECT_OK(parent_.get_compiled_result_byte_code(
      compiled_result_handle_, &data, &size));
  return BufferRef(data, size);
}

Expected<LiteRtParamIndex> CompiledResult::NumCalls() const {
  LiteRtParamIndex call_idx;
  LITERT_EXPECT_OK(parent_.get_compiled_result_num_calls(
      compiled_result_handle_, &call_idx));
  return call_idx;
}

Expected<absl::string_view> CompiledResult::CallInfo(
    LiteRtParamIndex call_idx) const {
  const void* data;
  size_t size;
  LITERT_EXPECT_OK(parent_.get_compiled_result_call_info(
      compiled_result_handle_, call_idx, &data, &size));
  return absl::string_view(reinterpret_cast<const char*>(data), size);
}

CompiledResult::~CompiledResult() {
  if (compiled_result_handle_ != nullptr) {
    parent_.destroy_compiled_result(compiled_result_handle_);
  }
}

CompiledResult::CompiledResult(CompiledResult&& other)
    : parent_(other.parent_),
      compiled_result_handle_(other.compiled_result_handle_) {
  other.parent_ = {};
  other.compiled_result_handle_ = nullptr;
}

CompiledResult& CompiledResult::operator=(CompiledResult&& other) {
  if (this != &other) {
    parent_ = other.parent_;
    other.parent_ = {};

    compiled_result_handle_ = other.compiled_result_handle_;
    other.compiled_result_handle_ = nullptr;
  }
  return *this;
}

//
// CompilerPlugin
//

namespace {

#define RESOLVE_API_FUNC(name, dest) \
  LITERT_RETURN_STATUS_IF_NOT_OK(    \
      ResolveLibSymbol<decltype(dest)>(lib_handle, name, &dest));

LiteRtStatus ResolvePluginApi(void* lib_handle,
                              LiteRtCompilerPluginApi& result) {
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginVersion,
                   result.get_compiler_plugin_version);
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginSocManufacturer,
                   result.get_compiler_plugin_soc_manufacturer);
  RESOLVE_API_FUNC(kLiteRtGetNumCompilerPluginSupportedSocModels,
                   result.get_num_compiler_plugin_supported_models);
  RESOLVE_API_FUNC(kLiteRtGetCompilerPluginSupportedSocModel,
                   result.get_compiler_plugin_supported_soc_model);

  RESOLVE_API_FUNC(kLiteRtCreateCompilerPlugin, result.create_compiler_plugin);
  RESOLVE_API_FUNC(kLiteRtDestroyCompilerPlugin,
                   result.destroy_compiler_plugin);

  RESOLVE_API_FUNC(kLiteRtCompilerPluginPartition,
                   result.compiler_plugin_partition);
  RESOLVE_API_FUNC(kLiteRtCompilerPluginCompile,
                   result.compiler_plugin_compile);

  RESOLVE_API_FUNC(kLiteRtDestroyCompiledResult,
                   result.destroy_compiled_result);
  RESOLVE_API_FUNC(kLiteRtGetCompiledResultByteCode,
                   result.get_compiled_result_byte_code);
  RESOLVE_API_FUNC(kLiteRtGetCompiledResultCallInfo,
                   result.get_compiled_result_call_info);
  RESOLVE_API_FUNC(kLiteRtGetNumCompiledResultCalls,
                   result.get_compiled_result_num_calls);

  return kLiteRtStatusOk;
}

Expected<SmallVec<std::string>> GetSocModels(
    const LiteRtCompilerPluginApi& api, LiteRtCompilerPlugin plugin_handle) {
  SmallVec<std::string> soc_models;

  LiteRtParamIndex num_models;
  LITERT_EXPECT_OK(
      api.get_num_compiler_plugin_supported_models(plugin_handle, &num_models));

  for (LiteRtParamIndex i = 0; i < num_models; ++i) {
    const char* model;
    if (api.get_compiler_plugin_supported_soc_model(plugin_handle, i, &model) !=
        kLiteRtStatusOk) {
      continue;
    }
    soc_models.push_back(std::string(model));
  }

  return soc_models;
}

std::string ResolveSocModel(const CompilerPlugin& plugin,
                            absl::string_view soc_model = "") {
  const auto& default_model = plugin.SocModels().front();
  if (soc_model.empty()) {
    LITERT_LOG(LITERT_INFO, "Using default soc_model: %s",
               default_model.c_str());
    return default_model;
  }
  return std::string(soc_model);
}

}  // namespace

Expected<CompilerPlugin> CompilerPlugin::LoadPlugin(
    const absl::string_view lib_path) {
  CompilerPlugin plugin;
  LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.data());

  LITERT_EXPECT_OK(OpenLib(lib_path, &plugin.lib_handle_));
  LITERT_LOG(LITERT_INFO, "Loaded plugin at: %s", lib_path.data());

  LITERT_EXPECT_OK(ResolvePluginApi(plugin.lib_handle_, plugin.plugin_api_));
  LITERT_LOG(LITERT_INFO, "Resolved plugin api at: %s", lib_path.data());

  LITERT_EXPECT_OK(
      plugin.plugin_api_.create_compiler_plugin(&plugin.plugin_handle_));
  LITERT_LOG(LITERT_INFO, "Initialize plugin at: %s", lib_path.data());

  auto api_version = plugin.ApiVersion();
  if (!api_version) {
    return api_version.Error();
  }

  if (api_version->major != LITERT_API_VERSION_MAJOR) {
    LITERT_LOG(
        LITERT_ERROR,
        "Unsupported Compiler Plugin version, found version %d.%d.%d and "
        "expected version %d.%d.%d",
        api_version.Value().major, api_version.Value().minor,
        api_version.Value().patch, LITERT_API_VERSION_MAJOR,
        LITERT_API_VERSION_MINOR, LITERT_API_VERSION_PATCH);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  // This should never change throughout the lifetime of the compiler
  // plugin so save to avoid recalling.
  auto soc_models = GetSocModels(plugin.plugin_api_, plugin.plugin_handle_);
  if (!soc_models) {
    return soc_models.Error();
  }
  plugin.soc_models_ = *soc_models;

  return plugin;
}

Expected<SmallVec<CompilerPlugin>> CompilerPlugin::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    // Skip paths that are not valid.
    if (Exists(search_path)) {
      LITERT_EXPECT_OK(FindLiteRtSharedLibs(search_path, plugin_lib_paths));
    }
  }

  SmallVec<CompilerPlugin> loaded_plugins;
  loaded_plugins.reserve(lib_search_paths.size());

  for (const auto& lib_path : plugin_lib_paths) {
    LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.c_str());
    auto plugin = LoadPlugin(lib_path);
    if (!plugin.HasValue()) {
      continue;
    }
    loaded_plugins.push_back(std::move(plugin.Value()));
  }

  return loaded_plugins;
}

Expected<CompilerPlugin> CompilerPlugin::LoadPlugin(
    absl::Span<const absl::string_view> lib_search_paths,
    absl::string_view soc_manufacturer) {
  auto compiler_plugins = LoadPlugins(lib_search_paths);
  if (!compiler_plugins) {
    return compiler_plugins.Error();
  }

  for (auto& plugin : *compiler_plugins) {
    if (plugin.SocManufacturer() == soc_manufacturer) {
      return std::move(plugin);
    }
  }

  return Error(kLiteRtStatusErrorNotFound);
}

CompilerPlugin::CompilerPlugin(CompilerPlugin&& other)
    : soc_models_(std::move(other.soc_models_)),
      lib_handle_(other.lib_handle_),
      plugin_api_(std::move(other.plugin_api_)),
      plugin_handle_(other.plugin_handle_) {
  other.soc_models_ = {};
  other.plugin_api_ = {};
  other.lib_handle_ = nullptr;
  other.plugin_handle_ = nullptr;
}

CompilerPlugin& CompilerPlugin::operator=(CompilerPlugin&& other) {
  if (this != &other) {
    soc_models_ = std::move(other.soc_models_);
    other.soc_models_ = {};

    lib_handle_ = other.lib_handle_;
    other.lib_handle_ = nullptr;

    plugin_api_ = std::move(other.plugin_api_);
    other.plugin_api_ = {};

    plugin_handle_ = other.plugin_handle_;
    other.plugin_handle_ = nullptr;
  }
  return *this;
}

CompilerPlugin::~CompilerPlugin() {
  if (plugin_handle_ != nullptr) {
    plugin_api_.destroy_compiler_plugin(plugin_handle_);
  }
  if (lib_handle_ != nullptr) {
    if (kLiteRtStatusOk != CloseLib(lib_handle_)) {
      LITERT_LOG(LITERT_WARNING, "%s", "Failed to close shared library\n");
    }
  }
}

Expected<LiteRtApiVersion> CompilerPlugin::ApiVersion() const {
  LiteRtApiVersion api_version;
  LITERT_EXPECT_OK(plugin_api_.get_compiler_plugin_version(&api_version));
  return api_version;
}

Expected<std::vector<LiteRtOp>> CompilerPlugin::Partition(
    const Subgraph& subgraph) {
  LiteRtOpListT ops;
  LITERT_EXPECT_OK(plugin_api_.compiler_plugin_partition(plugin_handle_,
                                                         subgraph.Get(), &ops));
  return ops.Vec();
}

Expected<CompiledResult> CompilerPlugin::Compile(
    absl::Span<LiteRtSubgraph> partitions, absl::string_view soc_model) {
  CompiledResult result = MakeResult();
  const auto soc_model_str = ResolveSocModel(*this, soc_model);
  LITERT_EXPECT_OK(plugin_api_.compiler_plugin_compile(
      plugin_handle_, soc_model_str.c_str(), partitions.data(),
      partitions.size(), &result.compiled_result_handle_));
  return result;
}

namespace {

LiteRtStatus PartitionSubgraph(CompilerPlugin& compiler_plugin,
                               LiteRtSubgraphT& subgraph,
                               PartitionResult& result) {
  // Get selected ops from plugin.
  auto selected_ops = compiler_plugin.Partition(Subgraph(&subgraph));
  if (!selected_ops) {
    LITERT_LOG(LITERT_ERROR, "Failed to get partitions from plugin");
    return selected_ops.Error().Status();
  }

  // Group selected ops into connected islands.
  auto islands = GroupPartitions(*selected_ops);
  if (islands.empty()) {
    LITERT_LOG(LITERT_ERROR, "Failed to group partitions");
    return kLiteRtStatusErrorRuntimeFailure;
  }

  // For each connected island, slice into new subgraph and replace use with
  // single dispatch op.
  for (auto& island : islands) {
    auto& new_subgraph = result.second.EmplaceBack();
    auto* dispatch_op = OutlinePartition(subgraph, &new_subgraph, island);
    result.first.push_back(dispatch_op);
  }

  return kLiteRtStatusOk;
}

}  // namespace

Expected<PartitionResult> PartitionModel(CompilerPlugin& compiler_plugin,
                                         LiteRtModelT& model) {
  // Accumulate partition results for each subgraph in model.
  PartitionResult result;
  for (auto* subgraph : model.Subgraphs()) {
    LITERT_EXPECT_OK(PartitionSubgraph(compiler_plugin, *subgraph, result));
  }
  ABSL_DCHECK_EQ(result.first.size(), result.second.Size());
  return result;
}

LiteRtStatus Apply(CompilerPlugin& compiler_plugin, LiteRtModelT& model,
                   absl::string_view soc_model, Serialization serialization) {
  // Collect partitions to pass to compilation.
  auto partitions = PartitionModel(compiler_plugin, model);
  if (!partitions) {
    LITERT_LOG(LITERT_ERROR, "Failed to partition model");
    return partitions.Error().Status();
  }

  auto& dispatch_ops = partitions->first;
  auto& subgraphs = partitions->second;

  // Pass sliced subgraphs to plugin for compilation.
  const auto soc_model_str = ResolveSocModel(compiler_plugin, soc_model);
  auto compiled_result =
      compiler_plugin.Compile(subgraphs.Elements(), soc_model_str);
  if (!compiled_result) {
    LITERT_LOG(LITERT_ERROR, "Failed to compile");
    return compiled_result.Error().Status();
  }

  // Attach per-partition call info to the respective op.
  // This data may be adjusted during serialization. Just passthrough for now.
  for (auto i = 0; i < dispatch_ops.size(); ++i) {
    auto call_info = compiled_result->CallInfo(i);
    if (!call_info) {
      LITERT_LOG(LITERT_ERROR,
                 "Failed to get call info from compilation result");
      return call_info.Error().Status();
    }
    auto exec_info = MakeExecInfo(*call_info, kByteCodeMetadataKey);
    if (!exec_info) {
      LITERT_LOG(LITERT_ERROR, "Failed to serialize call info");
      return exec_info.Error().Status();
    }
    dispatch_ops.at(i)->SetCustomOptions(std::move(*exec_info));
  }

  // Store the byte code in a metadata buffer. This data may be adjusted during
  // serialization. Just passthrough for now.
  auto byte_code = compiled_result->ByteCode();
  if (!byte_code) {
    LITERT_LOG(LITERT_ERROR, "Failed to get bytecode from compiled result");
    return byte_code.Error().Status();
  }
  model.PushMetadata(kByteCodeMetadataKey, byte_code->StrView());

  // Tag the model with make/model from the plugin.
  auto build_stamp = MakeBuildStamp(compiler_plugin.SocManufacturer(),
                                    soc_model_str, serialization);
  if (!build_stamp) {
    LITERT_LOG(LITERT_ERROR, "Failed to stamp model");
    return build_stamp.Error().Status();
  }
  LITERT_RETURN_STATUS_IF_NOT_OK(
      model.PushMetadata(kLiteRtBuildStampKey, std::move(*build_stamp)));

  return kLiteRtStatusOk;
}

}  // namespace litert::internal
