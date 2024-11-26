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
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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
  LITERT_EXPECT_OK(allocating_plugin_api_.get_compiled_result_byte_code(
      compiled_result_handle_, &data, &size));
  return BufferRef(data, size);
}

Expected<LiteRtParamIndex> CompiledResult::NumCalls() const {
  LiteRtParamIndex call_idx;
  LITERT_EXPECT_OK(allocating_plugin_api_.get_compiled_result_num_calls(
      compiled_result_handle_, &call_idx));
  return call_idx;
}

Expected<std::string> CompiledResult::CallInfo(
    LiteRtParamIndex call_idx) const {
  const void* data;
  size_t size;
  LITERT_EXPECT_OK(allocating_plugin_api_.get_compiled_result_call_info(
      compiled_result_handle_, call_idx, &data, &size));
  return std::string(reinterpret_cast<const char*>(data), size);
}

CompiledResult::~CompiledResult() {
  allocating_plugin_api_.destroy_compiled_result(compiled_result_handle_);
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
  RESOLVE_API_FUNC("LiteRtGetCompilerPluginVersion",
                   result.get_compiler_plugin_version);
  RESOLVE_API_FUNC("LiteRtGetCompilerPluginSocManufacturer",
                   result.get_compiler_plugin_soc_manufacturer);
  RESOLVE_API_FUNC("LiteRtGetNumCompilerPluginSupportedSocModels",
                   result.get_num_compiler_plugin_supported_models);
  RESOLVE_API_FUNC("LiteRtGetCompilerPluginSupportedSocModel",
                   result.get_compiler_plugin_supported_soc_model);

  RESOLVE_API_FUNC("LiteRtCreateCompilerPlugin", result.create_compiler_plugin);
  RESOLVE_API_FUNC("LiteRtDestroyCompilerPlugin",
                   result.destroy_compiler_plugin);

  RESOLVE_API_FUNC("LiteRtCompilerPluginPartitionModel",
                   result.compiler_plugin_partition_model);
  RESOLVE_API_FUNC("LiteRtCompilerPluginCompile",
                   result.compiler_plugin_compile);

  RESOLVE_API_FUNC("LiteRtDestroyCompiledResult",
                   result.destroy_compiled_result);
  RESOLVE_API_FUNC("LiteRtGetCompiledResultByteCode",
                   result.get_compiled_result_byte_code);
  RESOLVE_API_FUNC("LiteRtGetCompiledResultCallInfo",
                   result.get_compiled_result_call_info);
  RESOLVE_API_FUNC("LiteRtGetNumCompiledResultCalls",
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

Expected<std::vector<LiteRtOp>> CompilerPlugin::PartitionModel(
    const Model& model) {
  LiteRtOpListT ops;
  LiteRtModel model_handle = model.Get();
  LITERT_EXPECT_OK(plugin_api_.compiler_plugin_partition_model(
      plugin_handle_, model_handle, &ops));
  return ops.Vec();
}

LiteRtStatus CompilerPlugin::Compile(
    std::optional<absl::string_view> soc_model,
    const std::vector<LiteRtSubgraph>& partitions, std::ostream& byte_code_out,
    std::vector<std::string>& call_info_out) {
  CompiledResult result = MakeResult();

  const char* soc_model_str = soc_model ? soc_model->data() : nullptr;

  // Compile given partitions into result.
  // TODO: Use const where appropriate in the C compiler plugin api.
  LiteRtSubgraphArray partitions_arr =
      const_cast<LiteRtSubgraphArray>(partitions.data());
  if (auto stat = plugin_api_.compiler_plugin_compile(
          plugin_handle_, soc_model_str, partitions_arr, partitions.size(),
          &result.compiled_result_handle_);
      stat != kLiteRtStatusOk) {
    return stat;
  }

  // Parse call info from the result.
  {
    auto num_call = result.NumCalls();
    if (!num_call) {
      return num_call.Error().Status();
    }
    if (num_call.Value() != partitions.size()) {
      LITERT_LOG(
          LITERT_ERROR, "%s",
          "Plugin didn't return call info for each partition compiled.\n");
      return kLiteRtStatusErrorRuntimeFailure;
    }
    for (int i = 0; i < num_call.Value(); ++i) {
      auto call_info = result.CallInfo(i);
      if (!call_info) {
        return call_info.Error().Status();
      }
      call_info_out.emplace_back() = *call_info;
    }
  }

  // Parse byte code from result.
  {
    auto byte_code = result.ByteCode();
    if (!byte_code) {
      return byte_code.Error().Status();
    }
    LITERT_LOG(LITERT_INFO, "Compiled %d partitions in %lu bytes",
               partitions.size(), byte_code->Size());
    byte_code->WriteStr(byte_code_out);
  }

  return kLiteRtStatusOk;
}

Expected<OwningBufferRef<uint8_t>> ApplyPlugin(
    CompilerPlugin& compiler_plugin, Model& model,
    std::optional<absl::string_view> soc_model) {
  // Get selected ops from plugin.
  auto partition = compiler_plugin.PartitionModel(model);
  if (!partition) {
    LITERT_LOG(LITERT_ERROR, "Failed to get partitions from plugin");
    return Error(kLiteRtStatusErrorRuntimeFailure);
  }

  // Group selected ops into partitions.
  auto grouped_partitions = GroupPartitions(*partition);
  if (grouped_partitions.empty()) {
    LITERT_LOG(LITERT_ERROR, "Failed to group partitions");
    return Error(kLiteRtStatusErrorRuntimeFailure);
  }

  if (grouped_partitions.size() > 1) {
    LITERT_LOG(LITERT_ERROR, "Apply on multiple partitions not supported yet.");
    return Error(kLiteRtStatusErrorUnsupported);
  }

  // Outline the partitions into new subgraphs.
  std::vector<LiteRtOp> custom_ops;
  for (auto& partition : grouped_partitions) {
    auto custom_op =
        OutlinePartition(*model.Get()->Subgraphs().front(),
                         &model.Get()->EmplaceSubgraph(), partition);
    custom_ops.push_back(custom_op);
  }

  // Pass new subgraphs to the plugin for compilation.
  std::vector<LiteRtSubgraph> compilation_input;
  auto begin = model.Get()->Subgraphs().begin();
  auto end = model.Get()->Subgraphs().end();
  for (auto it = begin + 1; it < end; ++it) {
    compilation_input.push_back(*it);
  }

  // Compile partitions with plugin.
  std::stringstream byte_code;
  std::vector<std::string> exec_info;
  if (auto status = compiler_plugin.Compile(soc_model, compilation_input,
                                            byte_code, exec_info);
      status != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_ERROR, "Failed to compile partitions.");
    return Error(status);
  }

  if (exec_info.size() != custom_ops.size()) {
    LITERT_LOG(LITERT_ERROR,
               "Compilation did not return exec_info for every partition");
    return Error(kLiteRtStatusErrorRuntimeFailure);
  }

  // Attach entry point info to the custom ops.
  auto custom_op_it = custom_ops.begin();
  auto exec_info_it = exec_info.begin();
  for (; custom_op_it < custom_ops.end(); custom_op_it++, exec_info_it++) {
    LiteRtOp custom_op = *custom_op_it;
    const auto& exec_info = *exec_info_it;
    custom_op->SetCustomOptions(exec_info.data());
  }

  const auto byte_code_str = byte_code.str();
  return OwningBufferRef<uint8_t>(
      reinterpret_cast<const uint8_t*>(byte_code_str.data()),
      byte_code_str.size());
}

}  // namespace litert::internal
