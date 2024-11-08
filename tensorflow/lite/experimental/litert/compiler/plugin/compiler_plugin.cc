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

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin_api.h"

namespace litert::internal {

//
// CompiledResult
//

std::string CompiledResult::BytesT::String() const {
  return std::string(data, size);
}

LiteRtResult<CompiledResult::BytesT> CompiledResult::ByteCode() const {
  BytesT byte_code;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.get_compiled_result_byte_code(
          compiled_result_handle_,
          reinterpret_cast<const void**>(&byte_code.data), &byte_code.size),
      BytesT);
  return LiteRtResult<BytesT>::FromValue(byte_code);
}

LiteRtResult<LiteRtParamIndex> CompiledResult::NumCalls() const {
  LiteRtParamIndex call_idx;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.get_compiled_result_num_calls(
          compiled_result_handle_, &call_idx),
      LiteRtParamIndex);
  return LiteRtResult<LiteRtParamIndex>::FromValue(call_idx);
}

LiteRtResult<std::string> CompiledResult::CallInfo(
    LiteRtParamIndex call_idx) const {
  BytesT call_info;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.get_compiled_result_call_info(
          compiled_result_handle_, call_idx,
          reinterpret_cast<const void**>(&call_info.data), &call_info.size),
      std::string);
  return LiteRtResult<std::string>::FromValue(call_info.String());
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

absl::StatusOr<std::vector<std::string>> GetSocModels(
    const LiteRtCompilerPluginApi& api, LiteRtCompilerPlugin plugin_handle) {
  std::vector<std::string> soc_models;
  LiteRtParamIndex num_models;
  if (api.get_num_compiler_plugin_supported_models(
          plugin_handle, &num_models) != kLiteRtStatusOk) {
    return absl::InternalError("Failed to get number of supported SoC models");
  }
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

CompilerPlugin::ResultT CompilerPlugin::LoadPlugin(
    const absl::string_view lib_path) {
  LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.data());
  CompilerPlugin plugin;

  if (OpenLib(lib_path, &plugin.lib_handle_) != kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING, "Failed to load plugin at: %s", lib_path.data());
    return ResultT::FromStatus(kLiteRtStatusErrorDynamicLoading);
  }
  LITERT_LOG(LITERT_INFO, "Loaded plugin at: %s", lib_path.data());

  if (ResolvePluginApi(plugin.lib_handle_, plugin.plugin_api_) !=
      kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING, "Failed to resolve plugin api at: %s",
               lib_path.data());
    return ResultT::FromStatus(kLiteRtStatusErrorDynamicLoading);
  }
  LITERT_LOG(LITERT_INFO, "Resolved plugin api at: %s", lib_path.data());

  if (plugin.plugin_api_.create_compiler_plugin(&plugin.plugin_handle_) !=
      kLiteRtStatusOk) {
    LITERT_LOG(LITERT_WARNING, "Failed to initialize plugin at: %s",
               lib_path.data());
    if (CloseLib(plugin.lib_handle_) != kLiteRtStatusOk) {
      LITERT_LOG(LITERT_WARNING, "Failed to close loaded library at: %s",
                 lib_path.data());
    }
    return ResultT::FromStatus(kLiteRtStatusErrorDynamicLoading);
  }

  if (auto api_version = plugin.ApiVersion();
      api_version.Status() != kLiteRtStatusOk) {
    return ResultT::FromStatus(api_version.Status());
  } else if (api_version.Value().major != LITERT_API_VERSION_MAJOR) {
    LITERT_LOG(
        LITERT_ERROR,
        "Unsupported Compiler Plugin version, found version %d.%d.%d and "
        "expected version %d.%d.%d",
        api_version.Value().major, api_version.Value().minor,
        api_version.Value().patch, LITERT_API_VERSION_MAJOR,
        LITERT_API_VERSION_MINOR, LITERT_API_VERSION_PATCH);
    return ResultT::FromStatus(kLiteRtStatusErrorRuntimeFailure);
  }

  // This should never change throughout the lifetime of the compiler
  // plugin so save to avoid recalling.
  if (auto soc_models = GetSocModels(plugin.plugin_api_, plugin.plugin_handle_);
      soc_models.ok()) {
    plugin.soc_models_ = *soc_models;
  } else {
    return ResultT::FromStatus(kLiteRtStatusErrorRuntimeFailure);
  }

  return ResultT::TakeValue(std::move(plugin));
}

CompilerPlugin::ResultVecT CompilerPlugin::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    LITERT_RETURN_RESULT_IF_NOT_OK(
        FindLiteRtSharedLibs(search_path, plugin_lib_paths), VecT);
  }

  VecT loaded_plugins;
  loaded_plugins.reserve(lib_search_paths.size());

  for (const auto& lib_path : plugin_lib_paths) {
    LITERT_LOG(LITERT_INFO, "Loading plugin at: %s", lib_path.c_str());
    auto result = LoadPlugin(lib_path);
    if (!result.HasValue()) {
      continue;
    }
    loaded_plugins.push_back(std::move(result.Value()));
  }

  return ResultVecT::TakeValue(std::move(loaded_plugins));
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

LiteRtResult<LiteRtApiVersion> CompilerPlugin::ApiVersion() const {
  LiteRtApiVersion api_version;
  LITERT_RETURN_RESULT_IF_NOT_OK(
      plugin_api_.get_compiler_plugin_version(&api_version), LiteRtApiVersion);
  return LiteRtResult<LiteRtApiVersion>::FromValue(api_version);
}

LiteRtResult<std::vector<LiteRtOp>> CompilerPlugin::PartitionModel(
    const Model& model) {
  LiteRtOpListT ops;
  // TODO: Use const where appropriate in the C compiler plugin api.
  LiteRtModel c_model = model.Get();
  LITERT_RETURN_RESULT_IF_NOT_OK(plugin_api_.compiler_plugin_partition_model(
                                     plugin_handle_, c_model, &ops),
                                 std::vector<LiteRtOp>);
  return LiteRtResult<std::vector<LiteRtOp>>::TakeValue(ops.Vec());
}

LiteRtStatus CompilerPlugin::Compile(
    const absl::string_view soc_model,
    const std::vector<LiteRtSubgraph>& partitions, std::ostream& byte_code_out,
    std::vector<std::string>& call_info_out) {
  CompiledResult result = MakeResult();

  // Compile given partitions into result.
  // TODO: Use const where appropriate in the C compiler plugin api.
  LiteRtSubgraphArray partitions_arr =
      const_cast<LiteRtSubgraphArray>(partitions.data());
  LITERT_RETURN_STATUS_IF_NOT_OK(plugin_api_.compiler_plugin_compile(
      plugin_handle_, soc_model.data(), partitions_arr, partitions.size(),
      &result.compiled_result_handle_));

  // Parse call info from the result.
  {
    LITERT_ASSIGN_OR_RETURN_STATUS(auto num_call, result.NumCalls());
    LITERT_ENSURE(
        num_call == partitions.size(), kLiteRtStatusErrorRuntimeFailure,
        "Plugin didn't return call info for each partition compiled.\n");
    for (int i = 0; i < num_call; ++i) {
      LITERT_ASSIGN_OR_RETURN_STATUS(call_info_out.emplace_back(),
                                     result.CallInfo(i));
    }
  }

  // Parse byte code from result.
  {
    LITERT_ASSIGN_OR_RETURN_STATUS(const CompiledResult::BytesT byte_code,
                                   result.ByteCode());
    LITERT_LOG(LITERT_INFO, "Compiled %d partitions in %lu bytes",
               partitions.size(), byte_code.size);
    byte_code_out.write(byte_code.data, byte_code.size);
  }

  return kLiteRtStatusOk;
}

}  // namespace litert::internal
