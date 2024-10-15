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

#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/compiler_plugin.h"

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"
#include "tensorflow/lite/experimental/lrt/core/logging.h"
#include "tensorflow/lite/experimental/lrt/core/model.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin_api.h"

namespace lrt::internal {

//
// CompiledResult
//

std::string CompiledResult::BytesT::String() const {
  return std::string(data, size);
}

LrtResult<CompiledResult::BytesT> CompiledResult::ByteCode() const {
  BytesT byte_code;
  LRT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.compiled_result_get_byte_code(
          compiled_result_handle_,
          reinterpret_cast<const void**>(&byte_code.data), &byte_code.size),
      BytesT);
  return LrtResult<BytesT>::FromValue(byte_code);
}

LrtResult<lrt_param_index_t> CompiledResult::NumCalls() const {
  lrt_param_index_t call_idx;
  LRT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.compiled_result_get_num_calls(
          compiled_result_handle_, &call_idx),
      lrt_param_index_t);
  return LrtResult<lrt_param_index_t>::FromValue(call_idx);
}

LrtResult<std::string> CompiledResult::CallInfo(
    lrt_param_index_t call_idx) const {
  BytesT call_info;
  LRT_RETURN_RESULT_IF_NOT_OK(
      allocating_plugin_api_.compiled_result_get_call_info(
          compiled_result_handle_, call_idx,
          reinterpret_cast<const void**>(&call_info.data), &call_info.size),
      std::string);
  return LrtResult<std::string>::FromValue(call_info.String());
}

CompiledResult::~CompiledResult() {
  allocating_plugin_api_.compiled_result_destroy(compiled_result_handle_);
}

//
// CompilerPlugin
//

namespace {

#define RESOLVE_API_FUNC(ty, name, dest) \
  LRT_RETURN_STATUS_IF_NOT_OK(ResolveLibSymbol<ty>(lib_handle, name, &dest));

LrtStatus ResolvePluginApi(void* lib_handle, LrtCompilerPluginApi& result) {
  RESOLVE_API_FUNC(LrtPluginApiSocManufacturer, "LrtPluginSocManufacturer",
                   result.soc_manufacturer);
  RESOLVE_API_FUNC(LrtPluginApiNumSupportedModels,
                   "LrtPluginNumSupportedSocModels",
                   result.num_supported_models);
  RESOLVE_API_FUNC(LrtPluginApiGetSupportedSocModel,
                   "LrtPluginGetSupportedSocModel",
                   result.get_supported_soc_model);

  RESOLVE_API_FUNC(LrtPluginApiInit, "LrtPluginInit", result.init);
  RESOLVE_API_FUNC(LrtPluginApiDestroy, "LrtPluginDestroy", result.destroy);

  RESOLVE_API_FUNC(LrtPluginApiPartitionModel, "LrtPluginPartitionModel",
                   result.partition_model);
  RESOLVE_API_FUNC(LrtPluginApiCompile, "LrtPluginCompile", result.compile);

  RESOLVE_API_FUNC(LrtCompiledResultApiDestroy, "LrtCompiledResultDestroy",
                   result.compiled_result_destroy);
  RESOLVE_API_FUNC(LrtCompiledResultApiGetByteCode,
                   "LrtCompiledResultGetByteCode",
                   result.compiled_result_get_byte_code);
  RESOLVE_API_FUNC(LrtCompiledResultApiGetCallInfo,
                   "LrtCompiledResultGetCallInfo",
                   result.compiled_result_get_call_info);
  RESOLVE_API_FUNC(LrtCompiledResultApiGetNumCalls,
                   "LrtCompiledResultGetNumCalls",
                   result.compiled_result_get_num_calls);
  return kLrtStatusOk;
}

std::vector<std::string> GetSocModels(const LrtCompilerPluginApi& api,
                                      LrtCompilerPlugin plugin_handle) {
  std::vector<std::string> soc_models;
  const lrt_param_index_t num_models = api.num_supported_models(plugin_handle);
  for (lrt_param_index_t i = 0; i < num_models; ++i) {
    const char* model;
    if (api.get_supported_soc_model(plugin_handle, i, &model) != kLrtStatusOk) {
      continue;
    }
    soc_models.push_back(std::string(model));
  }
  return soc_models;
}

}  // namespace

CompilerPlugin::ResultT CompilerPlugin::LoadPlugin(
    const absl::string_view lib_path) {
  LITE_RT_LOG(LRT_INFO, "Loading plugin at: %s", lib_path.data());
  CompilerPlugin plugin;

  if (OpenLib(lib_path, &plugin.lib_handle_) != kLrtStatusOk) {
    LITE_RT_LOG(LRT_WARNING, "Failed to load plugin at: %s", lib_path.data());
    return ResultT::FromStatus(kLrtStatusDynamicLoadErr);
  }

  if (ResolvePluginApi(plugin.lib_handle_, plugin.plugin_api_) !=
      kLrtStatusOk) {
    LITE_RT_LOG(LRT_WARNING, "Failed to resolve plugin api at: %s",
                lib_path.data());
    return ResultT::FromStatus(kLrtStatusDynamicLoadErr);
  }

  if (plugin.plugin_api_.init(&plugin.plugin_handle_) != kLrtStatusOk) {
    LITE_RT_LOG(LRT_WARNING, "Failed to initialize plugin at: %s",
                lib_path.data());
    if (CloseLib(plugin.lib_handle_) != kLrtStatusOk) {
      LITE_RT_LOG(LRT_WARNING, "Failed to close loaded library at: %s",
                  lib_path.data());
    }
    return ResultT::FromStatus(kLrtStatusDynamicLoadErr);
  }

  // This should never change throughout the lifetime of the compiler
  // plugin so save to avoid recalling.
  plugin.soc_models_ = GetSocModels(plugin.plugin_api_, plugin.plugin_handle_);

  return ResultT::TakeValue(std::move(plugin));
}

CompilerPlugin::ResultVecT CompilerPlugin::LoadPlugins(
    absl::Span<const absl::string_view> lib_search_paths) {
  std::vector<std::string> plugin_lib_paths;
  for (auto search_path : lib_search_paths) {
    LRT_RETURN_RESULT_IF_NOT_OK(
        FindLrtSharedLibs(search_path, plugin_lib_paths), VecT);
  }

  VecT loaded_plugins;
  loaded_plugins.reserve(lib_search_paths.size());

  for (const auto& lib_path : plugin_lib_paths) {
    LITE_RT_LOG(LRT_INFO, "Loading plugin at: %s", lib_path.c_str());
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
    plugin_api_.destroy(plugin_handle_);
  }
  if (lib_handle_ != nullptr) {
    if (kLrtStatusOk != CloseLib(lib_handle_)) {
      LITE_RT_LOG(LRT_WARNING, "%s", "Failed to close shared library\n");
    }
  }
}

LrtResult<std::vector<LrtOp>> CompilerPlugin::PartitionModel(
    const LrtModelT& model) {
  LrtOpListT ops;
  // TODO: Use const where appropriate in the C compiler plugin api.
  LrtModel c_model = const_cast<LrtModel>(&model);
  LRT_RETURN_RESULT_IF_NOT_OK(
      plugin_api_.partition_model(plugin_handle_, c_model, &ops),
      std::vector<LrtOp>);

  return LrtResult<std::vector<LrtOp>>::TakeValue(std::move(ops.ops));
}

LrtStatus CompilerPlugin::Compile(const absl::string_view soc_model,
                                  const std::vector<LrtSubgraph>& partitions,
                                  std::ostream& byte_code_out,
                                  std::vector<std::string>& call_info_out) {
  CompiledResult result = MakeResult();

  // Compile given partitions into result.
  // TODO: Use const where appropriate in the C compiler plugin api.
  LrtSubgraphArray partitions_arr =
      const_cast<LrtSubgraphArray>(partitions.data());
  LRT_RETURN_STATUS_IF_NOT_OK(
      plugin_api_.compile(plugin_handle_, soc_model.data(), partitions_arr,
                          partitions.size(), &result.compiled_result_handle_));

  // Parse call info from the result.
  {
    LRT_ASSIGN_OR_RETURN_STATUS(auto num_call, result.NumCalls());
    LRT_ENSURE(num_call == partitions.size(), kLrtStatusErrorRuntimeFailure,
               "Plugin didn't return call info for each partition compiled.\n");
    for (int i = 0; i < num_call; ++i) {
      LRT_ASSIGN_OR_RETURN_STATUS(call_info_out.emplace_back(),
                                  result.CallInfo(i));
    }
  }

  // Parse byte code from result.
  {
    LRT_ASSIGN_OR_RETURN_STATUS(const CompiledResult::BytesT byte_code,
                                result.ByteCode());
    LITE_RT_LOG(LRT_INFO, "Compiled %d partitions in %lu bytes",
                partitions.size(), byte_code.size);
    byte_code_out.write(byte_code.data, byte_code.size);
  }

  return kLrtStatusOk;
}

}  // namespace lrt::internal
