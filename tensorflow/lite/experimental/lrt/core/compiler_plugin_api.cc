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

#include "tensorflow/lite/experimental/lrt/core/compiler_plugin_api.h"

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/dynamic_loading.h"

namespace lrt::internal {

#define RESOLVE_API_FUNC(ty, name, dest) \
  LRT_RETURN_STATUS_IF_NOT_OK(ResolveLibSymbol<ty>(lib_handle, name, &dest));

LrtStatus ResolvePluginApi(void* lib_handle, LrtPluginApi& result) {
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

}  // namespace lrt::internal
