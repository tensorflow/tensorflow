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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_

#include <cstddef>

#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/litert_compiler_plugin.h"

// Wrapper for dynamically loaded LiteRtCompilerPlugin library. See
// "litert_compiler_plugin.h".

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// Api Interface
//

typedef const char* (*LiteRtPluginApiSocManufacturer)();

typedef LiteRtStatus (*LiteRtPluginApiInit)(LiteRtCompilerPlugin*);

typedef void (*LiteRtPluginApiDestroy)(LiteRtCompilerPlugin);

typedef LiteRtParamIndex (*LiteRtPluginApiNumSupportedModels)(
    LiteRtCompilerPlugin);

typedef LiteRtStatus (*LiteRtPluginApiGetSupportedSocModel)(
    LiteRtCompilerPlugin, LiteRtParamIndex soc_model_idx,
    const char** soc_moel_idx);

typedef LiteRtStatus (*LiteRtPluginApiPartitionModel)(
    LiteRtCompilerPlugin, LiteRtModel model, LiteRtOpList selected_ops);

typedef LiteRtStatus (*LiteRtPluginApiCompile)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtSubgraphArray partitions,
    LiteRtParamIndex num_partitions, LiteRtCompiledResult* compiled_result);

typedef void (*LiteRtCompiledResultApiDestroy)(LiteRtCompiledResult);

typedef LiteRtStatus (*LiteRtCompiledResultApiGetByteCode)(
    LiteRtCompiledResult, const void** byte_code, size_t* byte_code_size);

typedef LiteRtStatus (*LiteRtCompiledResultApiGetCallInfo)(
    LiteRtCompiledResult, LiteRtParamIndex call_idx, const void** call_info,
    size_t* call_info_size);

typedef LiteRtStatus (*LiteRtCompiledResultApiGetNumCalls)(
    LiteRtCompiledResult, LiteRtParamIndex* num_calls);

//
// Function Pointer Container
//

// Wraps all resolved functions from api interface.
struct LiteRtCompilerPluginApi {
  LiteRtPluginApiInit init = nullptr;
  LiteRtPluginApiDestroy destroy = nullptr;

  LiteRtPluginApiSocManufacturer soc_manufacturer = nullptr;
  LiteRtPluginApiNumSupportedModels num_supported_models = nullptr;
  LiteRtPluginApiGetSupportedSocModel get_supported_soc_model = nullptr;

  LiteRtPluginApiPartitionModel partition_model = nullptr;
  LiteRtPluginApiCompile compile = nullptr;

  LiteRtCompiledResultApiDestroy compiled_result_destroy = nullptr;
  LiteRtCompiledResultApiGetByteCode compiled_result_get_byte_code = nullptr;
  LiteRtCompiledResultApiGetCallInfo compiled_result_get_call_info = nullptr;
  LiteRtCompiledResultApiGetNumCalls compiled_result_get_num_calls = nullptr;
};

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
