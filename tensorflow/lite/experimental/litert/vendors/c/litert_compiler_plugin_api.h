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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_

#include <cstddef>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"

// Wrapper for dynamically loaded LiteRtCompilerPlugin library. See
// "litert_compiler_plugin.h".

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//
// Api Interface
//

typedef const char* (*LiteRtGetCompilerPluginSocManufacturerT)();

typedef LiteRtStatus (*LiteRtCreateCompilerPluginT)(LiteRtCompilerPlugin*);

typedef void (*LiteRtDestroyCompilerPluginT)(LiteRtCompilerPlugin);

typedef LiteRtStatus (*LiteRtGetNumCompilerPluginSupportedSocModelsT)(
    LiteRtCompilerPlugin, LiteRtParamIndex*);

typedef LiteRtStatus (*LiteRtGetCompilerPluginSupportedSocModelT)(
    LiteRtCompilerPlugin, LiteRtParamIndex soc_model_idx,
    const char** soc_moel_idx);

typedef LiteRtStatus (*LiteRtCompilerPluginPartitionModelT)(
    LiteRtCompilerPlugin, LiteRtModel model, LiteRtOpList selected_ops);

typedef LiteRtStatus (*LiteRtCompilerPluginCompileT)(
    LiteRtCompilerPlugin, const char* soc_model, LiteRtSubgraphArray partitions,
    LiteRtParamIndex num_partitions, LiteRtCompiledResult* compiled_result);

typedef void (*LiteRtDestroyCompiledResultT)(LiteRtCompiledResult);

typedef LiteRtStatus (*LiteRtGetCompiledResultByteCodeT)(
    LiteRtCompiledResult, const void** byte_code, size_t* byte_code_size);

typedef LiteRtStatus (*LiteRtGetCompiledResultCallInfoT)(
    LiteRtCompiledResult, LiteRtParamIndex call_idx, const void** call_info,
    size_t* call_info_size);

typedef LiteRtStatus (*LiteRtGetNumCompiledResultCallsT)(
    LiteRtCompiledResult, LiteRtParamIndex* num_calls);

//
// Function Pointer Container
//

// Wraps all resolved functions from api interface.
struct LiteRtCompilerPluginApi {
  LiteRtCreateCompilerPluginT create_compiler_plugin = nullptr;
  LiteRtDestroyCompilerPluginT destroy_compiler_plugin = nullptr;

  LiteRtGetCompilerPluginSocManufacturerT get_compiler_plugin_soc_manufacturer =
      nullptr;
  LiteRtGetNumCompilerPluginSupportedSocModelsT
      get_num_compiler_plugin_supported_models = nullptr;
  LiteRtGetCompilerPluginSupportedSocModelT
      get_compiler_plugin_supported_soc_model = nullptr;

  LiteRtCompilerPluginPartitionModelT compiler_plugin_partition_model = nullptr;
  LiteRtCompilerPluginCompileT compiler_plugin_compile = nullptr;

  LiteRtDestroyCompiledResultT destroy_compiled_result = nullptr;
  LiteRtGetCompiledResultByteCodeT get_compiled_result_byte_code = nullptr;
  LiteRtGetCompiledResultCallInfoT get_compiled_result_call_info = nullptr;
  LiteRtGetNumCompiledResultCallsT get_compiled_result_num_calls = nullptr;
};

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_API_H_
