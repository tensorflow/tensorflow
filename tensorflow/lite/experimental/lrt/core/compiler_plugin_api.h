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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_API_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_API_H_

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/vendors/c/lite_rt_compiler_plugin.h"

// Wrapper for dynamically loaded LrtCompilerPlugin library. See
// "lrt_compiler_plugin.h".

namespace lrt::internal {

//
// Api Interface
//

typedef const char* (*LrtPluginApiSocManufacturer)();

typedef LrtStatus (*LrtPluginApiInit)(LrtCompilerPlugin*);

typedef void (*LrtPluginApiDestroy)(LrtCompilerPlugin);

typedef lrt_param_index_t (*LrtPluginApiNumSupportedModels)(LrtCompilerPlugin);

typedef LrtStatus (*LrtPluginApiGetSupportedSocModel)(
    LrtCompilerPlugin, lrt_param_index_t soc_model_idx,
    const char** soc_moel_idx);

typedef LrtStatus (*LrtPluginApiPartitionModel)(LrtCompilerPlugin,
                                                LrtModel model,
                                                LrtOpList selected_ops);

typedef LrtStatus (*LrtPluginApiCompile)(LrtCompilerPlugin,
                                         LrtSubgraphArray partitions,
                                         lrt_param_index_t num_partitions,
                                         LrtCompiledResult* compiled_result);

typedef void (*LrtCompiledResultApiDestroy)(LrtCompiledResult);

typedef LrtStatus (*LrtCompiledResultApiGetByteCode)(LrtCompiledResult,
                                                     const void** byte_code,
                                                     size_t* byte_code_size);

typedef LrtStatus (*LrtCompiledResultApiGetCallInfo)(LrtCompiledResult,
                                                     lrt_param_index_t call_idx,
                                                     const void** call_info,
                                                     size_t* call_info_size);

typedef LrtStatus (*LrtCompiledResultApiGetNumCalls)(
    LrtCompiledResult, lrt_param_index_t* num_calls);

//
// Load Plugin Library Into Api Container
//

// Wraps all resolved functions from api interface.
struct LrtPluginApi {
  LrtPluginApiInit init = nullptr;
  LrtPluginApiDestroy destroy = nullptr;

  LrtPluginApiSocManufacturer soc_manufacturer = nullptr;
  LrtPluginApiNumSupportedModels num_supported_models = nullptr;
  LrtPluginApiGetSupportedSocModel get_supported_soc_model = nullptr;

  LrtPluginApiPartitionModel partition_model = nullptr;
  LrtPluginApiCompile compile = nullptr;

  LrtCompiledResultApiDestroy compiled_result_destroy = nullptr;
  LrtCompiledResultApiGetByteCode compiled_result_get_byte_code = nullptr;
  LrtCompiledResultApiGetCallInfo compiled_result_get_call_info = nullptr;
  LrtCompiledResultApiGetNumCalls compiled_result_get_num_calls = nullptr;
};

// Resolve all api interface symbols with given loaded shared library handle.
LrtStatus ResolvePluginApi(void* lib_handle, LrtPluginApi& result);

}  // namespace lrt::internal

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_CORE_COMPILER_PLUGIN_API_H_
