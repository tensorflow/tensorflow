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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITERT_DEFINE_HANDLE(LiteRtCompilerPlugin);

// Artifact produced from compiling a selected partition of ops.
LITERT_DEFINE_HANDLE(LiteRtCompiledResult);

//
// Plugin
//

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version);

// Name associated with the manufacturer this plugin relates to (e.g,
// GoogleTensor, Qualcomm).
const char* LiteRtGetCompilerPluginSocManufacturer();

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin);

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin);

// Return the HW supported by this plugin (e.g., GPU, NPU)
LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware);

// Number of SoC models supported by this plugin.
LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models);

// Gets the name of the SoC model at the given index. The memory
// associated with the returned name is owned by the plugin.
LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name);

// Select desired ops for compilation. This will only be called once
// per subgraph, plugins should select all supportable ops.
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops);

// Prepare result to pass to the runtime for given model containing partitioned
// subgraphs. Optionally, handles a SoC model (parameter `soc_model` can be NULL
// to specify a default SoC model).
LiteRtStatus LiteRtCompilerPluginCompile(LiteRtCompilerPlugin compiler_plugin,
                                         const char* soc_model,
                                         LiteRtModel partitions,
                                         LiteRtCompiledResult* compiled_result);

// Set any flags for the compiler do use during compilation. Flag data may be
// released or reused after this function returns. Flags are string key ->
// optional string value pairs. A non-existent value is represented by an empty
// string. Calling this function will unset any previously set flags.
LiteRtStatus LiteRtCompilerPluginSetFlags(LiteRtCompilerPlugin compiler_plugin,
                                          LiteRtParamIndex num_flags,
                                          const char** keys,
                                          const char** values);

//
// Compiled Partition
//

void LiteRtDestroyCompiledResult(LiteRtCompiledResult result);

// Get the buffer for the compiled byte code for the given index.
LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size);

// The number of individual byte code modules.
LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code);

// Get per-op info related to a particular compiled partition as well as the
// index of the respective byte code buffer.
LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx);

// Get the number of calls that will be made to the HAL for this graph.
// This should equal the number of partitions given for compilation which
// is equal to the number of custom ops in the final model.
LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_C_LITERT_COMPILER_PLUGIN_H_
