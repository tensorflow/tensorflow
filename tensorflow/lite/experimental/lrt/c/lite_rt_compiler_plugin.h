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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMPILER_PLUGIN_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMPILER_PLUGIN_H_

#include <stddef.h>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtCompilerPlugin);

// Artifact produced from compiling a selected partition of ops.
LITE_RT_DEFINE_HANDLE(LrtCompiledResult);

//
// Plugin
//

LrtStatus LrtPluginInit(LrtCompilerPlugin* compiler_plugin);

void LrtPluginDestroy(LrtCompilerPlugin compiler_plugin);

// Name associated with the manufacturer this plugin relates to (e.g, Pixel,
// Qualcomm).
const char* LrtPluginSocManufacturer();

// Number of SoC models supported by this plugin.
lrt_param_index_t LrtPluginNumSupportedSocModels(
    LrtCompilerPlugin compiler_plugin);

// Gets the name of the SoC model at the given index. The memory
// associated with the returned name is owned by the plugin.
LrtStatus LrtPluginGetSupportedSocModel(LrtCompilerPlugin compiler_plugin,
                                        lrt_param_index_t soc_model_idx,
                                        const char** soc_model_name);

// Select desired ops for compilation. This will be called only once
// during the plugin application flow, all ops should be selected during this
// call.
LrtStatus LrtPluginPartitionModel(LrtCompilerPlugin compiler_plugin,
                                  LrtModel model, LrtOpList selected_ops);

// Prepare result to pass to the runtime for given partition and, optionally,
// for a given SoC model (parameter `soc_model` can be NULL). The given
// subgraphs are valid sub-DAG within the ops selected in partition step.
LrtStatus LrtPluginCompile(LrtCompilerPlugin compiler_plugin,
                           const char* soc_model, LrtSubgraphArray partitions,
                           lrt_param_index_t num_partitions,
                           LrtCompiledResult* compiled_result);

//
// Compiled Partition
//

void LrtCompiledResultDestroy(LrtCompiledResult result);

// Get serialized result to compiled modules available to all custom ops.
// This could be one module with multiple entry points or multiple modules
// concat together.
LrtStatus LrtCompiledResultGetByteCode(LrtCompiledResult compiled_result,
                                       const void** byte_code,
                                       size_t* byte_code_size);

// Get info to embed in a particular custom op. This could be  any opaque data
// parsed in the custom op.
LrtStatus LrtCompiledResultGetCallInfo(LrtCompiledResult compiled_result,
                                       lrt_param_index_t call_idx,
                                       const void** call_info,
                                       size_t* call_info_size);

// Get the number of calls that will be made to the HAL for this graph.
// This should equal the number of partitions given for compilation which
// is equal to the number of custom ops in the final model.
LrtStatus LrtCompiledResultGetNumCalls(LrtCompiledResult compiled_result,
                                       lrt_param_index_t* num_calls);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMPILER_PLUGIN_H_
