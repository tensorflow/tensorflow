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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_LITE_RT_COMPILER_PLUGIN_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_LITE_RT_COMPILER_PLUGIN_H_

#include <stddef.h>

#include "tensorflow/compiler/mlir/lite/experimental/lrt/api/lite_rt_model_api.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/lite_rt_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtCompilerPlugin);

// Artifact produced from compiling a selected partition of ops.
LITE_RT_DEFINE_HANDLE(LrtCompiledPartition);

//
// Plugin
//

LrtStatus PluginInit(LrtCompilerPlugin* compiler_plugin);

void PluginDestroy(LrtCompilerPlugin compiler_plugin);

// Select desired ops for compilation. This will be called only once
// during the plugin application flow, all ops should be selected during this
// call.
LrtStatus PluginPartitionModel(LrtCompilerPlugin compiler_plugin,
                               LrtModel model, LrtOpList selected_ops);

// Prepare artifact to pass to the runtime for given partition. The given
// subgraph is a single valid sub-DAG within the ops selected in partition step.
LrtStatus PluginCompilePartition(LrtCompilerPlugin compiler_plugin,
                                 LrtSubgraph partition,
                                 LrtCompiledPartition* compiled_partition);

// Name associated with the plugin.
const char* PluginGetNamespace(LrtCompilerPlugin compiler_plugin);

//
// Compiled Partition
//

void PluginCompiledPartitionDestroy(LrtCompiledPartition compiled_partition);

// Get serialized artifact to pass to the runtime for specific partition.
LrtStatus PluginCompiledPartitionGetByteCode(
    LrtCompiledPartition compiled_partition, const void** byte_code,
    size_t* byte_code_size);

// Get a name for specific partition.
LrtStatus PluginCompiledPartitionGetName(
    LrtCompiledPartition compiled_partition, const char** partition_name);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_COMPILER_PLUGIN_LITE_RT_COMPILER_PLUGIN_H_
