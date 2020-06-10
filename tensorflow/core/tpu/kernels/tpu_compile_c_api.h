/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_

#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"

enum TpuCoreTypeEnum {
  kTensorCore,
  kEmbeddingV1,
  kEmbeddingV2,
};

typedef struct XLA_TpuProgram XLA_TpuProgram;

// Property for creating compilation cache key.
struct CompilationCacheKeyProperty {
  const char* config_prefix;
  const char* shapes_prefix;
  const char* function_name;
  const char* mlir_module;
  const int32_t* device_ids;
  size_t device_ids_size;
  int32_t guaranteed_constants_size;
  uint64_t function_library_fingerprint;
  int32_t num_cores_per_replica;
  int32_t num_replicas;
  const XLA_TpuMeshState* mesh_state;
};

extern "C" {

// Creates a new TPU program.
XLA_TpuProgram* TpuProgram_New();

// Destroys the `tpu_program`.
void TpuProgram_Free(XLA_TpuProgram* tpu_program);


// Unloads and destroys the `tpu_program`. Once the TPU program is unloaded and
// destroyed, it is in an unusable state.
void TpuProgram_UnloadAndDestroy(XLA_TpuProgram* tpu_program,
                                 SE_Status* status);

// Gets TPU program size in bytes from the `tpu_program`.
int64_t TpuProgram_GetProgramSize(const XLA_TpuProgram* tpu_program);

// Logs the summary of current memory state snapshot of the `tpu_program`.
bool TpuProgram_LogProgramMemorySummary(const XLA_TpuProgram* tpu_program);

// Gets TPU program executable info from the `tpu_program`.
void TpuProgram_GetExecutableInfo(const XLA_TpuProgram* tpu_program,
                                  TpuSerializedProto* executable_info);

// Gets host transfer info proto.
void TpuProgram_GetHostTransferInfo(
    const XLA_TpuProgram* tpu_program,
    TpuSerializedProto* host_transfer_info);

// Gets HLO metadata proto.
void TpuProgram_GetHloMetadata(const XLA_TpuProgram* tpu_program,
                               TpuSerializedProto* hlo_metadata);

// Returns the number of available TPU core count.
int TpuTopology_AvailableCoreCount(const XLA_TpuMeshState* mesh_state,
                                   TpuCoreTypeEnum tpu_core_type);

// Creates a unique compilation cache `key` used for `put` and `get` operations.
// Returned buffer is heap-allocated and must be owned.
const char* TpuCompile_CreateCompilationCacheKey(
    CompilationCacheKeyProperty property);

// Creates a guaranteed const fingerprint. Guarantee const is normally used in
// TPU inference to avoid re-copying unchanged variables onto the TPU device.
// It promises the value is identical for every execution in the same session
// even if the actual value changes in later executions.
uint64_t TpuCompile_CreateGuaranteedConstFingerprint(uint64_t fingerprint,
                                                     const char* data,
                                                     size_t size);

// Executes the computations using XLA TPU compiler and returns TPU programs
// ready for execution.
void TpuCompile_CompileAheadOfTime(
    TpuSerializedProto aot_compilation_request,
    XLA_TpuProgram** tpu_programs[],
    size_t* count, SE_Status* status);

// Builds `DeviceAssignment` from `TpuCompileMetadata` serialized proto.
void TpuCompile_BuildXLADeviceAssignment(
    TpuSerializedProto serialized_tpu_compile_metadata,
    const XLA_TpuMeshState* mesh_state,
    TpuSerializedProto* serialized_device_assignment, SE_Status* status);

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
