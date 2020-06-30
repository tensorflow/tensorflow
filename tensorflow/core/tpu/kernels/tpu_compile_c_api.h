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
#include "tensorflow/core/tpu/kernels/tpu_program_c_api.h"
#include "tensorflow/core/tpu/kernels/tpu_util_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

enum TpuCoreTypeEnum {
  kTensorCore,
  kEmbeddingV1,
  kEmbeddingV2,
};

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

// Compilation cache key result returning both the key and a more verbose debug
// version.
struct CompilationCacheKeyResult {
  const char* key;
  const char* debug_string;
};

extern "C" {

// Returns the number of available TPU core count.
TFTPU_CAPI_EXPORT int TpuTopology_AvailableCoreCount(
    const XLA_TpuMeshState* mesh_state, TpuCoreTypeEnum tpu_core_type);

// Creates a unique compilation cache `key` used for `put` and `get` operations.
// Returned buffers are heap-allocated and must be owned.
TFTPU_CAPI_EXPORT CompilationCacheKeyResult
TpuCompile_CreateCompilationCacheKey(CompilationCacheKeyProperty property);

// Destroys the CompilationCacheKeyResult returned by calling the
// `TpuCompile_CreateCompilationCacheKey` API.
TFTPU_CAPI_EXPORT void TpuCompile_DestroyCompilationCacheKey(
    CompilationCacheKeyResult result);

// Creates a guaranteed const fingerprint. Guarantee const is normally used in
// TPU inference to avoid re-copying unchanged variables onto the TPU device.
// It promises the value is identical for every execution in the same session
// even if the actual value changes in later executions.
TFTPU_CAPI_EXPORT uint64_t TpuCompile_CreateGuaranteedConstFingerprint(
    uint64_t fingerprint, const char* data, size_t size);

// Executes the computations using XLA TPU compiler and returns TPU programs
// ready for execution.
TFTPU_CAPI_EXPORT void TpuCompile_CompileAheadOfTime(
    TpuSerializedProto aot_compilation_request, XLA_TpuProgram** tpu_programs[],
    size_t* count, SE_Status* status);

// Builds `DeviceAssignment` from `TpuCompileMetadata` serialized proto.
TFTPU_CAPI_EXPORT void TpuCompile_BuildXLADeviceAssignment(
    TpuSerializedProto serialized_tpu_compile_metadata,
    const XLA_TpuMeshState* mesh_state,
    TpuSerializedProto* serialized_device_assignment, SE_Status* status);

struct TfTpu_CompileApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_AvailableCoreCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_DestroyCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateGuaranteedConstFingerprint);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CompileAheadOfTime);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_BuildXLADeviceAssignment);
};

}  // extern "C"

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_COMPILE_C_API_H_
