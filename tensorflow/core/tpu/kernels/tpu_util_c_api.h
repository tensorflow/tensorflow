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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_

#include "tensorflow/core/tpu/kernels/tpu_mesh_state_c_api.h"
#include "tensorflow/core/tpu/libtftpu.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"

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

// Checks if whether a TPU compilation is enabled.
TFTPU_CAPI_EXPORT bool TpuCompile_IsTpuCompilationEnabled();

// XLA compilation cannot be cancelled. To avoid hanging the TF worker will exit
// when cancellation is requested for an XLA compile op. Some tests require this
// behavior to be disabled, and we test for this condition with the following
// flag function.
TFTPU_CAPI_EXPORT bool TpuCompile_ShouldTpuCompileOpIgnoreCancellation();

// Returns the number of available TPU core count.
TFTPU_CAPI_EXPORT int TpuTopology_AvailableCoreCount(
    const XLA_TpuMeshState* mesh_state, TpuCoreTypeEnum tpu_core_type);

// Recycle unused service port.
TFTPU_CAPI_EXPORT void TpuNetUtil_RecycleUnusedPort(int port);

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

}  // extern "C"

struct TfTpu_UtilApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_IsTpuCompilationEnabled);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_ShouldTpuCompileOpIgnoreCancellation);
  TFTPU_ADD_FN_IN_STRUCT(TpuTopology_AvailableCoreCount);
  TFTPU_ADD_FN_IN_STRUCT(TpuNetUtil_RecycleUnusedPort);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_DestroyCompilationCacheKey);
  TFTPU_ADD_FN_IN_STRUCT(TpuCompile_CreateGuaranteedConstFingerprint);
};

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_C_API_H_
