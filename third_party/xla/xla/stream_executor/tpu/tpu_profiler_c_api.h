/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_PROFILER_C_API_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_PROFILER_C_API_H_

#include <stddef.h>

#include <cstdint>

#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/libtftpu.h"

extern "C" {

typedef struct TpuProfiler TpuProfiler;

// Creates a TPU profiler that is ready to start profiling.
TFTPU_CAPI_EXPORT void TpuProfiler_Create(TpuProfiler** tpu_profiler,
                                          TF_Status* status);
// Destroys the given TPU profiler.
TFTPU_CAPI_EXPORT void TpuProfiler_Destroy(TpuProfiler* tpu_profiler);
// Starts profiling if not already started, returns an error otherwise.
TFTPU_CAPI_EXPORT void TpuProfiler_Start(TpuProfiler* tpu_profiler,
                                         TF_Status* status);
// Stops profiling if not already stopped, returns an error otherwise.
TFTPU_CAPI_EXPORT void TpuProfiler_Stop(TpuProfiler* tpu_profiler,
                                        TF_Status* status);
// Serializes profiled data into `buffer` and returns the size of `buffer`. The
// profile data held by the TPU driver will be cleared after retrieval.
//
// Step 1. Query the size of buffer required into `size_in_bytes`.
//
//   size_t size_in_bytes;
//   TpuProfiler_CollectData(profiler, status, nullptr, &size_in_bytes);
//
// Step 2. Retrieve the data into a `buffer` of size `size_in_bytes`.
//         Subsequently,The TPU driver clears its copy of the profile data.
//
//   uint8_t buffer = new uint8_t[size_in_bytes];
//   TpuProfiler_CollectData(profiler, status, buffer, size_in_bytes);
//
// Step 3. Unpack the data into an XSpace.
//
//   tensorflow::profiler::XSpace space;
//   space.ParseFromArray(buffer, size_in_bytes);
//
TFTPU_CAPI_EXPORT void TpuProfiler_CollectData(TpuProfiler* tpu_profiler,
                                               TF_Status* status,
                                               uint8_t* buffer,
                                               size_t* size_in_bytes);

// absl::Status helpers to create TFStatus for Profiler.
TF_Status* TpuStatus_New();
void TpuStatus_Free(TF_Status* status);
const char* TpuStatus_Message(TF_Status* status);
int TpuStatus_Code(TF_Status* status);

struct TfTpu_ProfilerApiFn {
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Create);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Destroy);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Start);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_Stop);
  TFTPU_ADD_FN_IN_STRUCT(TpuProfiler_CollectData);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_New);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Free);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Message);
  TFTPU_ADD_FN_IN_STRUCT(TpuStatus_Code);
};

}  // extern "C"

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_PROFILER_C_API_H_
