/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_PROFILER_PROFILER_H_
#define TENSORFLOW_C_EXPERIMENTAL_PROFILER_PROFILER_H_
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"

#define TP_MAJOR 0
#define TP_MINOR 0
#define TP_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TP_Profiler {
  size_t struct_size;
  void* ext;  // free-form data set by plugin.
  const char* type;
} TP_Profiler;

#define TP_PROFILER_STRUCT_SIZE TF_OFFSET_OF_END(TP_Profiler, type)

typedef struct TP_ProfilerFns {
  size_t struct_size;

  void* ext;  // reserved for future use.
  // Starts profiling.
  void (*start)(const TP_Profiler* profiler, TF_Status* status);
  // Stops profiling.
  void (*stop)(const TP_Profiler* profiler, TF_Status* status);

  // Saves collected profile data into XSpace and serializes it to the buffer.
  // If this have been called, subsequent calls might
  // return empty data.
  void (*collect_data_xspace)(const TP_Profiler* profiler, uint8_t* buffer,
                              size_t* size_in_bytes, TF_Status* status);
} TP_ProfilerFns;

#define TP_PROFILER_FNS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_ProfilerFns, collect_data_xspace)

typedef struct TF_ProfilerRegistrationParams {
  size_t struct_size;
  void* ext;  // reserved for future use

  // TensorFlow Profiler C API version.
  int32_t major_version;
  int32_t minor_version;
  int32_t patch_version;

  // [in/out] Memory owned by core but attributes within are populated by the
  // plugin.
  TP_Profiler* profiler;
  // [in/out] Memory owned by core but attributes within are populated by the
  // plugin.
  TP_ProfilerFns* profiler_fns;
  // [out] Pointer to plugin's `TP_Profiler` clean up function.
  // Cleans up fields inside `TP_Profiler` that were allocated
  // by the plugin. `profiler` itself must not be deleted by the plugin.
  void (*destroy_profiler)(TP_Profiler* profiler);
  // [out] Pointer to plugin's `TP_ProfilerFns` clean up function.
  // Cleans up fields inside `TP_ProfilerFns` that were allocated
  // by the plugin. `profiler_fns` itself must not be deleted by the plugin.
  void (*destroy_profiler_fns)(TP_ProfilerFns* profiler_fns);
} TF_ProfilerRegistrationParams;

#define TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TF_ProfilerRegistrationParams, destroy_profiler_fns)

void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_EXPERIMENTAL_PROFILER_PROFILER_H_
