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
#ifndef TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_H_
#define TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_H_
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/c/c_api_macros.h"
#include "tensorflow/c/tf_status.h"

// C API for Pluggable Profiler. The API is under active development and
// eventually should allow registering a profiler with TensorFlow.
//
// Conventions:
//   * Struct prefix indicates whether struct fields should be filled by the
//   plug-in or core TensorFlow implementation:
//     * TF_: Set/filled by core, unless marked otherwise.
//     * TP_: Set/filled by plug-in, unless marked otherwise.
//     * This prefix rule only applies to structures. Enumerations and methods
//     are all prefixed with TP_.
//   * Structs begin with two fields:
//     * size_t struct_size: Stores the unpadded size of the struct.
//     * void* ext: A reserved field that may be populated by a plugin in TP_*
//     structs or potential future extension points in TF_ structs. Must be set
//     to zero by default if it unused.
//  * We use struct_size for version checking by both core and plug-in.
//    * It is exempt from the TF/TP rule above and must be set both by core and
//    plug-in.
//    * It can be checked programmatically to determine which struct fields are
//    available in the structure.
//  * When a member is added to a struct, the struct size definition must be
//  updated to use the new last member of the struct.
//
// Example usage:
//   /* Sample TensorFlow code below, exact implementation might differ. */
//   // Version checking uses `struct_size`. It is exempt from the `TF/TP` rule
//   // above and should be set both by core and the plugin."
//
//   /* Plugin code below */
//   void profiler_start(const TP_Profiler* profiler, TF_Status* status) {
//     /* Enable profiler */
//     ...
//   }
//
//  void profiler_stop(const TP_Profiler* profiler, TF_Status* status) {
//    /* Disable Profiler */
//    ...
//  }
//
//  void profiler_collect_data_xspace(const TP_Profiler* profiler, uint8_t*
//  buffer, size_t* size_in_bytes, TF_Status* status) {
//    /* Plugin generates Xspace based on collected profiler data. */
//    Xspace xspace = get_my_xspace();
//    size_t buffer_size_in_bytes = *size_in_bytes;
//    *size_in_bytes = xspace.ByteSizeLong(); /* get the size of Xspace */
//    if (buffer == nullptr) {
//      /* TensorFlow will first get the size of Xspace, then allocate the big
//         enough buffer and pass it to the plugin for retrieving Xspace. */
//      return;
//    }
//    bool success = xspace.SerializeToArray(buffer, buffer_size_in_bytes);
//  }
//
// void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status*
// status) {
//   *params = { TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE };
//   params->profiler->struct_size = TP_PROFILER_STRUCT_SIZE;
//   params->profiler_fns->struct_size = TP_PROFILER_FNS_STRUCT_SIZE;
//
//   params->profiler->type = "MyDeviceType";
//
//   params->profiler_fns->start =  profiler_start;
//   params->profiler_fns->stop = profiler_stop;
//   params->profiler_fns->collect_data_xspace = profiler_collect_data_xspace;
//   params->destroy_profiler = profiler_destroy_profiler;
//   params->destroy_profiler_fns = profiler_destroy_profiler_fns;
// }

#define TP_MAJOR 0
#define TP_MINOR 0
#define TP_PATCH 1

#ifdef __cplusplus
extern "C" {
#endif

// --------------------------------------------------------------------------
// TP_Profiler holds a pointer to device type filed by the plug-in.
typedef struct TP_Profiler {
  size_t struct_size;
  void* ext;  // free-form data set by plugin.
  const char* device_type;

  // The struct size must be updated when adding new members.
#define TP_PROFILER_STRUCT_SIZE TF_OFFSET_OF_END(TP_Profiler, device_type)
} TP_Profiler;

// --------------------------------------------------------------------------
// TP_ProfilerFns holds the profiler interface function pointers filled by the
// plug-in.
typedef struct TP_ProfilerFns {
  size_t struct_size;

  void* ext;  // reserved for future use.
  // Starts profiling.
  void (*start)(const TP_Profiler* profiler, TF_Status* status);
  // Stops profiling.
  void (*stop)(const TP_Profiler* profiler, TF_Status* status);

  // Saves collected profile data into XSpace and serializes it to the buffer.
  // - If `buffer` is null, returns the required buffer size in `size_in_bytes`.
  // - If `buffer` is not null and `size_in_bytes` is the required buffer size,
  //   `buffer` is populated with profile data in serialized XSpace format.
  //
  // Only the first call with a non-null `buffer` following successful calls to
  // start and stop might return data. Subsequent calls might return empty data
  // unless start and stop are successfully called again.
  void (*collect_data_xspace)(const TP_Profiler* profiler, uint8_t* buffer,
                              size_t* size_in_bytes, TF_Status* status);

  // The struct size must be updated when adding new members.
#define TP_PROFILER_FNS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TP_ProfilerFns, collect_data_xspace)
} TP_ProfilerFns;

// TF_ProfilerRegistrationParams holds the pointers to TP_Profiler and
// TP_ProfilerFns, the memory of TP_Profiler and TP_ProfilerFns is owned by Core
// TensorFlow and populated by the plug-in.
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

  // The struct size must be updated when adding new members.
#define TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE \
  TF_OFFSET_OF_END(TF_ProfilerRegistrationParams, destroy_profiler_fns)
} TF_ProfilerRegistrationParams;

// TF_InitProfiler to do profiler registration.
// Plug-in should implement TF_InitProfiler to register the profiler.
void TF_InitProfiler(TF_ProfilerRegistrationParams* params, TF_Status* status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_C_EXPERIMENTAL_PLUGGABLE_PROFILER_PLUGGABLE_PROFILER_H_
