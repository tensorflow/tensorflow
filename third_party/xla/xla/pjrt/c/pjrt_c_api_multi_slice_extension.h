/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to multi-slice.

#define PJRT_API_MULTI_SLICE_EXTENSION_VERSION 1

typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;

typedef struct PJRT_MultiSlice_NumDevicesPerSlice
    PJRT_MultiSlice_NumDevicesPerSlice;

typedef struct PJRT_MultiSlice_SerializedConfig
    PJRT_MultiSlice_SerializedConfig;

struct PJRT_MultiSlice_Config_Destroy_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Config_Destroy_Args, config);

typedef PJRT_Error* PJRT_MultiSlice_Config_Destroy(
    PJRT_MultiSlice_Config_Destroy_Args* args);

struct PJRT_MultiSlice_Config_NumSlices_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  int32_t num_slices;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Config_NumSlices_Args, num_slices);

typedef PJRT_Error* PJRT_MultiSlice_Config_NumSlices(
    PJRT_MultiSlice_Config_NumSlices_Args* args);

struct PJRT_MultiSlice_Config_SliceId_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  int32_t slice_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Config_SliceId_Args, slice_id);

typedef PJRT_Error* PJRT_MultiSlice_Config_SliceId(
    PJRT_MultiSlice_Config_SliceId_Args* args);

struct PJRT_MultiSlice_Config_NumDevicesPerSlice_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  size_t num_devices_per_slice_map;                           // out
  const int32_t* slice_ids;                                   // out
  const int32_t* num_devices;                                 // out
  PJRT_MultiSlice_NumDevicesPerSlice* devices_per_slice_map;  // out
  void (*devices_per_slice_map_deleter)(
      PJRT_MultiSlice_NumDevicesPerSlice* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Config_NumDevicesPerSlice_Args,
                          devices_per_slice_map_deleter);

typedef PJRT_Error* PJRT_MultiSlice_Config_NumDevicesPerSlice(
    PJRT_MultiSlice_Config_NumDevicesPerSlice_Args* args);

struct PJRT_MultiSlice_Config_Serialize_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  const char* serialized;                               // out
  size_t size;                                          // out
  PJRT_MultiSlice_SerializedConfig* serialized_config;  // out
  void (*serialized_config_deleter)(
      PJRT_MultiSlice_SerializedConfig* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Config_Serialize_Args,
                          serialized_config_deleter);

typedef PJRT_Error* PJRT_MultiSlice_Config_Serialize(
    PJRT_MultiSlice_Config_Serialize_Args* args);

typedef struct PJRT_MultiSlice_Extension {
  PJRT_Extension_Base base;
  PJRT_MultiSlice_Config_Destroy* config_destroy;
  PJRT_MultiSlice_Config_NumSlices* config_num_slices;
  PJRT_MultiSlice_Config_SliceId* config_slice_id;
  PJRT_MultiSlice_Config_NumDevicesPerSlice* config_num_devices_per_slice;
  PJRT_MultiSlice_Config_Serialize* config_serialize;
} PJRT_MultiSlice_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MultiSlice_Extension, config_serialize);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MULTI_SLICE_EXTENSION_H_
