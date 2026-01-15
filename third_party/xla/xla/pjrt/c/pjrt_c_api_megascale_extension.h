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

#ifndef XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to Megascale.

#define PJRT_API_MEGASCALE_EXTENSION_VERSION 1

typedef struct PJRT_Megascale_AoTConfig PJRT_Megascale_AoTConfig;

struct PJRT_Megascale_GetAoTConfig_Args {
  size_t struct_size;
  const PJRT_TopologyDescription* topology;
  int32_t num_slices;

  // Output: serialized config.
  const char* serialized_config;
  size_t serialized_config_size;

  // Output: num_devices_per_slice.
  const int32_t* slice_ids;
  const int32_t* num_devices;
  size_t num_entries;

  PJRT_Megascale_AoTConfig* aot_config;
  void (*aot_config_deleter)(PJRT_Megascale_AoTConfig* config);
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_GetAoTConfig_Args, aot_config_deleter);

typedef PJRT_Error* PJRT_Megascale_GetAoTConfig(
    PJRT_Megascale_GetAoTConfig_Args* args);

typedef struct PJRT_Megascale_Extension {
  PJRT_Extension_Base base;
  PJRT_Megascale_GetAoTConfig* get_aot_config;
} PJRT_Megascale_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_Extension, get_aot_config);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_
