/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_

#include <stddef.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional and experimental extension.
// This extension allows to retrieve all supported types of memory
// supported by a given device description. This is useful for specifying
// non-default memories in AOT computations (as opposed to the
// physically-present memories associated with a PJRT_Client).

#define PJRT_API_MEMORY_DESCRIPTIONS_EXTENSION_VERSION 1

typedef struct PJRT_MemoryDescription PJRT_MemoryDescription;

struct PJRT_DeviceDescription_MemoryDescriptions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const PJRT_MemoryDescription* const* memory_descriptions;  // out
  size_t num_memory_descriptions;                            // out
  // Index into memory_descriptions. -1 if there's no default:
  size_t default_memory_index;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_MemoryDescriptions_Args,
                          default_memory_index);

// Returns all memory descriptions attached to this device.
// The memories are in no particular order.
typedef PJRT_Error* PJRT_DeviceDescription_MemoryDescriptions(
    PJRT_DeviceDescription_MemoryDescriptions_Args* args);

struct PJRT_MemoryDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_MemoryDescription* memory_description;
  // `kind` has same lifetime as `memory_description`.
  const char* kind;  // out
  size_t kind_size;  // out
  int kind_id;       // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MemoryDescription_Kind_Args, kind_id);

// Returns the kind of a given memory space description. This is a
// platform-dependent string and numeric ID that uniquely identifies the kind of
// memory space among those possible on this platform.
typedef PJRT_Error* PJRT_MemoryDescription_Kind(
    PJRT_MemoryDescription_Kind_Args* args);

typedef struct PJRT_MemoryDescriptions_Extension {
  PJRT_Extension_Base base;
  PJRT_DeviceDescription_MemoryDescriptions*
      PJRT_DeviceDescription_MemoryDescriptions;
  PJRT_MemoryDescription_Kind* PJRT_MemoryDescription_Kind;
} PJRT_MemoryDescriptions_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_MemoryDescriptions_Extension,
                          PJRT_MemoryDescription_Kind);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MEMORY_DESCRIPTIONS_EXTENSION_H_
