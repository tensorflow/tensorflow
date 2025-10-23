/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_EXTENSION_H_

#include <cstddef>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_HOST_ALLOCATOR_EXTENSION_VERSION 0
// ---------------------------------- Methods ----------------------------------

struct PJRT_HostAllocator_GetPreferredAlignment_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_Client* client;
  size_t preferred_alignment;  // out.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostAllocator_GetPreferredAlignment_Args,
                          preferred_alignment);

// Gets the alignment of the host allocator.
typedef PJRT_Error* (*PJRT_HostAllocator_GetPreferredAlignment)(
    PJRT_HostAllocator_GetPreferredAlignment_Args* args);

struct PJRT_HostAllocator_Allocate_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_Client* client;
  size_t size;
  size_t alignment;
  void* ptr;  // out.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostAllocator_Allocate_Args, ptr);

// Allocate memory from the host allocator.
typedef PJRT_Error* (*PJRT_HostAllocator_Allocate)(
    PJRT_HostAllocator_Allocate_Args* args);

struct PJRT_HostAllocator_Free_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_Client* client;
  void* ptr;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostAllocator_Free_Args, ptr);

// Free memory allocated by the host allocator.
typedef PJRT_Error* (*PJRT_HostAllocator_Free)(
    PJRT_HostAllocator_Free_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_HostAllocator_Extension {
  PJRT_Extension_Base base;
  PJRT_HostAllocator_GetPreferredAlignment get_preferred_alignment;
  PJRT_HostAllocator_Allocate allocate;
  PJRT_HostAllocator_Free free;
} PJRT_HostAllocator_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostAllocator_Extension, free);

#ifdef __cplusplus
}
#endif

namespace pjrt {

PJRT_HostAllocator_Extension CreateHostAllocatorExtension(
    PJRT_Extension_Base* next,
    PJRT_HostAllocator_GetPreferredAlignment get_preferred_alignment,
    PJRT_HostAllocator_Allocate allocate, PJRT_HostAllocator_Free free);

}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_HOST_ALLOCATOR_HOST_ALLOCATOR_EXTENSION_H_
