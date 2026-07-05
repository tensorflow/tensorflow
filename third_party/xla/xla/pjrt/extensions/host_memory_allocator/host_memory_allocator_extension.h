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

#ifndef XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_HOST_MEMORY_ALLOCATOR_EXTENSION_VERSION 0

struct PJRT_HostMemoryAllocator_Allocate_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_Client* client;
  size_t size;

  // AllocateOptions.
  int numa_node;

  uint8_t* ptr;                           // out.
  void (*deleter)(void* ptr, void* arg);  // out.
  void* deleter_arg;                      // out.
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostMemoryAllocator_Allocate_Args, deleter_arg);

#define PJRT_HostMemoryAllocator_Allocate_Args_STRUCT_SIZE \
  sizeof(struct PJRT_HostMemoryAllocator_Allocate_Args)

typedef PJRT_Error* (*PJRT_HostMemoryAllocator_Allocate)(
    PJRT_HostMemoryAllocator_Allocate_Args* args);

typedef struct PJRT_HostMemoryAllocator_Extension {
  PJRT_Extension_Base base;
  PJRT_HostMemoryAllocator_Allocate allocate;
} PJRT_HostMemoryAllocator_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_HostMemoryAllocator_Extension, allocate);

#define PJRT_HostMemoryAllocator_Extension_STRUCT_SIZE \
  sizeof(struct PJRT_HostMemoryAllocator_Extension)

#ifdef __cplusplus
}
#endif
#endif  // XLA_PJRT_EXTENSIONS_HOST_MEMORY_ALLOCATOR_HOST_MEMORY_ALLOCATOR_EXTENSION_H_
