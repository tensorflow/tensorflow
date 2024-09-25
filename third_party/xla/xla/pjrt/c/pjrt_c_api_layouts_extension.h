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

#ifndef XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides capabilities around custom on-device memory layouts
// for PJRT_Buffers. The extension is both optional and experimental, meaning
// ABI-breaking and other incompatible changes may be introduced at any time.
//
// If this extension is provided, JAX and possibly other frameworks will assume
// that the compiler MLIR input can contain "mhlo.layout_mode" attributes on
// program inputs and outputs, which should then be reflected by the runtime
// methods in this extension. See
// https://github.com/openxla/xla/blob/main/xla/pjrt/layout_mode.h for more
// details.

#define PJRT_API_LAYOUTS_EXTENSION_VERSION 1

// -------------------------------- Data types ---------------------------------

typedef struct PJRT_Layouts_MemoryLayout PJRT_Layouts_MemoryLayout;
typedef struct PJRT_Layouts_SerializedLayout PJRT_Layouts_SerializedLayout;

// ---------------------------------- Methods ----------------------------------

struct PJRT_Layouts_MemoryLayout_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Layouts_MemoryLayout* layout;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_MemoryLayout_Destroy_Args, layout);

// Frees `layout`. `layout` can be nullptr.
typedef PJRT_Error* PJRT_Layouts_MemoryLayout_Destroy(
    PJRT_Layouts_MemoryLayout_Destroy_Args* args);

struct PJRT_Layouts_MemoryLayout_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Layouts_MemoryLayout* layout;

  // Lives only as long as serialized_layout
  const char* serialized_bytes;  // out
  size_t serialized_bytes_size;  // out

  PJRT_Layouts_SerializedLayout* serialized_layout;  // backs serialized_bytes.

  // cleanup fn must be called to free the backing memory for serialized_bytes.
  // Should only be called once on serialized_layout.
  void (*serialized_layout_deleter)(
      PJRT_Layouts_SerializedLayout* s_layout);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_MemoryLayout_Serialize_Args,
                          serialized_layout_deleter);

// Serializes the memory layout into a string.
typedef PJRT_Error* PJRT_Layouts_MemoryLayout_Serialize(
    PJRT_Layouts_MemoryLayout_Serialize_Args* args);

struct PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Layouts_MemoryLayout* layout;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args, layout);

// Returns the memory layout of the data in this buffer. Returned `layout` must
// be freed via PJRT_Layouts_MemoryLayout_Destroy.
typedef PJRT_Error* PJRT_Layouts_PJRT_Buffer_MemoryLayout(
    PJRT_Layouts_PJRT_Buffer_MemoryLayout_Args* args);

struct PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Layouts_MemoryLayout* layout;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args,
                          layout);

// Returns the default memory layout of the client given buffer type and dims.
typedef PJRT_Error* PJRT_Layouts_PJRT_Client_GetDefaultLayout(
    PJRT_Layouts_PJRT_Client_GetDefaultLayout_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_Layouts_Extension {
  size_t struct_size;
  PJRT_Extension_Type type;
  PJRT_Extension_Base* next;

  PJRT_Layouts_MemoryLayout_Destroy* PJRT_Layouts_MemoryLayout_Destroy;
  PJRT_Layouts_MemoryLayout_Serialize* PJRT_Layouts_MemoryLayout_Serialize;

  PJRT_Layouts_PJRT_Client_GetDefaultLayout*
      PJRT_Layouts_PJRT_Client_GetDefaultLayout;

  PJRT_Layouts_PJRT_Buffer_MemoryLayout* PJRT_Layouts_PJRT_Buffer_MemoryLayout;
} PJRT_Layouts_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Layouts_Extension,
                          PJRT_Layouts_PJRT_Buffer_MemoryLayout);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_LAYOUTS_EXTENSION_H_
