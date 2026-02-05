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

#ifndef XLA_PJRT_EXTENSIONS_EXECUTABLE_METADATA_EXECUTABLE_METADATA_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_EXECUTABLE_METADATA_EXECUTABLE_METADATA_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_client.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality for retrieving executable metadata,
// such as the fingerprint, from an executable.

#define PJRT_API_EXECUTABLE_METADATA_EXTENSION_VERSION 1

typedef struct PJRT_ExecutableMetadata {
  const char* serialized_metadata;
  size_t serialized_metadata_size;
} PJRT_ExecutableMetadata;

// ---------------------------------- Methods ----------------------------------

struct PJRT_ExecutableMetadata_GetExecutableMetadata_Args {
  PJRT_Executable* executable;
  PJRT_ExecutableMetadata* metadata;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableMetadata_GetExecutableMetadata_Args,
                          metadata);

// Extracts the metadata from the executable, stored it in the
// PjRtExecutableMetadata proto, and returns the serialized proto string.
typedef PJRT_Error* (*PJRT_ExecutableMetadata_GetExecutableMetadata)(
    PJRT_ExecutableMetadata_GetExecutableMetadata_Args* args);

struct PJRT_ExecutableMetadata_DestroySerializedMetadata_Args {
  PJRT_ExecutableMetadata* metadata;
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_ExecutableMetadata_DestroySerializedMetadata_Args, metadata);

// Destroys the serialized metadata and releases the memory.
typedef void (*PJRT_ExecutableMetadata_DestroySerializedMetadata)(
    PJRT_ExecutableMetadata_DestroySerializedMetadata_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_ExecutableMetadata_Extension {
  PJRT_Extension_Base base;
  PJRT_ExecutableMetadata_GetExecutableMetadata get_executable_metadata;
  PJRT_ExecutableMetadata_DestroySerializedMetadata destroy_serialized_metadata;
} PJRT_ExecutableMetadata_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecutableMetadata_Extension,
                          destroy_serialized_metadata);

#ifdef __cplusplus
}
#endif

namespace pjrt {
PJRT_ExecutableMetadata_Extension CreateExecutableMetadataExtension(
    PJRT_Extension_Base* next,
    PJRT_ExecutableMetadata_GetExecutableMetadata get_executable_metadata);
}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_EXECUTABLE_METADATA_EXECUTABLE_METADATA_EXTENSION_H_
