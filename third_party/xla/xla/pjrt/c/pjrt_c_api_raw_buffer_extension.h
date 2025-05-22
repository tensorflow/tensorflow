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

#ifndef XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PJRT_RawBuffer PJRT_RawBuffer;

struct PJRT_RawBuffer_CreateRawAliasOfBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_RawBuffer* raw_buffer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_CreateRawAliasOfBuffer_Args,
                          raw_buffer);

typedef PJRT_Error* PJRT_RawBuffer_CreateRawAliasOfBuffer(
    PJRT_RawBuffer_CreateRawAliasOfBuffer_Args* args);

struct PJRT_RawBuffer_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_Destroy_Args, buffer);

// Frees the PJRT_RawBuffer.
typedef PJRT_Error* PJRT_RawBuffer_Destroy(PJRT_RawBuffer_Destroy_Args* args);

struct PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  size_t on_device_size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args,
                          on_device_size_in_bytes);

// Gets the number of bytes of the buffer storage on the device
typedef PJRT_Error* PJRT_RawBuffer_GetOnDeviceSizeInBytes(
    PJRT_RawBuffer_GetOnDeviceSizeInBytes_Args* args);

struct PJRT_RawBuffer_GetMemorySpace_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  PJRT_Memory* memory_space;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_GetMemorySpace_Args, memory_space);

// Gets the memory space that this buffer is attached to.
typedef PJRT_Error* PJRT_RawBuffer_GetMemorySpace(
    PJRT_RawBuffer_GetMemorySpace_Args* args);

struct PJRT_RawBuffer_CopyRawDeviceToHost_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  void* dst;
  int64_t offset;
  int64_t transfer_size;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_CopyRawDeviceToHost_Args, event);

typedef PJRT_Error* PJRT_RawBuffer_CopyRawDeviceToHost(
    PJRT_RawBuffer_CopyRawDeviceToHost_Args* args);

struct PJRT_RawBuffer_CopyRawHostToDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  const void* src;
  int64_t offset;
  int64_t transfer_size;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_CopyRawHostToDevice_Args, event);

typedef PJRT_Error* PJRT_RawBuffer_CopyRawHostToDevice(
    PJRT_RawBuffer_CopyRawHostToDevice_Args* args);

// This extension provides capabilities around constructing raw buffers which
// alias PJRT_Buffers. The extension is both optional and experimental, meaning
// ABI-breaking and other incompatible changes may be introduced at any time.

#define PJRT_API_RAW_BUFFER_EXTENSION_VERSION 1
#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

typedef struct PJRT_RawBuffer_Extension {
  PJRT_Extension_Base base;
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CreateRawAliasOfBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_GetOnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_GetMemorySpace);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CopyRawHostToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CopyRawDeviceToHost);
} PJRT_RawBuffer_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_Extension,
                          PJRT_RawBuffer_CopyRawDeviceToHost);

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTENSION_H_
