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
#include "xla/pjrt/c/pjrt_c_api_device_event.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PJRT_RawBuffer PJRT_RawBuffer;

typedef struct PJRT_RawBuffer_FunctionTable {
  size_t struct_size;
  size_t instance_size;
  PJRT_Extension_Base* extension_start;
  void (*inc_ref)(PJRT_RawBuffer* raw_buffer);
  void (*dec_ref)(PJRT_RawBuffer* raw_buffer);
  // Gets the number of bytes of the buffer storage on the device
  size_t (*get_on_device_size_in_bytes)(const PJRT_RawBuffer* raw_buffer);
  // Gets the memory space that this buffer is attached to.
  PJRT_Memory* (*get_memory_space)(const PJRT_RawBuffer* raw_buffer);
  // If visible to the host, returns the base pointer for direct access.
  void* (*get_host_pointer)(const PJRT_RawBuffer* raw_buffer);
  // Transfers the buffer to a sub-range of the on-device representation.
  // offset+transfer_size must be less than get_on_device_size_in_bytes. The
  // returned event transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `src` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  PJRT_Error* (*copy_raw_host_to_device_and_return_event)(
      PJRT_RawBuffer* raw_buffer, const void* src, int64_t offset,
      int64_t transfer_size, PJRT_DeviceEventVector* dependencies,
      PJRT_DeviceEvent* event);
  // Transfers a sub-range of the on-device representation of the buffer.
  // offset+transfer_size must be less than get_on_device_size_in_bytes. The
  // returned event transitions to ready on error, or after the transfer has
  // completed.
  //
  // Note that the underlying driver may have requirements
  // on the alignment of `dst` and `offset` as well. Look at implementations of
  // this method for specific alignment requirements.
  PJRT_Error* (*copy_raw_device_to_host_and_return_event)(
      PJRT_RawBuffer* raw_buffer, void* dst, int64_t offset,
      int64_t transfer_size, PJRT_DeviceEventVector* dependencies,
      PJRT_DeviceEvent* event);
  // Return opaque device memory pointer to the underlying memory.
  void* (*opaque_device_memory_data_pointer)(const PJRT_RawBuffer* raw_buffer);
  // Fill `event` with the event that signals when the buffer allocation is
  // complete.
  PJRT_Error* (*make_allocation_ready_event)(PJRT_RawBuffer* raw_buffer,
                                             PJRT_DeviceEvent* event);
  // Fill `event` with the event associated with the buffer (async value).
  PJRT_Error* (*get_raw_buffer_async_value)(PJRT_RawBuffer* raw_buffer,
                                            PJRT_DeviceEvent* event);
  // Returns true if the buffer is mutable.
  bool (*is_mutable)(const PJRT_RawBuffer* raw_buffer);
  // Slices the buffer.
  PJRT_Error* (*slice)(PJRT_RawBuffer* raw_buffer, int64_t offset,
                       int64_t slice_size, PJRT_RawBuffer** sliced_buffer);
  // Blocks on a list of dependencies and then copies directly into
  // dst_raw_buffer. Must set definition_event_promise,
  // when dst_raw_buffer is ready, allocation_event before using dst_raw_buffer
  // and src_usage_event_promise when done using this buffer.
  // transfer_dependency_events can be nullptr in which case the copy runs
  // inline (no deps).
  void (*schedule_copy_to)(PJRT_RawBuffer* src_buffer,
                           PJRT_DeviceEventVector* transfer_dependency_events,
                           PJRT_RawBuffer* dst_buffer,
                           PJRT_DeviceEventPromise* definition_event_promise,
                           PJRT_DeviceEventPromise* src_usage_event_promise,
                           void (*allocation_event_callback)(PJRT_Error* status,
                                                             void* user_data),
                           void* allocation_event_user_data);
} PJRT_RawBuffer_FunctionTable;

struct PJRT_RawBuffer {
  const PJRT_RawBuffer_FunctionTable* vtable;
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_FunctionTable, schedule_copy_to);
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer, vtable);

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

struct PJRT_RawBuffer_GetHostPointer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  void* host_pointer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_GetHostPointer_Args, host_pointer);

// If visible to the host, returns the base pointer for direct access.
typedef PJRT_Error* PJRT_RawBuffer_GetHostPointer(
    PJRT_RawBuffer_GetHostPointer_Args* args);

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

#define PJRT_API_RAW_BUFFER_EXTENSION_VERSION 2
#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

typedef struct PJRT_RawBuffer_Extension {
  PJRT_Extension_Base base;
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CreateRawAliasOfBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_GetOnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_GetMemorySpace);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CopyRawHostToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_CopyRawDeviceToHost);
  _PJRT_API_STRUCT_FIELD(PJRT_RawBuffer_GetHostPointer);
} PJRT_RawBuffer_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RawBuffer_Extension,
                          PJRT_RawBuffer_GetHostPointer);

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_RAW_BUFFER_EXTENSION_H_
