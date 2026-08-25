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

#ifndef XLA_PJRT_C_PJRT_C_API_RMA_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_RMA_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides capabilities for Remote Memory Access (RMA) /
// one-sided RDMA transfers across device memory spaces without active target
// CPU involvement.

#define PJRT_API_RMA_EXTENSION_VERSION 1

typedef struct PJRT_Rma_RemoteWindow PJRT_Rma_RemoteWindow;

// 1. Export window descriptor from a registered local RawBuffer.
struct PJRT_Rma_ExportWindow_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* buffer;
  const char* serialized_descriptor;  // out: managed by plugin or buffer
  size_t serialized_descriptor_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_ExportWindow_Args,
                          serialized_descriptor_size);

typedef PJRT_Error* PJRT_Rma_ExportWindow(PJRT_Rma_ExportWindow_Args* args);

// 2. Import remote window descriptor into a remote window handle.
struct PJRT_Rma_ImportWindow_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  const char* serialized_descriptor;
  size_t serialized_descriptor_size;
  PJRT_Rma_RemoteWindow* remote_window;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_ImportWindow_Args, remote_window);

typedef PJRT_Error* PJRT_Rma_ImportWindow(PJRT_Rma_ImportWindow_Args* args);

// 3. Destroy a remote window handle.
struct PJRT_Rma_DestroyRemoteWindow_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Rma_RemoteWindow* remote_window;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_DestroyRemoteWindow_Args, remote_window);

typedef PJRT_Error* PJRT_Rma_DestroyRemoteWindow(
    PJRT_Rma_DestroyRemoteWindow_Args* args);

// 4. One-sided Put directly into remote HBM / device memory.
struct PJRT_Rma_Put_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* src_buffer;
  int64_t src_offset_bytes;
  PJRT_Rma_RemoteWindow* dst_remote_window;
  int64_t dst_offset_bytes;
  int64_t transfer_size_bytes;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_Put_Args, event);

typedef PJRT_Error* PJRT_Rma_Put(PJRT_Rma_Put_Args* args);

// 5. Remote hardware signaling / atomic sync.
struct PJRT_Rma_Signal_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Rma_RemoteWindow* dst_remote_window;
  uint64_t signal_id;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_Signal_Args, event);

typedef PJRT_Error* PJRT_Rma_Signal(PJRT_Rma_Signal_Args* args);

struct PJRT_Rma_WaitSignal_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_RawBuffer* local_buffer;
  uint64_t signal_id;
  PJRT_Event* event;  // out
  const char* local_window_descriptor;
  size_t local_window_descriptor_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_WaitSignal_Args,
                          local_window_descriptor_size);

typedef PJRT_Error* PJRT_Rma_WaitSignal(PJRT_Rma_WaitSignal_Args* args);

// 6. Free descriptor memory allocated by PJRT_Rma_ExportWindow.
struct PJRT_Rma_DestroyDescriptor_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* serialized_descriptor;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_DestroyDescriptor_Args,
                          serialized_descriptor);

typedef PJRT_Error* PJRT_Rma_DestroyDescriptor(
    PJRT_Rma_DestroyDescriptor_Args* args);

// Extension struct
#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

typedef struct PJRT_Rma_Extension {
  PJRT_Extension_Base base;
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_ExportWindow);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_ImportWindow);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_DestroyRemoteWindow);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_Put);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_Signal);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_WaitSignal);
  PJRT_NO_DISCARD _PJRT_API_STRUCT_FIELD(PJRT_Rma_DestroyDescriptor);
} PJRT_Rma_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Rma_Extension, PJRT_Rma_DestroyDescriptor);

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_RMA_EXTENSION_H_
