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

#ifndef XLA_PJRT_EXTENSIONS_CROSS_HOST_TRANSFERS_PJRT_C_API_CROSS_HOST_TRANSFERS_EXTENSION_H_
#define XLA_PJRT_EXTENSIONS_CROSS_HOST_TRANSFERS_PJRT_C_API_CROSS_HOST_TRANSFERS_EXTENSION_H_

#include <cstddef>
#include <cstdint>

#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality for cross-host device transfers, which
// are supported with the PjRtClient::MakeCrossHostReceiveBuffers() and
// PjRtBuffer::CopyToRemoteDevice() APIs.

// Version 2 adds an alternate API for cross-host transfers:
// CrossHostSendBuffers and CrossHostReceiveBuffers. These methods allow PjRt
// clients to implement various optimizations for cross-host transfers.

#define PJRT_API_CROSS_HOST_TRANSFERS_EXTENSION_VERSION 5

// ---------------------------------- Methods ----------------------------------

// Structs and methods prefixed with
// PJRT_Transfers_PJRT_Client_CrossHost{Send,Receive}Buffers correspond to the
// second cross-host transfers API.
struct PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  size_t num_buffers;
  PJRT_Buffer** buffers;
  const xla::GlobalDeviceId* dst_global_device_ids;  // Has size num_buffers.
  const xla::CrossHostTransferKey* transfer_keys;  // Has size num_buffers.
  PJRT_Event** send_events;  // Output; has size num_buffers.
};

PJRT_DEFINE_STRUCT_TRAITS(PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args,
                          send_events);

typedef PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args* args);

struct PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  size_t num_shapes;
  size_t* shape_num_dims;
  const int64_t** num_dims;
  PJRT_Buffer_Type* element_types;
  PJRT_Buffer_MemoryLayout** layouts;
  PJRT_Device* device;
  const xla::GlobalDeviceId* src_global_device_ids;      // Has size num_shapes.
  const xla::CrossHostTransferKey* transfer_keys;        // Has size num_shapes.
  PJRT_Buffer** buffers;  // Output; has size num_shapes.
};

PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args, buffers);

typedef PJRT_Error* PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(
    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args* args);

// The structs and methods below correspond to the original cross-host transfers
// API.
typedef void (*PJRT_Transfers_CrossHostOnCanceledCallback)(PJRT_Error* error,
                                                           void* user_arg);

typedef void (*PJRT_Transfers_CrossHostSendCancelNotifier)(
    const char* serialized_descriptor, size_t serialized_descriptor_size,
    PJRT_Error_Code reason, const char* error_message,
    size_t error_message_size,
    PJRT_Transfers_CrossHostOnCanceledCallback on_canceled,
    void* on_canceled_user_arg, void* user_arg);

typedef void (*PJRT_Transfers_CrossHostRecvNotifier)(
    PJRT_Error* error, const char** serialized_descriptors,
    size_t* descriptors_sizes, size_t num_descriptors, void* user_arg,
    PJRT_Transfers_CrossHostSendCancelNotifier cancel_notifier,
    void* cancel_notifier_user_arg);

struct PJRT_Transfers_CrossHostRecvNotifierInfo {
  void* user_arg;
  PJRT_Transfers_CrossHostRecvNotifier notifier;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Transfers_CrossHostRecvNotifierInfo, notifier);

struct PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  size_t num_shapes;
  size_t* shape_num_dims;
  const int64_t** num_dims;
  PJRT_Buffer_Type* element_types;
  PJRT_Buffer_MemoryLayout** layouts;
  PJRT_Device* device;
  PJRT_Transfers_CrossHostRecvNotifierInfo notifier;
  PJRT_Buffer** buffers;  // out
  size_t num_buffers;     // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args, num_buffers);

typedef PJRT_Error* PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers(
    PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args* args);

typedef void (*PJRT_Transfers_CrossHostRemoteSendCallback)(
    PJRT_Error* error, bool sends_were_enqueued, void* user_arg);

struct PJRT_Transfers_CrossHostRemoteSendCallbackInfo {
  void* user_arg;
  PJRT_Transfers_CrossHostRemoteSendCallback on_done;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Transfers_CrossHostRemoteSendCallbackInfo,
                          on_done);

struct PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  // `PJRT_Buffer_CopyToRemoteDevice` is responsible for freeing the event.
  PJRT_Event* event;
  // The lifetime of the descriptor data extends until the event is set.
  char** serialized_descriptor;
  size_t* serialized_descriptor_size;
  PJRT_Transfers_CrossHostRemoteSendCallbackInfo on_done;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args,
                          on_done);

typedef void PJRT_Buffer_CopyToRemoteDevice(
    PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args* args);

// --------------------------- Extension entrypoint ----------------------------

// NOLINTBEGIN: Non-lowercase struct member names follow the convention of the
// PJRT C API.
typedef struct PJRT_CrossHostTransfers_Extension {
  PJRT_Extension_Base base;
  PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers*
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers;
  PJRT_Buffer_CopyToRemoteDevice* PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice;
  PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers*
      PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers;
  PJRT_Transfers_PJRT_Client_CrossHostSendBuffers*
      PJRT_Transfers_PJRT_Client_CrossHostSendBuffers;
} PJRT_CrossHostTransfers_Extension;
// NOLINTEND

PJRT_DEFINE_STRUCT_TRAITS(PJRT_CrossHostTransfers_Extension,
                          PJRT_Transfers_PJRT_Client_CrossHostSendBuffers);

#ifdef __cplusplus
}
#endif

namespace pjrt {
PJRT_CrossHostTransfers_Extension CreateCrossHostTransfersExtension(
    PJRT_Extension_Base* next = nullptr);
PJRT_Transfers_CrossHostRecvNotifierInfo CppCrossHostRecvNotifierToC(
    const PJRT_Api* c_api, xla::PjRtCrossHostRecvNotifier cpp_notifier);
PJRT_Transfers_CrossHostRemoteSendCallbackInfo
CppCrossHostRemoteSendCallbackToC(
    const PJRT_Api* c_api, xla::PjRtBuffer::RemoteSendCallback cpp_callback);
}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_CROSS_HOST_TRANSFERS_PJRT_C_API_CROSS_HOST_TRANSFERS_EXTENSION_H_
