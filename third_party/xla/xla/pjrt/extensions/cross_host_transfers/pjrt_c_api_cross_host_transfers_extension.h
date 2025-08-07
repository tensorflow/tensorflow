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

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality for cross-host device transfers, which
// are supported with the PjRtClient::MakeCrossHostReceiveBuffers() and
// PjRtBuffer::CopyToRemoteDevice() APIs.

#define PJRT_API_CROSS_HOST_TRANSFERS_EXTENSION_VERSION 1

// ---------------------------------- Methods ----------------------------------

typedef void (*PJRT_Transfers_CrossHostRecvNotifier)(
    PJRT_Error* error, const char** serialized_descriptors,
    size_t* descriptors_sizes, size_t num_descriptors, void* user_arg);

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

struct PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  const char* serialized_descriptor;
  size_t serialized_descriptor_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args,
                          serialized_descriptor_size);

typedef void PJRT_Buffer_CopyToRemoteDevice(
    PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args* args);

// --------------------------- Extension entrypoint ----------------------------

typedef struct PJRT_CrossHostTransfers_Extension {
  PJRT_Extension_Base base;

  PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers*
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers;
  PJRT_Buffer_CopyToRemoteDevice* PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice;
} PJRT_CrossHostTransfers_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CrossHostTransfers_Extension,
                          PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice);

#ifdef __cplusplus
}
#endif

namespace pjrt {
PJRT_CrossHostTransfers_Extension CreateCrossHostTransfersExtension(
    PJRT_Extension_Base* next = nullptr);
}  // namespace pjrt

#endif  // XLA_PJRT_EXTENSIONS_CROSS_HOST_TRANSFERS_PJRT_C_API_CROSS_HOST_TRANSFERS_EXTENSION_H_
