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

#ifndef XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to Megascale.

#define PJRT_API_MEGASCALE_EXTENSION_VERSION 1

typedef struct PJRT_Megascale_ClientContext PJRT_Megascale_ClientContext;
typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;
typedef struct PJRT_Megascale_NumDevicesPerSlice
    PJRT_Megascale_NumDevicesPerSlice;
typedef struct PJRT_Megascale_SerializedConfig PJRT_Megascale_SerializedConfig;

struct PJRT_Megascale_CreateClientContextFromPjRtClient_Args {
  size_t struct_size;
  PJRT_Client* client;
  PJRT_Megascale_ClientContext* client_context;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_CreateClientContextFromPjRtClient_Args,
                          client_context);

// Creates a Megascale client context from a PjRt client.
typedef PJRT_Error* PJRT_Megascale_CreateClientContextFromPjRtClient(
    PJRT_Megascale_CreateClientContextFromPjRtClient_Args* args);

struct PJRT_Megascale_CreateDefaultClientContext_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_CreateDefaultClientContext_Args,
                          client_context);

// Creates a Megascale client context with default options.
typedef PJRT_Error* PJRT_Megascale_CreateDefaultClientContext(
    PJRT_Megascale_CreateDefaultClientContext_Args* args);

struct PJRT_Megascale_DeleteClientContext_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_DeleteClientContext_Args,
                          client_context);

// Deletes a Megascale client context.
typedef PJRT_Error* PJRT_Megascale_DeleteClientContext(
    PJRT_Megascale_DeleteClientContext_Args* args);

struct PJRT_Megascale_CreateAoTConfig_Args {
  size_t struct_size;
  const PJRT_TopologyDescription* topology;
  int32_t num_slices;

  PJRT_MultiSlice_Config* multi_slice_config;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_CreateAoTConfig_Args,
                          multi_slice_config);

// Creates a Megascale AoT config.
typedef PJRT_Error* PJRT_Megascale_CreateAoTConfig(
    PJRT_Megascale_CreateAoTConfig_Args* args);

struct PJRT_Megascale_CreateMultiSliceConfig_Args {
  size_t struct_size;
  const PJRT_TopologyDescription* topology;
  int32_t num_slices;
  int32_t local_slice_id;
  int32_t local_host_id;
  // Serialized xla::megascale::runtime::EndpointAddresses proto.
  const char* endpoint_addresses;
  int32_t endpoint_addresses_size;
  // Serialized xla::megascale::runtime::DCNTopology proto.
  const char* dcn_topology;
  int32_t dcn_topology_size;
  PJRT_Megascale_ClientContext* client_context;

  PJRT_MultiSlice_Config* multi_slice_config;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_CreateMultiSliceConfig_Args,
                          multi_slice_config);

// Creates a Megascale multi-slice config.
typedef PJRT_Error* PJRT_Megascale_CreateMultiSliceConfig(
    PJRT_Megascale_CreateMultiSliceConfig_Args* args);

struct PJRT_Megascale_DeleteMultiSliceConfig_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* multi_slice_config;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_DeleteMultiSliceConfig_Args,
                          multi_slice_config);

// Deletes a Megascale multi-slice config.
typedef PJRT_Error* PJRT_Megascale_DeleteMultiSliceConfig(
    PJRT_Megascale_DeleteMultiSliceConfig_Args* args);

struct PJRT_Megascale_ClientContext_Initialize_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ClientContext_Initialize_Args,
                          client_context);
typedef PJRT_Error* PJRT_Megascale_ClientContext_Initialize(
    PJRT_Megascale_ClientContext_Initialize_Args* args);

struct PJRT_Megascale_ClientContext_UnblockPendingWork_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;
  int32_t launch_id;
  int64_t expire_after_ms;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ClientContext_UnblockPendingWork_Args,
                          expire_after_ms);
typedef PJRT_Error* PJRT_Megascale_ClientContext_UnblockPendingWork(
    PJRT_Megascale_ClientContext_UnblockPendingWork_Args* args);

struct PJRT_Megascale_ClientContext_MegascalePort_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;
  int32_t port;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ClientContext_MegascalePort_Args,
                          port);
typedef PJRT_Error* PJRT_Megascale_ClientContext_MegascalePort(
    PJRT_Megascale_ClientContext_MegascalePort_Args* args);

struct PJRT_Megascale_MultiSliceConfig_NumSlices_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  int32_t num_slices;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_MultiSliceConfig_NumSlices_Args,
                          num_slices);
typedef PJRT_Error* PJRT_Megascale_MultiSliceConfig_NumSlices(
    PJRT_Megascale_MultiSliceConfig_NumSlices_Args* args);

struct PJRT_Megascale_MultiSliceConfig_SliceId_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  int32_t slice_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_MultiSliceConfig_SliceId_Args,
                          slice_id);
typedef PJRT_Error* PJRT_Megascale_MultiSliceConfig_SliceId(
    PJRT_Megascale_MultiSliceConfig_SliceId_Args* args);

struct PJRT_Megascale_MultiSliceConfig_GetNumDevicesPerSlice_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  size_t num_devices_per_slice_map;                              // out
  const int32_t* slice_ids;                                      // out
  const int32_t* num_devices;                                    // out
  PJRT_Megascale_NumDevicesPerSlice* devices_per_slice_map_ptr;  // out
  void (*devices_per_slice_map_deleter)(
      PJRT_Megascale_NumDevicesPerSlice* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_Megascale_MultiSliceConfig_GetNumDevicesPerSlice_Args,
    devices_per_slice_map_deleter);
typedef PJRT_Error* PJRT_Megascale_MultiSliceConfig_GetNumDevicesPerSlice(
    PJRT_Megascale_MultiSliceConfig_GetNumDevicesPerSlice_Args* args);

struct PJRT_Megascale_MultiSliceConfig_Serialize_Args {
  size_t struct_size;
  PJRT_MultiSlice_Config* config;
  const char* serialized;                                  // out
  size_t size;                                             // out
  PJRT_Megascale_SerializedConfig* serialized_config_ptr;  // out
  void (*serialized_config_deleter)(
      PJRT_Megascale_SerializedConfig* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_MultiSliceConfig_Serialize_Args,
                          serialized_config_deleter);
typedef PJRT_Error* PJRT_Megascale_MultiSliceConfig_Serialize(
    PJRT_Megascale_MultiSliceConfig_Serialize_Args* args);

typedef struct PJRT_Megascale_Extension {
  PJRT_Extension_Base base;
  PJRT_Megascale_CreateClientContextFromPjRtClient*
      create_client_context_from_pjrt_client;
  PJRT_Megascale_CreateDefaultClientContext* create_default_client_context;
  PJRT_Megascale_DeleteClientContext* delete_client_context;
  PJRT_Megascale_CreateAoTConfig* create_aot_config;
  PJRT_Megascale_CreateMultiSliceConfig* create_multi_slice_config;
  PJRT_Megascale_DeleteMultiSliceConfig* delete_multi_slice_config;
  PJRT_Megascale_MultiSliceConfig_NumSlices* multi_slice_config_num_slices;
  PJRT_Megascale_MultiSliceConfig_SliceId* multi_slice_config_slice_id;
  PJRT_Megascale_MultiSliceConfig_GetNumDevicesPerSlice*
      multi_slice_config_get_num_devices_per_slice;
  PJRT_Megascale_MultiSliceConfig_Serialize* multi_slice_config_serialize;
  PJRT_Megascale_ClientContext_Initialize* client_context_initialize;
  PJRT_Megascale_ClientContext_UnblockPendingWork*
      client_context_unblock_pending_work;
  PJRT_Megascale_ClientContext_MegascalePort* client_context_megascale_port;
} PJRT_Megascale_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_Extension,
                          client_context_megascale_port);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_
