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
#include "xla/pjrt/c/pjrt_c_api_multi_slice_extension.h"

#ifdef __cplusplus
extern "C" {
#endif

// This extension provides functionality related to Megascale.

#define PJRT_API_MEGASCALE_EXTENSION_VERSION 1

// NOLINTBEGIN(modernize-use-using)
typedef struct PJRT_Megascale_ClientContext PJRT_Megascale_ClientContext;
typedef struct PJRT_MultiSlice_Config PJRT_MultiSlice_Config;
typedef struct PJRT_Collectives PJRT_Collectives;
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

struct PJRT_Megascale_ProcessesInfo {
  const char** addresses;
  size_t* address_sizes;
  size_t num_addresses;
  int32_t* slice_indexes;  // May be null if not set.
  size_t num_slice_indexes;
  int32_t* per_slice_indexes;  // May be null if not set.
  size_t num_per_slice_indexes;
  int32_t num_devices_per_process;
};

struct PJRT_Megascale_CreateMegascaleCollectives_Args {
  size_t struct_size;
  PJRT_Megascale_ClientContext* client_context;
  PJRT_Megascale_ProcessesInfo* processes_info;
  const char* dcn_topology;  // Serialized DCNTopology proto. May be null.
  size_t dcn_topology_size;
  PJRT_Collectives* collectives;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_CreateMegascaleCollectives_Args,
                          collectives);

typedef PJRT_Error* PJRT_Megascale_CreateMegascaleCollectives(
    PJRT_Megascale_CreateMegascaleCollectives_Args* args);
struct PJRT_Megascale_DeviceId_To_MegascaleId_Args {
  size_t struct_size;
  int32_t slice_id;
  int32_t per_slice_device_id;
  int64_t megascale_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_DeviceId_To_MegascaleId_Args,
                          megascale_id);
typedef PJRT_Error* PJRT_Megascale_DeviceId_To_MegascaleId(
    PJRT_Megascale_DeviceId_To_MegascaleId_Args* args);

struct PJRT_Megascale_MegascaleId_To_DeviceId_Args {
  size_t struct_size;
  int64_t megascale_id;
  int32_t slice_id;             // out
  int32_t per_slice_device_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_MegascaleId_To_DeviceId_Args,
                          per_slice_device_id);
typedef PJRT_Error* PJRT_Megascale_MegascaleId_To_DeviceId(
    PJRT_Megascale_MegascaleId_To_DeviceId_Args* args);

typedef void (*PJRT_Megascale_ErrorHandler)(const char* serialized_error,
                                            size_t serialized_error_size,
                                            void* user_data);

struct PJRT_Megascale_RegisterErrorHandler_Args {
  size_t struct_size;
  const char* handler_name;
  size_t handler_name_size;
  PJRT_Megascale_ErrorHandler handler;
  void* user_data;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_RegisterErrorHandler_Args, user_data);

typedef PJRT_Error* PJRT_Megascale_RegisterErrorHandler(
    PJRT_Megascale_RegisterErrorHandler_Args* args);

struct PJRT_Megascale_UnregisterErrorHandler_Args {
  size_t struct_size;
  const char* handler_name;
  size_t handler_name_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_UnregisterErrorHandler_Args,
                          handler_name_size);

typedef PJRT_Error* PJRT_Megascale_UnregisterErrorHandler(
    PJRT_Megascale_UnregisterErrorHandler_Args* args);

typedef struct PJRT_Megascale_ErrorAggregator PJRT_Megascale_ErrorAggregator;
typedef struct PJRT_Megascale_ErrorDigest PJRT_Megascale_ErrorDigest;

struct PJRT_Megascale_ErrorAggregator_Create_Args {
  size_t struct_size;
  const char* app_type;
  size_t app_type_size;
  PJRT_Megascale_ErrorAggregator* aggregator;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_Create_Args,
                          aggregator);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_Create(
    PJRT_Megascale_ErrorAggregator_Create_Args* args);

struct PJRT_Megascale_ErrorAggregator_Delete_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_Delete_Args,
                          aggregator);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_Delete(
    PJRT_Megascale_ErrorAggregator_Delete_Args* args);

struct PJRT_Megascale_ErrorDigest_Delete_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorDigest* digest;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorDigest_Delete_Args, digest);
typedef PJRT_Error* PJRT_Megascale_ErrorDigest_Delete(
    PJRT_Megascale_ErrorDigest_Delete_Args* args);

struct PJRT_Megascale_ErrorAggregator_AddError_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
  const char* worker_id;
  size_t worker_id_size;
  const char* serialized_error;
  size_t serialized_error_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_AddError_Args,
                          serialized_error_size);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_AddError(
    PJRT_Megascale_ErrorAggregator_AddError_Args* args);

struct PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
  PJRT_Megascale_ErrorDigest* digest;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args, digest);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_ProcessAndShutdown(
    PJRT_Megascale_ErrorAggregator_ProcessAndShutdown_Args* args);

struct PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
  PJRT_Megascale_ErrorDigest* digest;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args,
                          digest);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_LogErrorDigest(
    PJRT_Megascale_ErrorAggregator_LogErrorDigest_Args* args);

struct PJRT_Megascale_ErrorAggregator_Size_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
  size_t size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_Size_Args, size);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_Size(
    PJRT_Megascale_ErrorAggregator_Size_Args* args);

struct PJRT_Megascale_ErrorAggregator_Active_Args {
  size_t struct_size;
  PJRT_Megascale_ErrorAggregator* aggregator;
  bool active;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_ErrorAggregator_Active_Args, active);
typedef PJRT_Error* PJRT_Megascale_ErrorAggregator_Active(
    PJRT_Megascale_ErrorAggregator_Active_Args* args);

typedef struct PJRT_Megascale_SerializedAddresses
    PJRT_Megascale_SerializedAddresses;

struct PJRT_Megascale_GetInterfaceAddressesHelper_Args {
  size_t struct_size;
  const char* megascale_port_name;
  size_t megascale_port_name_size;
  int32_t megascale_port;
  const char** interface_prefixes;
  const size_t* interface_prefixes_sizes;
  size_t num_interface_prefixes;
  bool use_all_interfaces;
  bool limit_to_process_numa_local_interfaces;
  const char** serialized_addresses;                                   // out
  size_t* serialized_addresses_sizes;                                  // out
  size_t num_addresses;                                                // out
  PJRT_Megascale_SerializedAddresses* addresses;                       // out
  void (*addresses_deleter)(PJRT_Megascale_SerializedAddresses* ptr);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_GetInterfaceAddressesHelper_Args,
                          addresses_deleter);
typedef PJRT_Error* PJRT_Megascale_GetInterfaceAddressesHelper(
    PJRT_Megascale_GetInterfaceAddressesHelper_Args* args);

typedef struct PJRT_Megascale_Extension {
  PJRT_Extension_Base base;
  PJRT_Megascale_CreateClientContextFromPjRtClient*
      create_client_context_from_pjrt_client;
  PJRT_Megascale_CreateDefaultClientContext* create_default_client_context;
  PJRT_Megascale_DeleteClientContext* delete_client_context;
  PJRT_Megascale_CreateAoTConfig* create_aot_config;
  PJRT_Megascale_CreateMultiSliceConfig* create_multi_slice_config;
  void* delete_multi_slice_config;                     // deprecated
  void* multi_slice_config_num_slices;                 // deprecated
  void* multi_slice_config_slice_id;                   // deprecated
  void* multi_slice_config_get_num_devices_per_slice;  // deprecated
  void* multi_slice_config_serialize;                  // deprecated
  PJRT_Megascale_ClientContext_Initialize* client_context_initialize;
  PJRT_Megascale_ClientContext_UnblockPendingWork*
      client_context_unblock_pending_work;
  PJRT_Megascale_ClientContext_MegascalePort* client_context_megascale_port;
  PJRT_Megascale_CreateMegascaleCollectives* create_megascale_collectives;
  PJRT_Megascale_DeviceId_To_MegascaleId* device_id_to_megascale_id;
  PJRT_Megascale_MegascaleId_To_DeviceId* megascale_id_to_device_id;
  PJRT_Megascale_RegisterErrorHandler* register_megascale_error_handler;
  PJRT_Megascale_UnregisterErrorHandler* unregister_megascale_error_handler;
  PJRT_Megascale_ErrorAggregator_Create* error_aggregator_create;
  PJRT_Megascale_ErrorAggregator_Delete* error_aggregator_delete;
  PJRT_Megascale_ErrorDigest_Delete* error_digest_delete;
  PJRT_Megascale_ErrorAggregator_AddError* error_aggregator_add_error;
  PJRT_Megascale_ErrorAggregator_ProcessAndShutdown*
      error_aggregator_process_and_shutdown;
  PJRT_Megascale_ErrorAggregator_LogErrorDigest*
      error_aggregator_log_error_digest;
  PJRT_Megascale_ErrorAggregator_Size* error_aggregator_size;
  PJRT_Megascale_ErrorAggregator_Active* error_aggregator_active;
  PJRT_Megascale_GetInterfaceAddressesHelper* get_interface_addresses_helper;
} PJRT_Megascale_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Megascale_Extension,
                          get_interface_addresses_helper);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_MEGASCALE_EXTENSION_H_
