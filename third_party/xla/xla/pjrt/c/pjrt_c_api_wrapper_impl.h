/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_
#define XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "tsl/platform/casts.h"

struct PJRT_Error {
  absl::Status status;
};

struct PJRT_TopologyDescription {
  // nullptr iff the PjRtTopologyDescription isn't owned by the caller. The PJRT
  // C API sometimes returns a topo desc that's owned by the caller and must be
  // freed using PJRT_TopologyDescription_Destroy
  // (e.g. PJRT_TopologyDescription_Create), and sometimes returns a topo desc
  // that's owned by something else (e.g. PJRT_Client_TopologyDescription).
  std::unique_ptr<xla::PjRtTopologyDescription> owned_topology;
  const xla::PjRtTopologyDescription* topology;
  std::vector<std::unique_ptr<const xla::PjRtDeviceDescription>>
      cpp_descriptions;
  std::vector<PJRT_DeviceDescription> descriptions;
  std::vector<PJRT_DeviceDescription*> description_pointers;
  std::vector<PJRT_NamedValue> attributes;
};

struct PJRT_Client {
  std::unique_ptr<xla::PjRtClient> client;
  std::vector<PJRT_Device> owned_devices;
  // `devices` contains the addresses of the contents of `owned_devices`.
  std::vector<PJRT_Device*> devices;
  // `addressable_devices` contains pointers to the `owned_devices` that the
  // client can issue commands to.
  std::vector<PJRT_Device*> addressable_devices;
  // Map from wrapped C++ devices to C devices. The values are the same as
  // `owned_devices`.
  absl::flat_hash_map<xla::PjRtDevice*, PJRT_Device*> c_device_from_cpp_device;
  // TODO(yueshengys): Add a `memories` member when global memories are
  // supported.
  std::vector<PJRT_Memory> owned_memories;
  // `addressable_memories` contains pointers to the `owned_memories` that the
  // client can transfer to and from.
  std::vector<PJRT_Memory*> addressable_memories;
  // Map from wrapped C++ memories to C memories. The values are the same as
  // `owned_memories`.
  absl::flat_hash_map<xla::PjRtMemorySpace*, PJRT_Memory*>
      c_memory_from_cpp_memory;
  absl::StatusOr<std::unique_ptr<PJRT_TopologyDescription>> topology;

  explicit PJRT_Client(std::unique_ptr<xla::PjRtClient> cpp_client);
};

// PJRT_DeviceDescriptions are owned by their corresponding PJRT_Device.
struct PJRT_DeviceDescription {
  // The xla::PjRtDeviceDescription* is owned transitively by the
  // corresponding xla::PjRtClient.
  const xla::PjRtDeviceDescription* device_description;
  // The device specific attributes which are initialized once per device.
  std::vector<PJRT_NamedValue> attributes;
};

// PJRT_Devices are owned by their corresponding PJRT_Client.
struct PJRT_Device {
  // The xla::PjRtDevice* is owned by the corresponding xla::PjRtClient.
  xla::PjRtDevice* device;
  PJRT_DeviceDescription description;
  std::vector<PJRT_Memory*> addressable_memories;
  PJRT_Client* client;
};

struct PJRT_Memory {
  // The xla::PjRtMemorySpace* is owned by the corresponding xla::PjRtClient.
  xla::PjRtMemorySpace* memory_space;
  std::vector<PJRT_Device*> devices;
  PJRT_Client* client;
};

struct PJRT_Executable {
  // Must be shared_ptr so that we can share with PJRT_LoadedExecutable.
  std::shared_ptr<xla::PjRtExecutable> executable;

  absl::StatusOr<std::string> fingerprint;

  // Used to synchronize concurrent setting of cached values.
  mutable absl::Mutex mutex;

  // Cost analysis properties and name strings are populated after cost analysis
  // has been run. These are returned from cost analysis calls, and do not
  // change after the first call.
  bool cost_analysis_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<std::string> cost_analysis_names;
  std::vector<PJRT_NamedValue> cost_analysis_properties;

  bool memory_kind_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<const char*> memory_kinds;
  std::vector<size_t> memory_kind_sizes;

  bool out_type_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<PJRT_Buffer_Type> out_types;

  bool out_dimension_ran ABSL_GUARDED_BY(mutex) = false;
  std::vector<int64_t> out_dimensions;
  std::vector<size_t> out_dimension_sizes;

  explicit PJRT_Executable(std::shared_ptr<xla::PjRtExecutable> executable);

  const xla::PjRtExecutable* get() const { return executable.get(); }
  xla::PjRtExecutable* get() { return executable.get(); }
};

struct PJRT_LoadedExecutable {
  // Must be shared_ptr so that we can share with PJRT_Executable.
  std::shared_ptr<xla::PjRtLoadedExecutable> executable;
  PJRT_Client* client;
  // These pointers are a subset of `client`'s `addressable_devices`, i.e. those
  // addressed by the compiled executable program. `client` owns the objects
  // these point to.
  std::vector<PJRT_Device*> addressable_devices;

  PJRT_LoadedExecutable(std::shared_ptr<xla::PjRtLoadedExecutable> executable,
                        PJRT_Client* client);

  const xla::PjRtLoadedExecutable* get() const { return executable.get(); }
  xla::PjRtLoadedExecutable* get() { return executable.get(); }
};

struct PJRT_Buffer {
  std::unique_ptr<xla::PjRtBuffer> buffer;
  PJRT_Client* client;
  // Set and cached the first time PJRT_Buffer_GetMemoryLayout is called.
  std::optional<pjrt::BufferMemoryLayoutData> layout_data;
  // Set and cached the first time PJRT_Buffer_UnpaddedDimensions is called.
  std::optional<std::vector<int64_t>> unpadded_dims;
  // Set and cached the first time PJRT_Buffer_DynamicDimensionIndices is
  // called.
  std::optional<std::vector<size_t>> dynamic_dim_indices;
  // Used to synchronize concurrent setting of cached values.
  absl::Mutex mu;
  // Manages, holds, and takes ownership of external references.
  std::vector<std::unique_ptr<xla::PjRtBuffer::ExternalReference>>
      external_references;
};

struct PJRT_Event {
  xla::PjRtFuture<> future;
  // TODO(b/333538339): It's safe to Await() on PjRtFuture<> multiple times,
  // remove this workaround.
  //
  // Set and stored upon future.Await(), as PjRtFuture only allows its result to
  // be queried through Await() and Await() can only safely be called once. This
  // variable allows C API users to check for error status any time after
  // Await() has been called.
  std::optional<absl::Status> status;
};

struct PJRT_SerializedExecutable {
  std::string serialized;
};

struct PJRT_SerializedTopology {
  std::string serialized;
};

struct PJRT_TransferMetadata {
  // Decompose xla::Shape into C API type fields, without any Tuple information.
  // TODO(b/238999986) support other `xla::Shape` fields when they are fully
  // implemented.
  xla::Shape device_shape;
};

struct PJRT_CopyToDeviceStream {
  std::unique_ptr<xla::CopyToDeviceStream> stream;
};

namespace pjrt {
// C API definitions

void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);
void PJRT_Error_Message(PJRT_Error_Message_Args* args);
PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args);

PJRT_Error* PJRT_Plugin_Attributes_Empty(PJRT_Plugin_Attributes_Args* args);
PJRT_Error* PJRT_Plugin_Attributes_Xla(PJRT_Plugin_Attributes_Args* args);

PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args);
PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args);
PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args);
PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args);
PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args);

PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);
PJRT_Error* PJRT_Client_PlatformName(PJRT_Client_PlatformName_Args* args);
PJRT_Error* PJRT_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args);
PJRT_Error* PJRT_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args);
PJRT_Error* PJRT_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args* args);
PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);
PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args);
PJRT_Error* PJRT_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args);
PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args);
PJRT_Error* PJRT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args);
PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args);
PJRT_Error* PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args);
PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args);
PJRT_Error* PJRT_Client_CreateViewOfDeviceBuffer(
    PJRT_Client_CreateViewOfDeviceBuffer_Args* args);

PJRT_Error* PJRT_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args);
PJRT_Error* PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args);
PJRT_Error* PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args);
PJRT_Error* PJRT_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args);
PJRT_Error* PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args);
PJRT_Error* PJRT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args);

PJRT_Error* PJRT_Device_GetDescription(PJRT_Device_GetDescription_Args* args);
PJRT_Error* PJRT_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args);
PJRT_Error* PJRT_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args);
PJRT_Error* PJRT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args);
PJRT_Error* PJRT_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args);
PJRT_Error* PJRT_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args);

PJRT_Error* PJRT_Memory_Id(PJRT_Memory_Id_Args* args);
PJRT_Error* PJRT_Memory_Kind(PJRT_Memory_Kind_Args* args);
PJRT_Error* PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args);
PJRT_Error* PJRT_Memory_DebugString(PJRT_Memory_DebugString_Args* args);
PJRT_Error* PJRT_Memory_ToString(PJRT_Memory_ToString_Args* args);
PJRT_Error* PJRT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args);

PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);
PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);
PJRT_Error* PJRT_Executable_NumReplicas(PJRT_Executable_NumReplicas_Args* args);
PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args);
PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args);
PJRT_Error* PJRT_Executable_NumOutputs(PJRT_Executable_NumOutputs_Args* args);
PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);
PJRT_Error* PJRT_Executable_Fingerprint(PJRT_Executable_Fingerprint_Args* args);
PJRT_Error* PJRT_Executable_GetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args* args);
PJRT_Error* PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args);
PJRT_Error* PJRT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args);
PJRT_Error* PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args);
PJRT_Error* PJRT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args);
PJRT_Error* PJRT_Executable_Serialize(PJRT_Executable_Serialize_Args* args);
PJRT_Error* PJRT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args);

PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args);
PJRT_Error* PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args);
PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args);
PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args);
PJRT_Error* PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args);
PJRT_Error* PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args);
// TODO: b/306669267 - this method is deprecated. Return unimplemented error,
// until the next major version upgrade.
PJRT_Error* PJRT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args);

PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);
PJRT_Error* PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args);
PJRT_Error* PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args);
PJRT_Error* PJRT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args);
PJRT_Error* PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args);
PJRT_Error* PJRT_Buffer_GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args* args);
PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args);
PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args);
PJRT_Error* PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args* args);
PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args);
PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);
PJRT_Error* PJRT_Buffer_CopyToDevice(PJRT_Buffer_CopyToDevice_Args* args);
PJRT_Error* PJRT_Buffer_CopyToMemory(PJRT_Buffer_CopyToMemory_Args* args);
PJRT_Error* PJRT_Buffer_ToHostBuffer(PJRT_Buffer_ToHostBuffer_Args* args);
PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);
PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);
PJRT_Error* PJRT_Buffer_UnsafePointer(PJRT_Buffer_UnsafePointer_Args* args);
PJRT_Error* PJRT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);
PJRT_Error* PJRT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);
PJRT_Error* PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);

PJRT_Error* PJRT_CopyToDeviceStream_Destroy(
    PJRT_CopyToDeviceStream_Destroy_Args* args);
PJRT_Error* PJRT_CopyToDeviceStream_AddChunk(
    PJRT_CopyToDeviceStream_AddChunk_Args* args);
PJRT_Error* PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args* args);
PJRT_Error* PJRT_CopyToDeviceStream_GranuleSize(
    PJRT_CopyToDeviceStream_GranuleSize_Args* args);
PJRT_Error* PJRT_CopyToDeviceStream_CurrentBytes(
    PJRT_CopyToDeviceStream_CurrentBytes_Args* args);

PJRT_Error* PJRT_TopologyDescription_Destroy(
    PJRT_TopologyDescription_Destroy_Args* args);
PJRT_Error* PJRT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args);
PJRT_Error* PJRT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args);
PJRT_Error* PJRT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args);
PJRT_Error* PJRT_TopologyDescription_Serialize(
    PJRT_TopologyDescription_Serialize_Args* args);
PJRT_Error* PJRT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args* args);

PJRT_Error* PJRT_Compile(PJRT_Compile_Args* args);

// Helper macros and functions

#define PJRT_RETURN_IF_ERROR(expr)                                \
  do {                                                            \
    absl::Status _status = (expr);                                \
    if (!_status.ok()) {                                          \
      PJRT_Error* _c_status = new PJRT_Error{std::move(_status)}; \
      return _c_status;                                           \
    }                                                             \
  } while (false)

#define PJRT_ASSIGN_OR_RETURN(lhs, rexpr)                                  \
  _PJRT_ASSIGN_OR_RETURN_IMPL(_PJRT_CONCAT(_status_or_value, __COUNTER__), \
                              lhs, rexpr,                                  \
                              _PJRT_CONCAT(_c_status, __COUNTER__));

#define _PJRT_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr, c_status) \
  auto statusor = (rexpr);                                          \
  if (!statusor.ok()) {                                             \
    PJRT_Error* c_status = new PJRT_Error();                        \
    c_status->status = statusor.status();                           \
    return c_status;                                                \
  }                                                                 \
  lhs = std::move(*statusor)

#define _PJRT_CONCAT(x, y) _PJRT_CONCAT_IMPL(x, y)
#define _PJRT_CONCAT_IMPL(x, y) x##y

// Returns a specific error message when the program format is unknown.
// Does not check the program format itself.
std::string ProgramFormatErrorMsg(absl::string_view program_format);

// Creates a C PJRT topology from a C++ PJRT topology.
//
// The returned topology is owned by the caller and should be destroyed with
// PJRT_TopologyDescription_Destroy. This can be used to implement functions
// like PJRT_TopologyDescription_Create that return an owned topo desc.
PJRT_TopologyDescription* CreateWrapperDeviceTopology(
    std::unique_ptr<xla::PjRtTopologyDescription> cpp_topology);

// Creates a C PJRT topology from a C++ PJRT topology.
//
// The returned topology is *not* owned by the caller and should *not* be
// destroyed with PJRT_TopologyDescription_Destroy. This can be used to
// implement functions like PJRT_Client_TopologyDescription that return a topo
// desc owned by something else.
PJRT_TopologyDescription* CreateWrapperDeviceTopology(
    const xla::PjRtTopologyDescription* cpp_topology);

// Creates a C PJRT client from a C++ PJRT client and creates C PJRT devices
// from cpp_client's devices. The returned client is owned by the caller and
// should be destroyed with PJRT_Client_Destroy.
PJRT_Client* CreateWrapperClient(std::unique_ptr<xla::PjRtClient> cpp_client);

// Helper functions for converting C key-value store callbacks to C++ callbacks.
std::shared_ptr<xla::KeyValueStoreInterface> ToCppKeyValueStore(
    PJRT_KeyValueGetCallback c_get_callback, void* get_user_arg,
    PJRT_KeyValuePutCallback c_put_callback, void* put_user_arg);

// A method that does not nothing other than returning a nullptr. Can be used as
// the implementation of PJRT_Plugin_Initialize for plugins that do not require
// specific initialization.
PJRT_Error* PJRT_Plugin_Initialize_NoOp(PJRT_Plugin_Initialize_Args* args);

// Creates a PJRT_Api with create_fn from the input and other functions in
// pjrt_c_api_wrapper_impl.
PJRT_Api CreatePjrtApi(PJRT_Client_Create* create_fn,
                       PJRT_TopologyDescription_Create* topology_create_fn,
                       PJRT_Plugin_Initialize* plugin_initialize_fn,
                       PJRT_Extension_Base* extension_start = nullptr,
                       PJRT_Plugin_Attributes* plugin_attributes_fn =
                           pjrt::PJRT_Plugin_Attributes_Empty);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_WRAPPER_IMPL_H_
