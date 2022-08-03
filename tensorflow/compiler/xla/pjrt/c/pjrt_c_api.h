/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_

#include <stddef.h>
#include <stdint.h>

// TODO(b/238999986): Remove this.
#include "tensorflow/stream_executor/tpu/c_api_decl.h"

#define PJRT_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------- Errors -----------------------------------

// PJRT C API methods generally return a PJRT_Error*, which is nullptr if there
// is no error and set if there is. The implementation allocates any returned
// PJRT_Errors, but the caller is always responsible for freeing them via
// PJRT_Error_Destroy.

typedef struct PJRT_Error PJRT_Error;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Error* error;
} PJRT_Error_Destroy_Args;
const size_t PJRT_Error_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Error_Destroy_Args, error);

// Frees `error`. `error` can be nullptr.
typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  const PJRT_Error* error;
  // Has the lifetime of `error`.
  const char* message;  // out
  size_t message_size;  // out
} PJRT_Error_Message_Args;
const size_t PJRT_Error_Message_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Error_Message_Args, message_size);

// Gets the human-readable reason for `error`. `message` has the lifetime of
// `error`.
typedef void PJRT_Error_Message(PJRT_Error_Message_Args* args);

// ---------------------------------- Client -----------------------------------

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Executable PJRT_Executable;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;  // out
} PJRT_Client_Create_Args;
const size_t PJRT_Client_Create_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Create_Args, client);

// Creates and initializes a new PJRT_Client and returns in `client`.
typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
} PJRT_Client_Destroy_Args;
const size_t PJRT_Client_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Destroy_Args, client);

// Shuts down and frees `client`. `client` can be nullptr.
typedef PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_name` has the same lifetime as `client`. It is owned by `client`.
  const char* platform_name;  // out
  size_t platform_name_size;  // out
} PJRT_Client_PlatformName_Args;

const size_t PJRT_Client_PlatformName_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_PlatformName_Args, platform_name_size);

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
typedef PJRT_Error* PJRT_Client_PlatformName(
    PJRT_Client_PlatformName_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int process_index;  // out
} PJRT_Client_ProcessIndex_Args;
const size_t PJRT_Client_ProcessIndex_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_ProcessIndex_Args, process_index);

// Return the process index of this client. Always 0 in single-process
// settings.
typedef PJRT_Error* PJRT_Client_ProcessIndex(
    PJRT_Client_ProcessIndex_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_version` has the same lifetime as `client`. It's owned by
  // `client`.
  const char* platform_version;  // out
  size_t platform_version_size;  // out
} PJRT_Client_PlatformVersion_Args;

const size_t PJRT_Client_PlatformVersion_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_PlatformVersion_Args, platform_version_size);

// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
typedef PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  PJRT_Device** devices;  // out
  size_t num_devices;     // out
} PJRT_Client_Devices_Args;
const size_t PJRT_Client_Devices_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Devices_Args, num_devices);

// Returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
typedef PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  PJRT_Device** addressable_devices;  // out
  size_t num_addressable_devices;     // out
} PJRT_Client_AddressableDevices_Args;
const size_t PJRT_Client_AddressableDevices_Args_STRUCT_SIZE = PJRT_STRUCT_SIZE(
    PJRT_Client_AddressableDevices_Args, num_addressable_devices);

// Returns a list of devices that are addressable from the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment.
typedef PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int id;
  // `device` has the same lifetime as `client`. It is owned by `client`.
  PJRT_Device* device;  // out
} PJRT_Client_LookupDevice_Args;

const size_t PJRT_Client_LookupDevice_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_LookupDevice_Args, device);

// Returns a PJRT_Device* with the specified ID as returned by PJRT_Device_Id.
typedef PJRT_Error* PJRT_Client_LookupDevice(
    PJRT_Client_LookupDevice_Args* args);

// TODO(jieying): add debug_option.
// TODO(b/240560013): consider putting some of option fields in priv.
typedef struct {
  size_t struct_size;
  void* priv;
  // If true, the supplied module expects its arguments to be wrapped in a
  // tuple and passed as a single parameter.
  bool parameter_is_tupled_arguments;
  // If set, this is the device to build the computation for. A value of -1
  // indicates this option has not been set.
  int device_ordinal;
  // The number of replicas of this computation that are to be executed.
  int num_replicas;
  // The number of partitions in this computation.
  int num_partitions;
  // Whether to use SPMD (true) or MPMD (false) partitioning when
  // num_partitions > 1 and XLA is requested to partition the input program.
  bool use_spmd_partitioning;
  // Whether to allow sharding propagation to propagate to the outputs.
  bool allow_spmd_sharding_propagation_to_output;
  const char* device_assignment;  // Serialized device assignment.
  size_t device_assignment_size;
} PJRT_CompileOptions;
const size_t PJRT_CompileOptions_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_CompileOptions, device_assignment_size);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // Serialized MLIR module. Only needs to stay alive for the duration of the
  // Compile call.
  const char* module;
  size_t module_size;
  // Only needs to stay alive for the duration of the Compile call.
  PJRT_CompileOptions* options;
  PJRT_Executable* executable;  // out
} PJRT_Client_Compile_Args;

const size_t PJRT_Client_Compile_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_Compile_Args, executable);

// Compiles an MLIR module with given `options`.
typedef PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args);

// --------------------------------- Devices -----------------------------------

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int id;  // out
} PJRT_Device_Id_Args;
const size_t PJRT_Device_Id_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_Id_Args, id);

// The ID of this device. IDs are unique among devices of this type
// (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
// hosts' devices.
typedef PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int local_hardware_id;  // out
} PJRT_Device_LocalHardwareId_Args;
const size_t PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_LocalHardwareId_Args, local_hardware_id);

// Opaque hardware ID, e.g., the CUDA device number. In general, not guaranteed
// to be dense, and -1 if undefined.
typedef PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int process_index;  // out
} PJRT_Device_ProcessIndex_Args;
const size_t PJRT_Device_ProcessIndex_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_ProcessIndex_Args, process_index);

// The index of the process that this device belongs to, i.e. is addressable
// from. This is not always identical to PJRT_Client_ProcessIndex in a
// multi-process setting, where each client can see devices from all
// processes, but only a subset of them are addressable and have the same
// process_index as the client.
typedef PJRT_Error* PJRT_Device_ProcessIndex(
    PJRT_Device_ProcessIndex_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  bool is_addressable;  // out
} PJRT_Device_IsAddressable_Args;
const size_t PJRT_Device_IsAddressable_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_IsAddressable_Args, is_addressable);

// Whether client can issue command to this device.
typedef PJRT_Error* PJRT_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  const char* name;
  size_t name_size;
  enum {
    PJRT_Device_Attribute_kString = 0,
    PJRT_Device_Attribute_kInt64,
    PJRT_Device_Attribute_kInt64List
  } type;
  union {
    int64_t int64_value;
    const int64_t* int64_array_value;
    const char* string_value;
  };
  // `value_size` is the number of elements for array/string and 1 for scalar
  // values.
  size_t value_size;
} PJRT_Device_Attribute;
const size_t PJRT_Device_Attribute_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_Attribute, value_size);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  size_t num_attributes;              // out
  PJRT_Device_Attribute* attributes;  // out
} PJRT_Device_Attributes_Args;
const size_t PJRT_Device_Attributes_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_Attributes_Args, attributes);

// Returns an array of device specific attributes with attribute name, value
// and value type.
typedef PJRT_Error* PJRT_Device_Attributes(PJRT_Device_Attributes_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  // `device_kind` string is owned by `device` and has same lifetime as
  // `device`.
  const char* device_kind;  // out
  size_t device_kind_size;  // out
} PJRT_Device_Kind_Args;
const size_t PJRT_Device_Kind_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_Kind_Args, device_kind_size);

// A vendor-dependent string that uniquely identifies the kind of device,
// e.g., "Tesla V100-SXM2-16GB".
typedef PJRT_Error* PJRT_Device_Kind(PJRT_Device_Kind_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  const char* debug_string;  // out
  size_t debug_string_size;  // out
} PJRT_Device_DebugString_Args;
const size_t PJRT_Device_DebugString_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_DebugString_Args, debug_string_size);

// Debug string suitable for logging when errors occur. Should be verbose
// enough to describe the current device unambiguously.
typedef PJRT_Error* PJRT_Device_DebugString(PJRT_Device_DebugString_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  const char* to_string;  // out
  size_t to_string_size;  // out
} PJRT_Device_ToString_Args;
const size_t PJRT_Device_ToString_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Device_ToString_Args, to_string_size);

// Debug string suitable for reading by end users, should be reasonably terse,
// for example: "CpuDevice(id=0)".
typedef PJRT_Error* PJRT_Device_ToString(PJRT_Device_ToString_Args* args);

// ------------------------------- Executables ---------------------------------

typedef struct PJRT_Buffer PJRT_Buffer;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
} PJRT_Executable_Destroy_Args;
const size_t PJRT_Executable_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_Destroy_Args, executable);

// Frees `executable` and deletes the underlying runtime object as if
// `PJRT_Executable_Delete` were called. `executable` can be nullptr.
typedef PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  // `executable_name` has the same lifetime as `executable`. It is owned by
  // `executable`.
  const char* executable_name;  // out
  size_t executable_name_size;  // out
} PJRT_Executable_Name_Args;

const size_t PJRT_Executable_Name_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_Name_Args, executable_name_size);

// Returns a string that identifies the executable.
typedef PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  PJRT_Device** addressable_devices;  // out
  size_t num_addressable_devices;     // out
} PJRT_Executable_AddressableDevices_Args;

const size_t PJRT_Executable_AddressableDevices_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_AddressableDevices_Args,
                     num_addressable_devices);

// Returns a list of devices this executable will run on.
typedef PJRT_Error* PJRT_Executable_AddressableDevices(
    PJRT_Executable_AddressableDevices_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
} PJRT_Executable_Delete_Args;
const size_t PJRT_Executable_Delete_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_Delete_Args, executable);

// Drops `executable`'s reference to the internal runtime object and
// associated resources, without freeing the `executable` object itself.
// `executable` can only be used with PJRT_Executable_IsDeleted and
// PJRT_Executable_Destroy after calling this method. The internal runtime
// executable will be freed after the last execution completes.
typedef PJRT_Error* PJRT_Executable_Delete(PJRT_Executable_Delete_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  bool is_deleted;  // out
} PJRT_Executable_IsDeleted_Args;
const size_t PJRT_Executable_IsDeleted_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_IsDeleted_Args, is_deleted);

// True if and only if PJRT_Executable_Delete has previously been called.
typedef PJRT_Error* PJRT_Executable_IsDeleted(
    PJRT_Executable_IsDeleted_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  // If non-zero, identifies this execution as part of a potentially
  // multi-device launch. This can be used to detect scheduling errors, e.g. if
  // multi-host programs are launched in different orders on different hosts,
  // the launch IDs may be used by the runtime to detect the mismatch.
  int launch_id;
} PJRT_ExecuteOptions;
const size_t PJRT_ExecuteOptions_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_ExecuteOptions, launch_id);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  // Only needs to stay alive for the duration of the Execute call.
  PJRT_ExecuteOptions* options;
  // Execution input of size [`num_devices`, `num_args`].
  PJRT_Buffer*** argument_lists;
  size_t num_devices;
  size_t num_args;
  // Execution output of size [`num_devices`, num_outputs`], where `num_outputs`
  // is the number of outputs returned by this executable per device. Both the
  // outer (`PJRT_Buffer***`) and inner lists (`PJRT_Buffer**`) must be
  // allocated and deallocated by the caller. PJRT_Buffer_Destroy must be called
  // on the output PJRT_Buffer*.
  PJRT_Buffer*** output_lists;  // in/out
} PJRT_Executable_Execute_Args;
const size_t PJRT_Executable_Execute_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_Execute_Args, output_lists);

// Executes on devices addressable by the client.
typedef PJRT_Error* PJRT_Executable_Execute(PJRT_Executable_Execute_Args* args);

// ---------------------------------- Buffers ----------------------------------

// This trimmed shape doesn't have any Tuple information. In case of Tuple,
// assert is triggered from the C API  Client.
// TODO(b/238999986): This is a temporary solution. Remove this later.
typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  int element_type;             // out
  Int64List dimensions;         // out
  BoolList dynamic_dimensions;  // out
  XLA_Layout layout;            // out
} PJRT_Buffer_OnDeviceTrimmedShape_Args;
const size_t PJRT_Buffer_OnDeviceTrimmedShape_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_OnDeviceTrimmedShape_Args, layout);

// Return the trimmed shape from PjRtBuffer.
// TODO(b/238999986): Replace this with decomposed shape methods.
typedef PJRT_Error* PJRT_Buffer_OnDeviceTrimmedShape(
    PJRT_Buffer_OnDeviceTrimmedShape_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  size_t on_device_size_in_bytes;  // out
} PJRT_Buffer_OnDeviceSizeInBytes_Args;
const size_t PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_OnDeviceSizeInBytes_Args,
                     on_device_size_in_bytes);

// Gets the number of bytes of the buffer storage on the device
typedef PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
} PJRT_Buffer_Delete_Args;
const size_t PJRT_Buffer_Delete_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_Delete_Args, buffer);

// Drop the buffer's reference to its associated device memory, without freeing
// the `buffer` object itself. `buffer` can only be used with
// PJRT_Buffer_IsDeleted and PJRT_Buffer_Destroy after calling this method. The
// device memory will be freed when all async operations using the buffer have
// completed, according to the allocation semantics of the underlying platform.
typedef PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  bool is_deleted;  // out
} PJRT_Buffer_IsDeleted_Args;
const size_t PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_IsDeleted_Args, is_deleted);

// True if and only if PJRT_Buffer_Delete has previously been called.
typedef PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  PJRT_Device* dst_device;
  PJRT_Buffer* dst_buffer;  // out
} PJRT_Buffer_CopyToDevice_Args;
const size_t PJRT_Buffer_CopyToDevice_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_CopyToDevice_Args, dst_buffer);

// Copies the buffer to device `dst_device`. Caller is responsible for freeing
// returned `dst_buffer` with PJRT_Buffer_Destroy. Returns an error if the
// buffer is already on `dst_device`.
typedef PJRT_Error* PJRT_Buffer_CopyToDevice(
    PJRT_Buffer_CopyToDevice_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  bool is_on_cpu;  // out
} PJRT_Buffer_IsOnCpu_Args;
const size_t PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_IsOnCpu_Args, is_on_cpu);

// Whether this buffer is on CPU and thus allows for certain optimizations.
typedef PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  PJRT_Device* device;  // out
} PJRT_Buffer_Device_Args;
const size_t PJRT_Buffer_Device_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_Device_Args, device);

// Returns this buffer's storage device.
typedef PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args);

// -------------------------------- API access ---------------------------------

#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

// Please modify PJRT_Api_STRUCT_SIZE if the last field of PJRT_Api is changed.
typedef struct {
  size_t struct_size;
  void* priv;

  _PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_Message);

  _PJRT_API_STRUCT_FIELD(PJRT_Client_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformVersion);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Devices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_LookupDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Compile);

  _PJRT_API_STRUCT_FIELD(PJRT_Device_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_IsAddressable);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_Attributes);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_Kind);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_LocalHardwareId);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_DebugString);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_ToString);

  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Name);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Execute);

  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceTrimmedShape);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Device);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsOnCpu);
} PJRT_Api;

const size_t PJRT_Api_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Buffer_IsOnCpu);

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
