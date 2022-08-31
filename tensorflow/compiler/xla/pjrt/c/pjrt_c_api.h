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
#include "tensorflow/compiler/xla/stream_executor/tpu/c_api_decl.h"

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

// Codes are based on https://abseil.io/docs/cpp/guides/status-codes
typedef enum {
  PJRT_Error_Code_CANCELLED = 1,
  PJRT_Error_Code_UNKNOWN = 2,
  PJRT_Error_Code_INVALID_ARGUMENT = 3,
  PJRT_Error_Code_DEADLINE_EXCEEDED = 4,
  PJRT_Error_Code_NOT_FOUND = 5,
  PJRT_Error_Code_ALREADY_EXISTS = 6,
  PJRT_Error_Code_PERMISSION_DENIED = 7,
  PJRT_Error_Code_RESOURCE_EXHAUSTED = 8,
  PJRT_Error_Code_FAILED_PRECONDITION = 9,
  PJRT_Error_Code_ABORTED = 10,
  PJRT_Error_Code_OUT_OF_RANGE = 11,
  PJRT_Error_Code_UNIMPLEMENTED = 12,
  PJRT_Error_Code_INTERNAL = 13,
  PJRT_Error_Code_UNAVAILABLE = 14,
  PJRT_Error_Code_DATA_LOSS = 15,
  PJRT_Error_Code_UNAUTHENTICATED = 16
} PJRT_Error_Code;

typedef struct {
  size_t struct_size;
  void* priv;
  const PJRT_Error* error;
  PJRT_Error_Code code;  // out
} PJRT_Error_GetCode_Args;
const size_t PJRT_Error_GetCode_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Error_GetCode_Args, error);

typedef PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args);

// ---------------------------------- Events -----------------------------------

// Represents a notifying event that is returned by PJRT APIs that enqueue
// asynchronous work, informing callers when the work is complete and reporting
// a value of type `PJRT_Error*` or `nullptr` as error status.
//
// Callers are always responsible for freeing `PJRT_Event`s by calling
// `PJRT_Event_Destroy`.
typedef struct PJRT_Event PJRT_Event;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
} PJRT_Event_Destroy_Args;
const size_t PJRT_Event_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Event_Destroy_Args, event);

// Frees `event`. `event` can be `nullptr`.
typedef PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
  bool is_ready;  // out
} PJRT_Event_IsReady_Args;
const size_t PJRT_Event_IsReady_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Event_IsReady_Args, is_ready);

// Returns true if this PJRT_Event has completed, including if an error has
// occurred.
typedef PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
} PJRT_Event_Error_Args;
const size_t PJRT_Event_Error_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Event_Error_Args, event);

// Should only be called if PJRT_Event_IsReady returns true.
// Returns `nullptr` if there is no error.
// The returned error should be freed with `PJRT_Error_Destroy`.
//
// If `PJRT_Event_Await` has been called, this will return a pointer to an
// identical error status as that call, as will subsequent calls to
// `PJRT_Event_Error`. However, each of these `PJRT_Error *` pointers are
// independent of `PJRT_Error *`s returned by other function calls, so they must
// each be freed separately using `PJRT_Error_Destroy`.
typedef PJRT_Error* PJRT_Event_Error(PJRT_Event_Error_Args* args);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
} PJRT_Event_Await_Args;

const size_t PJRT_Event_Await_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Event_Await_Args, event);

// Blocks the calling thread until `event` is ready, then returns the error
// status (with `nullptr` indicating no error). The returned status should be
// freed with `PJRT_Error_Destroy`.
typedef PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args);

// A callback to be performed once an event is ready. It will be called on the
// event's error state and a pointer to an object of the caller's choice.
// Ownership of `error` is passed to the callback. The callback must destroy
// `error` via `PJRT_Error_Destroy`. The caller retains ownership of `user_arg`.
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
  PJRT_Event_OnReadyCallback callback;
  // `user_arg` allows `callback` to be called with arbitrary arguments (e.g.
  // via pointers in a struct cast to void*).
  void* user_arg;
} PJRT_Event_OnReady_Args;
const size_t PJRT_Event_OnReady_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Event_OnReady_Args, user_arg);

// Registers `callback` to be called once `event` is ready, with `event`'s
// error status and a pointer to an object of the caller's choice as arguments.
typedef PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args);

// ---------------------------------- Client -----------------------------------

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_Buffer PJRT_Buffer;

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

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int num_replicas;
  int num_partitions;
  // Must be greater than or equal to `num_replicas * num_partitions`
  size_t default_assignment_size;
  // Points to an array of size `default_assignment_size`.
  // This API writes `num_replicas * num_partitions` ints within that buffer.
  // The caller retains ownership of this memory.
  int* default_assignment;  // pointer to array in; values written as out
} PJRT_Client_DefaultDeviceAssignment_Args;

const size_t PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_DefaultDeviceAssignment_Args,
                     default_assignment);

typedef PJRT_Error* PJRT_Client_DefaultDeviceAssignment(
    PJRT_Client_DefaultDeviceAssignment_Args* args);

typedef enum {
  // Invalid primitive type to serve as default.
  PJRT_Buffer_Type_INVALID,

  // Predicates are two-state booleans.
  PJRT_Buffer_Type_PRED,

  // Signed integral values of fixed width.
  PJRT_Buffer_Type_S8,
  PJRT_Buffer_Type_S16,
  PJRT_Buffer_Type_S32,
  PJRT_Buffer_Type_S64,

  // Unsigned integral values of fixed width.
  PJRT_Buffer_Type_U8,
  PJRT_Buffer_Type_U16,
  PJRT_Buffer_Type_U32,
  PJRT_Buffer_Type_U64,

  // Floating-point values of fixed width.
  PJRT_Buffer_Type_F16,
  PJRT_Buffer_Type_F32,
  PJRT_Buffer_Type_F64,

  // Truncated 16 bit floating-point format. This is similar to IEEE's 16 bit
  // floating-point format, but uses 1 bit for the sign, 8 bits for the exponent
  // and 7 bits for the mantissa.
  PJRT_Buffer_Type_BF16,

  // Complex values of fixed width.
  //
  // Paired F32 (real, imag), as in std::complex<float>.
  PJRT_Buffer_Type_C64,
  // Paired F64 (real, imag), as in std::complex<double>.
  PJRT_Buffer_Type_C128,
} PJRT_Buffer_Type;

typedef enum {
  // The runtime may not hold references to `data` after the call to
  // `PJRT_Client_BufferFromHostBuffer` completes. The caller promises that
  // `data` is immutable and will not be freed only for the duration of the
  // PJRT_Client_BufferFromHostBuffer call.
  PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,

  // The PjRtBuffer may alias `data` internally and the runtime may use the
  // `data` contents as long as the buffer is alive. The caller promises to
  // keep `data` alive and not to mutate its contents as long as the buffer is
  // alive; to notify the caller that the buffer may be freed, the runtime
  // will call `done_with_host_buffer` when the PjRtBuffer is freed.
  PJRT_HostBufferSemantics_kZeroCopy,
} PJRT_HostBufferSemantics;

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // Pointer to the host buffer
  const void* data;
  // The type of the `data`, and the type of the resulting output `buffer`
  PJRT_Buffer_Type type;
  // The array dimensions of `data`.
  const int64_t* dims;
  size_t num_dims;
  // Number of bytes to traverse per dimension. Must be the same size as `dims`,
  // or empty. If empty, the array is assumed to have a dense layout with
  // dimensions in major-to-minor order
  // Caution: `byte_strides` are allowed to be negative, in which case `data`
  // may need to point to the interior of the buffer, not necessarily its start.
  const int64_t* byte_strides;
  size_t num_byte_strides;

  PJRT_HostBufferSemantics host_buffer_semantics;

  // Device to copy host data to.
  PJRT_Device* device;

  // Event indicating when it's safe to free `data`. The caller is responsible
  // for calling PJRT_Event_Destroy.
  PJRT_Event* done_with_host_buffer;  // out

  // Output device buffer. The caller is responsible for calling
  // PJRT_Buffer_Destroy.
  PJRT_Buffer* buffer;  // out
} PJRT_Client_BufferFromHostBuffer_Args;

const size_t PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Client_BufferFromHostBuffer_Args, buffer);

// Asynchronously copies a buffer stored on host to device memory.
typedef PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args);

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

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  size_t num_outputs;  // out
} PJRT_Executable_NumOutputs_Args;
const size_t PJRT_Executable_NumOutputs_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Executable_NumOutputs_Args, num_outputs);

// Gets the number of outputs per device produced by `executable`.
typedef PJRT_Error* PJRT_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args* args);

// ---------------------------------- Buffers ----------------------------------

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
} PJRT_Buffer_Destroy_Args;
const size_t PJRT_Buffer_Destroy_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_Destroy_Args, buffer);

// Deletes the underlying runtime objects as if 'PJRT_Buffer_Delete' were
// called and frees `buffer`. `buffer` can be nullptr.
typedef PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);

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
  bool has_layout;
  XLA_Layout layout;  // out
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
  PJRT_Buffer* src;

  // `dst` can be nullptr to query required size which will be set into
  // `dst_size`.
  void* dst;  // in/out
  // Size of `dst` in bytes. If `dst` is nullptr, then `dst_size` is set to the
  // size needed. Otherwise, `dst_size` must be greater than or equal to the
  // needed size.
  size_t dst_size;  // in/out

  // Event that signals when the copy has completed.
  PJRT_Event* event;  // out
} PJRT_Buffer_ToHostBuffer_Args;
const size_t PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_ToHostBuffer_Args, event);

// Asynchronously copies the buffer's value into a preallocated host buffer.
typedef PJRT_Error* PJRT_Buffer_ToHostBuffer(
    PJRT_Buffer_ToHostBuffer_Args* args);

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

typedef struct {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  // The caller is responsible for calling PJRT_Event_Destroy on `event`.
  PJRT_Event* event;  // out
} PJRT_Buffer_ReadyEvent_Args;
const size_t PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Buffer_ReadyEvent_Args, event);

// Returns an event that is triggered when either of the following happens:
// * the data in the PJRT_Buffer becomes ready, or
// * an error has occurred.
//
// TODO(b/241967811): change these weird semantics
// If the buffer has been deleted or donated, the returned event will
// immediately indicate an error. However, if PJRT_Buffer_ReadyEvent() is
// called on the buffer before PJRT_Buffer_Delete() is, the returned event will
// not transition to an error state after PJRT_Buffer_Delete() is called.
typedef PJRT_Error* PJRT_Buffer_ReadyEvent(PJRT_Buffer_ReadyEvent_Args* args);

// -------------------------------- API access ---------------------------------

#define _PJRT_API_STRUCT_FIELD(fn_type) fn_type* fn_type

// Please modify PJRT_Api_STRUCT_SIZE if the last field of PJRT_Api is changed.
typedef struct {
  size_t struct_size;
  void* priv;

  _PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_Message);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_GetCode);

  _PJRT_API_STRUCT_FIELD(PJRT_Event_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_IsReady);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_Error);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_Await);
  _PJRT_API_STRUCT_FIELD(PJRT_Event_OnReady);

  _PJRT_API_STRUCT_FIELD(PJRT_Client_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_PlatformVersion);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Devices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_LookupDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Compile);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_DefaultDeviceAssignment);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_BufferFromHostBuffer);

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
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumOutputs);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Execute);

  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceTrimmedShape);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Device);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ToHostBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsOnCpu);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ReadyEvent);
} PJRT_Api;

const size_t PJRT_Api_STRUCT_SIZE =
    PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Buffer_ReadyEvent);

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
