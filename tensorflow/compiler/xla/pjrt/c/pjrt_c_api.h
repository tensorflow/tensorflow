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

#define PJRT_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                        \
  const size_t sname##_STRUCT_SIZE = PJRT_STRUCT_SIZE(sname, last_field);

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------- Errors -----------------------------------

// PJRT C API methods generally return a PJRT_Error*, which is nullptr if there
// is no error and set if there is. The implementation allocates any returned
// PJRT_Errors, but the caller is always responsible for freeing them via
// PJRT_Error_Destroy.

typedef struct PJRT_Error PJRT_Error;

struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_Error* error;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Destroy_Args, error);

// Frees `error`. `error` can be nullptr.
typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);

struct PJRT_Error_Message_Args {
  size_t struct_size;
  void* priv;
  const PJRT_Error* error;
  // Has the lifetime of `error`.
  const char* message;  // out
  size_t message_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Message_Args, message_size);

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

struct PJRT_Error_GetCode_Args {
  size_t struct_size;
  void* priv;
  const PJRT_Error* error;
  PJRT_Error_Code code;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_GetCode_Args, error);

typedef PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args);

// ---------------------------------- Events -----------------------------------

// Represents a notifying event that is returned by PJRT APIs that enqueue
// asynchronous work, informing callers when the work is complete and reporting
// a value of type `PJRT_Error*` or `nullptr` as error status.
//
// Callers are always responsible for freeing `PJRT_Event`s by calling
// `PJRT_Event_Destroy`.
typedef struct PJRT_Event PJRT_Event;

struct PJRT_Event_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Destroy_Args, event);

// Frees `event`. `event` can be `nullptr`.
typedef PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args);

struct PJRT_Event_IsReady_Args {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
  bool is_ready;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_IsReady_Args, is_ready);

// Returns true if this PJRT_Event has completed, including if an error has
// occurred.
typedef PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args);

struct PJRT_Event_Error_Args {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Error_Args, event);

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

struct PJRT_Event_Await_Args {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Await_Args, event);

// Blocks the calling thread until `event` is ready, then returns the error
// status (with `nullptr` indicating no error). The returned status should be
// freed with `PJRT_Error_Destroy`.
typedef PJRT_Error* PJRT_Event_Await(PJRT_Event_Await_Args* args);

// A callback to be performed once an event is ready. It will be called on the
// event's error state and a pointer to an object of the caller's choice.
// Ownership of `error` is passed to the callback. The callback must destroy
// `error` via `PJRT_Error_Destroy`. The caller retains ownership of `user_arg`.
typedef void (*PJRT_Event_OnReadyCallback)(PJRT_Error* error, void* user_arg);

struct PJRT_Event_OnReady_Args {
  size_t struct_size;
  void* priv;
  PJRT_Event* event;
  PJRT_Event_OnReadyCallback callback;
  // `user_arg` allows `callback` to be called with arbitrary arguments (e.g.
  // via pointers in a struct cast to void*).
  void* user_arg;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_OnReady_Args, user_arg);

// Registers `callback` to be called once `event` is ready, with `event`'s
// error status and a pointer to an object of the caller's choice as arguments.
typedef PJRT_Error* PJRT_Event_OnReady(PJRT_Event_OnReady_Args* args);

// ------------------------ Other Common Data Types ----------------------------

typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64,
  PJRT_NamedValue_kInt64List,
  PJRT_NamedValue_kFloat,
} PJRT_NamedValue_Type;

// Named value for key-value pairs.
struct PJRT_NamedValue {
  size_t struct_size;
  void* priv;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
  };
  // `value_size` is the number of elements for array/string and 1 for scalar
  // values.
  size_t value_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_NamedValue, value_size);

// ---------------------------------- Client -----------------------------------

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Buffer PJRT_Buffer;

struct PJRT_Client_Create_Args {
  size_t struct_size;
  void* priv;
  // Extra platform-specific options to create a client.
  PJRT_NamedValue* create_options;
  size_t num_options;
  PJRT_Client* client;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Create_Args, client);

// Creates and initializes a new PJRT_Client and returns in `client`.
typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);

struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Destroy_Args, client);

// Shuts down and frees `client`. `client` can be nullptr.
typedef PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);

struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_name` has the same lifetime as `client`. It is owned by `client`.
  const char* platform_name;  // out
  size_t platform_name_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_PlatformName_Args, platform_name_size);

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
typedef PJRT_Error* PJRT_Client_PlatformName(
    PJRT_Client_PlatformName_Args* args);

struct PJRT_Client_ProcessIndex_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int process_index;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_ProcessIndex_Args, process_index);

// Return the process index of this client. Always 0 in single-process
// settings.
typedef PJRT_Error* PJRT_Client_ProcessIndex(
    PJRT_Client_ProcessIndex_Args* args);

struct PJRT_Client_PlatformVersion_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // `platform_version` has the same lifetime as `client`. It's owned by
  // `client`.
  const char* platform_version;  // out
  size_t platform_version_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_PlatformVersion_Args,
                          platform_version_size);

// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
typedef PJRT_Error* PJRT_Client_PlatformVersion(
    PJRT_Client_PlatformVersion_Args* args);

struct PJRT_Client_Devices_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  PJRT_Device** devices;  // out
  size_t num_devices;     // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Devices_Args, num_devices);

// Returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
typedef PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);

struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  PJRT_Device** addressable_devices;  // out
  size_t num_addressable_devices;     // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_AddressableDevices_Args,
                          num_addressable_devices);

// Returns a list of devices that are addressable from the client.
// Addressable devices are those that the client can issue commands to.
// All devices are addressable in a single-process environment.
typedef PJRT_Error* PJRT_Client_AddressableDevices(
    PJRT_Client_AddressableDevices_Args* args);

struct PJRT_Client_LookupDevice_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int id;
  // `device` has the same lifetime as `client`. It is owned by `client`.
  PJRT_Device* device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupDevice_Args, device);

// Returns a PJRT_Device* with the specified ID as returned by PJRT_Device_Id.
typedef PJRT_Error* PJRT_Client_LookupDevice(
    PJRT_Client_LookupDevice_Args* args);

struct PJRT_Client_LookupAddressableDevice_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  int local_hardware_id;
  // `addressable_device` has the same lifetime as `client`. It is owned by
  // `client`.
  PJRT_Device* addressable_device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupAddressableDevice_Args,
                          addressable_device);

// Returns an addressable PJRT_Device* with the specified ID as returned by
// PJRT_Device_LocalHardwareId.
typedef PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args);

struct PJRT_Program {
  size_t struct_size;
  void* priv;
  // Serialized code in the specified format below.
  // String is owned by the caller.
  char* code;  // in/out depending on usage
  size_t code_size;
  // Supported formats are:
  // "hlo": code string takes serialized HloModuleProto.
  // "hlo_with_config": code string takes serialized HloModuleProtoWithConfig.
  // "mlir": code string takes MLIR module bytecode (or string).
  // Ownership of `format` varies across API functions.
  const char* format;
  size_t format_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Program, format_size);

struct PJRT_Client_Compile_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  // Only needs to stay alive for the duration of the Compile call.
  // `program->format` and `program->format_size` are owned by the caller.
  PJRT_Program* program;
  // TODO(b/240560013): consider putting some of option fields in priv.
  // Serialized CompileOptionsProto
  // (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/compile_options.proto)
  const char* compile_options;
  size_t compile_options_size;
  PJRT_LoadedExecutable* executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Compile_Args, executable);

// Compiles a program in specified format (such as MLIR or HLO) with given
// `options`.
typedef PJRT_Error* PJRT_Client_Compile(PJRT_Client_Compile_Args* args);

struct PJRT_Client_DefaultDeviceAssignment_Args {
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
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_DefaultDeviceAssignment_Args,
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

  // The runtime may hold onto `data` after the call to
  // `PJRT_Client_BufferFromHostBuffer`
  // returns while the runtime completes a transfer to the device. The caller
  // promises not to mutate or free `data` until the transfer completes, at
  // which point `done_with_host_buffer` will be triggered.
  PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes,

  // The PjRtBuffer may alias `data` internally and the runtime may use the
  // `data` contents as long as the buffer is alive. The caller promises to
  // keep `data` alive and not to mutate its contents as long as the buffer is
  // alive; to notify the caller that the buffer may be freed, the runtime
  // will call `done_with_host_buffer` when the PjRtBuffer is freed.
  PJRT_HostBufferSemantics_kZeroCopy,
} PJRT_HostBufferSemantics;

struct PJRT_Client_BufferFromHostBuffer_Args {
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
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_BufferFromHostBuffer_Args, buffer);

// Asynchronously copies a buffer stored on host to device memory.
typedef PJRT_Error* PJRT_Client_BufferFromHostBuffer(
    PJRT_Client_BufferFromHostBuffer_Args* args);

// --------------------------------- Devices -----------------------------------

struct PJRT_Device_Id_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_Id_Args, id);

// The ID of this device. IDs are unique among devices of this type
// (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
// hosts' devices.
typedef PJRT_Error* PJRT_Device_Id(PJRT_Device_Id_Args* args);

struct PJRT_Device_LocalHardwareId_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int local_hardware_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_LocalHardwareId_Args, local_hardware_id);

// Opaque hardware ID, e.g., the CUDA device number. In general, not guaranteed
// to be dense, and -1 if undefined.
typedef PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args);

struct PJRT_Device_ProcessIndex_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  int process_index;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_ProcessIndex_Args, process_index);

// The index of the process that this device belongs to, i.e. is addressable
// from. This is not always identical to PJRT_Client_ProcessIndex in a
// multi-process setting, where each client can see devices from all
// processes, but only a subset of them are addressable and have the same
// process_index as the client.
typedef PJRT_Error* PJRT_Device_ProcessIndex(
    PJRT_Device_ProcessIndex_Args* args);

struct PJRT_Device_IsAddressable_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  bool is_addressable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_IsAddressable_Args, is_addressable);

// Whether client can issue command to this device.
typedef PJRT_Error* PJRT_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args* args);

struct PJRT_Device_Attributes_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  size_t num_attributes;        // out
  PJRT_NamedValue* attributes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_Attributes_Args, attributes);

// Returns an array of device specific attributes with attribute name, value
// and value type.
typedef PJRT_Error* PJRT_Device_Attributes(PJRT_Device_Attributes_Args* args);

struct PJRT_Device_Kind_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  // `device_kind` string is owned by `device` and has same lifetime as
  // `device`.
  const char* device_kind;  // out
  size_t device_kind_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_Kind_Args, device_kind_size);

// A vendor-dependent string that uniquely identifies the kind of device,
// e.g., "Tesla V100-SXM2-16GB".
typedef PJRT_Error* PJRT_Device_Kind(PJRT_Device_Kind_Args* args);

struct PJRT_Device_DebugString_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  const char* debug_string;  // out
  size_t debug_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_DebugString_Args, debug_string_size);

// Debug string suitable for logging when errors occur. Should be verbose
// enough to describe the current device unambiguously.
typedef PJRT_Error* PJRT_Device_DebugString(PJRT_Device_DebugString_Args* args);

struct PJRT_Device_ToString_Args {
  size_t struct_size;
  void* priv;
  PJRT_Device* device;
  const char* to_string;  // out
  size_t to_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_ToString_Args, to_string_size);

// Debug string suitable for reading by end users, should be reasonably terse,
// for example: "CpuDevice(id=0)".
typedef PJRT_Error* PJRT_Device_ToString(PJRT_Device_ToString_Args* args);

// ------------------------------- Executables ---------------------------------

struct PJRT_Executable_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Destroy_Args, executable);

// Frees `executable`. `executable` can be nullptr.
typedef PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);

struct PJRT_LoadedExecutable_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Destroy_Args, executable);

// Frees `executable` and deletes the underlying runtime object as if
// `PJRT_LoadedExecutable_Delete` were called. `executable` can be nullptr.
typedef PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args);

struct PJRT_LoadedExecutable_GetExecutable_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* loaded_executable;
  PJRT_Executable* executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_GetExecutable_Args, executable);

// Constructs a PJRT_Executable from a PJRT_LoadedExecutable. The returned
// executable should be freed by the caller with PJRT_Executable_Destroy.
typedef PJRT_Error* PJRT_LoadedExecutable_GetExecutable(
    PJRT_LoadedExecutable_GetExecutable_Args* args);

struct PJRT_Executable_Name_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  // `executable_name` has the same lifetime as `executable`. It is owned by
  // `executable`.
  const char* executable_name;  // out
  size_t executable_name_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Name_Args, executable_name_size);

// Returns a string that identifies the executable.
typedef PJRT_Error* PJRT_Executable_Name(PJRT_Executable_Name_Args* args);

// TODO(b/269178731): Revisit whether num_replicas is needed.
struct PJRT_Executable_NumReplicas_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  size_t num_replicas;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumReplicas_Args, num_replicas);

// Returns the number of replicas of the executable.
typedef PJRT_Error* PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args);

struct PJRT_Executable_NumPartitions_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  size_t num_partitions;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumPartitions_Args, num_partitions);

// Returns the number of partitions of the executable.
typedef PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args);

struct PJRT_LoadedExecutable_AddressableDevices_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
  PJRT_Device** addressable_devices;  // out
  size_t num_addressable_devices;     // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_AddressableDevices_Args,
                          num_addressable_devices);

// Returns a list of devices this executable will run on.
typedef PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args);

struct PJRT_Executable_OptimizedProgram_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  PJRT_Program* program;  // out, but read below
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OptimizedProgram_Args, program);

// Retrieves the optimized program for a given PJRT_Executable (SPMD).
// The caller should populate `program->format` and `format_size`.
//
// The implementation will set `program->format` and `program->format_size`
// to inform callers of the format of the optimized program returned.
// These members are owned by the implementation.
//
// If called with nullptr as `program->code`, `PJRT_Executable_OptimizedProgram`
// will populate `program->code_size` as an output indicating the number of
// bytes the string `program->code` requires.
//
// If `program->code` is not null, `PJRT_Executable_OptimizedProgram` will fill
// the buffer pointed to by `program->code` with the serialization of the
// optimized HLO program. `program->code` must point to a client-owned buffer of
// size >= `program->code_size`, which must be at large enough to hold the
// serialization of the optimized program.
//
// Callers should generally call this function twice with the same `args`.
// In the first call, `program->code` must be nullptr. This call will populate
// `program->code_size`. Clients should then allocate a buffer `code_buff` of at
// least `code_size` bytes. Before the second call, callers should set
// `program->code = code_buff`. The second call will then write the serialized
// program to `code_buff`.
typedef PJRT_Error* PJRT_Executable_OptimizedProgram(
    PJRT_Executable_OptimizedProgram_Args* args);

struct PJRT_LoadedExecutable_Delete_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Delete_Args, executable);

// Drops `executable`'s reference to the internal runtime object and
// associated resources, without freeing the `executable` object itself.
// `executable` can only be used with PJRT_LoadedExecutable_IsDeleted and
// PJRT_LoadedExecutable_Destroy after calling this method. The internal runtime
// executable will be freed after the last execution completes.
typedef PJRT_Error* PJRT_LoadedExecutable_Delete(
    PJRT_LoadedExecutable_Delete_Args* args);

struct PJRT_LoadedExecutable_IsDeleted_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
  bool is_deleted;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_IsDeleted_Args, is_deleted);

// True if and only if PJRT_LoadedExecutable_Delete has previously been called.
typedef PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args);

struct PJRT_Chunk {
  void* data;
  size_t size;
  void (*deleter)(void* data, void* deleter_arg);
  // `deleter_arg` will be passed to `deleter` as `deleter_arg` argument.
  void* deleter_arg;
};

// TODO(b/263390934) implement C API that calls `AddChunk` and other
// `xla::CopyToDeviceStream`.
typedef struct PJRT_CopyToDeviceStream PJRT_CopyToDeviceStream;

struct PJRT_TransferMetadata;

// Returns PJRT_Error* with an error status. The status carries a callback's
// error status code and message.
typedef PJRT_Error* (*PJRT_CallbackError)(PJRT_Error_Code code,
                                          const char* message,
                                          size_t message_size);

// Returns PJRT_Error* created by PJRT_CallbackError in case of error.
// Otherwise, returns nullptr. The callback must call
// `chunk->deleter(chunk->data, chunk->deleter_arg)` when it's finished with
// `chunk`.
typedef PJRT_Error* (*PJRT_SendCallback)(PJRT_Chunk* chunk,
                                         PJRT_CallbackError* callback_error,
                                         size_t total_size_in_bytes, bool done,
                                         void* user_arg);
typedef void (*PJRT_RecvCallback)(PJRT_CopyToDeviceStream* stream,
                                  void* user_arg);

struct PJRT_SendCallbackInfo {
  // Used to associate this callback with the correct send op.
  int64_t channel_id;
  // Will be passed to `send_callback` as `user_arg` argument.
  void* user_arg;
  PJRT_SendCallback send_callback;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_SendCallbackInfo, send_callback);

struct PJRT_RecvCallbackInfo {
  // Used to associate this callback with the correct recv op.
  int64_t channel_id;
  // Will be passed to `recv_callback` as `user_arg` argument.
  void* user_arg;
  PJRT_RecvCallback recv_callback;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_RecvCallbackInfo, recv_callback);

struct PJRT_ExecuteOptions {
  size_t struct_size;
  void* priv;
  // Callbacks for when send/recv ops are executed. The outer lists correspond
  // to each device returned by `PJRT_Executable_AddressableDevices` for
  // `executable` (i.e. they will have length `num_devices`). Each inner list
  // contains callback info for each send/recv op in `executable`; the order
  // doesn't matter as the channel IDs are used instead. The callbacks can be
  // stateful and the user code is responsible for managing state. The callback
  // functions must outlive the execution (but not the info structs or lists).
  PJRT_SendCallbackInfo** send_callbacks;
  PJRT_RecvCallbackInfo** recv_callbacks;
  size_t num_send_ops = 0;
  size_t num_recv_ops = 0;
  // If non-zero, identifies this execution as part of a potentially
  // multi-device launch. This can be used to detect scheduling errors, e.g. if
  // multi-host programs are launched in different orders on different hosts,
  // the launch IDs may be used by the runtime to detect the mismatch.
  int launch_id;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteOptions, launch_id);

struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
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
  // If `device_complete_events` isn't nullptr, `device_complete_events` needs
  // to be the same length as `output_lists` (i.e. of length `num_devices`), and
  // each `PJRT_Event` will become ready once the corresponding device execution
  // is complete. If Execute returns an error, then `device_complete_events`
  // will not be populated. The caller is responsible for calling
  // PJRT_Event_Destroy on the returned PJRT_Event*s.
  PJRT_Event** device_complete_events;  // in/out
  // The device to execute on. If nullptr, will execute on the device(s)
  // specified at compile time. If set, must be an addressable device, and
  // `num_devices` should be 1 with `argument_lists` only containing arguments
  // for `execute_device`. Can be set with a multi-device executable to launch
  // just on this device. In this case, it's the responsibility of the caller to
  // make sure the executable is launched on all participating devices specified
  // at compile time. Setting this field may not be supported on all platforms
  // or executables.
  PJRT_Device* execute_device;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Execute_Args, execute_device);

// Executes on devices addressable by the client.
typedef PJRT_Error* PJRT_LoadedExecutable_Execute(
    PJRT_LoadedExecutable_Execute_Args* args);

struct PJRT_Executable_NumOutputs_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  size_t num_outputs;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumOutputs_Args, num_outputs);

// Gets the number of outputs per device produced by `executable`.
typedef PJRT_Error* PJRT_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args* args);

struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
  size_t struct_size;
  void* priv;
  PJRT_Executable* executable;
  int64_t size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args,
                          size_in_bytes);  // last field in the struct

typedef PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);

struct PJRT_LoadedExecutable_GetCostAnalysis_Args {
  size_t struct_size;
  void* priv;
  PJRT_LoadedExecutable* executable;
  size_t num_properties;  // out
  // `properties` and any embedded data are owned by and have the same lifetime
  // as `executable`.
  PJRT_NamedValue* properties;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_GetCostAnalysis_Args,
                          properties);

// Get the cost properties for the executable. Different platforms may return
// different properties; for example, some platforms may return the number of
// operations, or memory size of the input/output of the executable, based on
// program analysis.
typedef PJRT_Error* PJRT_LoadedExecutable_GetCostAnalysis(
    PJRT_LoadedExecutable_GetCostAnalysis_Args* args);

typedef struct PJRT_SerializedExecutable PJRT_SerializedExecutable;

struct PJRT_Executable_Serialize_Args {
  size_t struct_size;
  void* priv;
  const PJRT_Executable* executable;
  PJRT_SerializedExecutable* serialized_executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Serialize_Args,
                          serialized_executable);

// Returns a platform-specific serialization of `executable`. The serialization
// is not guaranteed to be stable over time.
typedef PJRT_Error* PJRT_Executable_Serialize(
    PJRT_Executable_Serialize_Args* args);

struct PJRT_Executable_DeserializeAndLoad_Args {
  size_t struct_size;
  void* priv;
  PJRT_Client* client;
  const char* serialized_executable;
  size_t serialized_executable_size;
  PJRT_LoadedExecutable* loaded_executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_DeserializeAndLoad_Args,
                          loaded_executable);

// Deserializes an executable serialized by `PJRT_Executable_Serialize`.
// `serialized_executable` must have been produced by the same platform and
// library version as this one.
typedef PJRT_Error* PJRT_Executable_DeserializeAndLoad(
    PJRT_Executable_DeserializeAndLoad_Args* args);

// -------------------------- Serialized Executables ---------------------------

struct PJRT_SerializedExecutable_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_SerializedExecutable* serialized_executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_SerializedExecutable_Destroy_Args,
                          serialized_executable);

// Destroys a `PJRT_SerializedExecutable`.
typedef PJRT_Error* PJRT_SerializedExecutable_Destroy(
    PJRT_SerializedExecutable_Destroy_Args* args);

// The string pointed to by `data` is owned by `serialized_executable` and has
// the same object lifetime.
struct PJRT_SerializedExecutable_Data_Args {
  size_t struct_size;
  void* priv;
  PJRT_SerializedExecutable* serialized_executable;
  const char* data;  // out
  size_t data_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_SerializedExecutable_Data_Args, data_size);

// Returns the data of a `PJRT_SerializedExecutable` and its length in bytes
typedef PJRT_Error* PJRT_SerializedExecutable_Data(
    PJRT_SerializedExecutable_Data_Args* args);

// ---------------------------------- Buffers ----------------------------------

struct PJRT_Buffer_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Destroy_Args, buffer);

// Deletes the underlying runtime objects as if 'PJRT_Buffer_Delete' were
// called and frees `buffer`. `buffer` can be nullptr.
typedef PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);

// This trimmed shape doesn't have any Tuple information. In case of Tuple,
// assert is triggered from the C API  Client.
// TODO(b/238999986): This is a temporary solution. Remove this later.
struct PJRT_Buffer_OnDeviceTrimmedShape_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  int element_type;             // out
  Int64List dimensions;         // out
  BoolList dynamic_dimensions;  // out
  bool has_layout;
  // Whether it calls logical_on_device_shape.
  bool is_logical_on_device_shape;
  XLA_Layout layout;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OnDeviceTrimmedShape_Args, layout);

// Return the trimmed shape from PjRtBuffer.
// TODO(b/238999986): Replace this with decomposed shape methods.
typedef PJRT_Error* PJRT_Buffer_OnDeviceTrimmedShape(
    PJRT_Buffer_OnDeviceTrimmedShape_Args* args);

struct PJRT_Buffer_ToHostBuffer_Args {
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
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ToHostBuffer_Args, event);

// Asynchronously copies the buffer's value into a preallocated host buffer.
typedef PJRT_Error* PJRT_Buffer_ToHostBuffer(
    PJRT_Buffer_ToHostBuffer_Args* args);

struct PJRT_Buffer_OnDeviceSizeInBytes_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  size_t on_device_size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OnDeviceSizeInBytes_Args,
                          on_device_size_in_bytes);

// Gets the number of bytes of the buffer storage on the device
typedef PJRT_Error* PJRT_Buffer_OnDeviceSizeInBytes(
    PJRT_Buffer_OnDeviceSizeInBytes_Args* args);

struct PJRT_Buffer_Delete_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Delete_Args, buffer);

// Drop the buffer's reference to its associated device memory, without freeing
// the `buffer` object itself. `buffer` can only be used with
// PJRT_Buffer_IsDeleted and PJRT_Buffer_Destroy after calling this method. The
// device memory will be freed when all async operations using the buffer have
// completed, according to the allocation semantics of the underlying platform.
typedef PJRT_Error* PJRT_Buffer_Delete(PJRT_Buffer_Delete_Args* args);

struct PJRT_Buffer_IsDeleted_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  bool is_deleted;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsDeleted_Args, is_deleted);

// True if and only if PJRT_Buffer_Delete has previously been called.
typedef PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);

struct PJRT_Buffer_CopyToDevice_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  PJRT_Device* dst_device;
  PJRT_Buffer* dst_buffer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyToDevice_Args, dst_buffer);

// Copies the buffer to device `dst_device`. Caller is responsible for freeing
// returned `dst_buffer` with PJRT_Buffer_Destroy. Returns an error if the
// buffer is already on `dst_device`.
typedef PJRT_Error* PJRT_Buffer_CopyToDevice(
    PJRT_Buffer_CopyToDevice_Args* args);

struct PJRT_Buffer_IsOnCpu_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  bool is_on_cpu;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsOnCpu_Args, is_on_cpu);

// Whether this buffer is on CPU and thus allows for certain optimizations.
typedef PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);

struct PJRT_Buffer_Device_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  PJRT_Device* device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Device_Args, device);

// Returns this buffer's storage device.
typedef PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args);

struct PJRT_Buffer_ReadyEvent_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  // The caller is responsible for calling PJRT_Event_Destroy on `event`.
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ReadyEvent_Args, event);

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

struct PJRT_Buffer_UnsafePointer_Args {
  size_t struct_size;
  void* priv;
  PJRT_Buffer* buffer;
  uintptr_t buffer_pointer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_UnsafePointer_Args, buffer_pointer);

// Returns platform-dependent address for the given buffer that is often but
// not guaranteed to be the physical/device address.
typedef PJRT_Error* PJRT_Buffer_UnsafePointer(
    PJRT_Buffer_UnsafePointer_Args* args);

// ---------------------------- CopyToDeviceStream -----------------------------

struct PJRT_CopyToDeviceStream_AddChunk_Args {
  size_t struct_size;
  void* priv;
  PJRT_CopyToDeviceStream* stream;
  // Takes ownership of `chunk` (i.e. implementation will call chunk.deleter).
  PJRT_Chunk* chunk;
  PJRT_Event* transfer_complete;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_AddChunk_Args, chunk);

// Emplaces a new chunk of data to copy to the device. The transfer is started
// immediately, and the returned event is triggered when the transfer completes
// or fails.
//
// The returned event will indicate an error if the chunk's size causes the
// amount of transferred data to exceed the total bytes, if the stream is
// already complete, or if the chunk is not a multiple of the granule size.
typedef PJRT_Error* PJRT_CopyToDeviceStream_AddChunk(
    PJRT_CopyToDeviceStream_AddChunk_Args* args);

struct PJRT_CopyToDeviceStream_TotalBytes_Args {
  size_t struct_size;
  void* priv;
  PJRT_CopyToDeviceStream* stream;
  int64_t total_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_TotalBytes_Args, total_bytes);

// Returns the total amount of data the stream expects to be transferred.
typedef PJRT_Error* PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args* args);

struct PJRT_CopyToDeviceStream_GranuleSize_Args {
  size_t struct_size;
  void* priv;
  PJRT_CopyToDeviceStream* stream;
  int64_t granule_size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_GranuleSize_Args,
                          granule_size_in_bytes);

// Returns the granule size in bytes. The size of the chunk added to this stream
// must be a multiple of this number.
typedef PJRT_Error* PJRT_CopyToDeviceStream_GranuleSize(
    PJRT_CopyToDeviceStream_GranuleSize_Args* args);

struct PJRT_CopyToDeviceStream_CurrentBytes_Args {
  size_t struct_size;
  void* priv;
  PJRT_CopyToDeviceStream* stream;
  int64_t current_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_CurrentBytes_Args,
                          current_bytes);

// Returns the amount of data the stream currently has either transferred or has
// buffered to transfer.
typedef PJRT_Error* PJRT_CopyToDeviceStream_CurrentBytes(
    PJRT_CopyToDeviceStream_CurrentBytes_Args* args);

// ------------------------------ Device Topology ------------------------------

typedef struct PJRT_DeviceTopology PJRT_DeviceTopology;

struct PJRT_DeviceTopology_Create_Args {
  size_t struct_size;
  void* priv;
  PJRT_DeviceTopology* topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceTopology_Create_Args, topology);

// Creates and initializes a new PJRT_DeviceTopology and returns in `topology`.
typedef PJRT_Error* PJRT_DeviceTopology_Create(
    PJRT_DeviceTopology_Create_Args* args);

struct PJRT_DeviceTopology_Destroy_Args {
  size_t struct_size;
  void* priv;
  PJRT_DeviceTopology* topology;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceTopology_Destroy_Args, topology);

// Frees `topology`. `topology` can be nullptr.
typedef PJRT_Error* PJRT_DeviceTopology_Destroy(
    PJRT_DeviceTopology_Destroy_Args* args);

struct PJRT_DeviceTopology_PlatformVersion_Args {
  size_t struct_size;
  void* priv;
  PJRT_DeviceTopology* topology;
  // `platform_version` has the same lifetime as `topology`. It's owned by
  // `topology`.
  const char* platform_version;  // out
  size_t platform_version_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceTopology_PlatformVersion_Args,
                          platform_version_size);

// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
typedef PJRT_Error* PJRT_DeviceTopology_PlatformVersion(
    PJRT_DeviceTopology_PlatformVersion_Args* args);

struct PJRT_DeviceTopology_PlatformName_Args {
  size_t struct_size;
  void* priv;
  PJRT_DeviceTopology* topology;
  // `platform_name` has the same lifetime as `topology`. It is owned by
  // `topology`.
  const char* platform_name;  // out
  size_t platform_name_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceTopology_PlatformName_Args,
                          platform_name_size);

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
typedef PJRT_Error* PJRT_DeviceTopology_PlatformName(
    PJRT_DeviceTopology_PlatformName_Args* args);

struct PJRT_Compile_Args {
  size_t struct_size;
  void* priv;
  const PJRT_DeviceTopology* topology;
  // Only needs to stay alive for the duration of the Compile call.
  // `program->format` and `program->format_size` are owned by the caller.
  PJRT_Program* program;
  // TODO(b/240560013): consider putting some of option fields in priv.
  // Serialized CompileOptionsProto
  // (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/compile_options.proto)
  const char* compile_options;
  size_t compile_options_size;
  // Optionally provided for performance-guided optimizations.
  PJRT_Client* client;
  PJRT_Executable* executable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Compile_Args, executable);

// Compiles a program in specified format (such as MLIR or HLO) with given
// `options`. The returned executable must be loaded by a compatible
// PJRT_Client before execution.
typedef PJRT_Error* PJRT_Compile(PJRT_Compile_Args* args);

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
  _PJRT_API_STRUCT_FIELD(PJRT_Client_LookupAddressableDevice);
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
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumReplicas);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumPartitions);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumOutputs);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_SizeOfGeneratedCodeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OptimizedProgram);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Serialize);

  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_GetExecutable);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_GetCostAnalysis);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Execute);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_DeserializeAndLoad);

  _PJRT_API_STRUCT_FIELD(PJRT_SerializedExecutable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_SerializedExecutable_Data);

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
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_UnsafePointer);

  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_AddChunk);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_TotalBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_GranuleSize);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_CurrentBytes);

  _PJRT_API_STRUCT_FIELD(PJRT_DeviceTopology_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceTopology_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceTopology_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceTopology_PlatformVersion);

  _PJRT_API_STRUCT_FIELD(PJRT_Compile);
} PJRT_Api;

const size_t PJRT_Api_STRUCT_SIZE = PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Compile);

#undef _PJRT_API_STRUCT_FIELD
#undef PJRT_DEFINE_STRUCT_TRAITS

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_C_PJRT_C_API_H_
