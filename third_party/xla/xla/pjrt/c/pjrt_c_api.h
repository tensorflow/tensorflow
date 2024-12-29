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

#ifndef XLA_PJRT_C_PJRT_C_API_H_
#define XLA_PJRT_C_PJRT_C_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Read more on C API ABI versioning and compatibility here:
// https://docs.google.com/document/d/1TKB5NyGtdzrpgw5mpyFjVAhJjpSNdF31T6pjPl_UT2o/edit?usp=sharing

#define PJRT_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)

// Must update PJRT_DEFINE_STRUCT_TRAITS with the new `last_field` after
// adding a new member to a struct.
#define PJRT_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                        \
  enum { sname##_STRUCT_SIZE = PJRT_STRUCT_SIZE(sname, last_field) }

#ifdef __cplusplus
extern "C" {
#endif

// ------------------------------- Extensions ----------------------------------

typedef enum {
  PJRT_Extension_Type_Gpu_Custom_Call = 0,
  PJRT_Extension_Type_Profiler,
  PJRT_Extension_Type_Custom_Partitioner,
  PJRT_Extension_Type_Stream,
  PJRT_Extension_Type_Layouts,
  PJRT_Extension_Type_FFI,
  PJRT_Extension_Type_MemoryDescriptions,
} PJRT_Extension_Type;

// PJRT_Extension_Base contains a type and a pointer to next
// PJRT_Extension_Base. The framework can go through this chain to find an
// extension and identify it with the type.
typedef struct PJRT_Extension_Base {
  size_t struct_size;
  PJRT_Extension_Type type;
  struct PJRT_Extension_Base* next;
} PJRT_Extension_Base;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Extension_Base, next);

// --------------------------------- Version -----------------------------------

// Incremented when an ABI-incompatible change is made to the interface.
// Changes include:
// * Deleting a method or argument
// * Changing the type of an argument
// * Rearranging fields in the PJRT_Api or argument structs
#define PJRT_API_MAJOR 0

// Incremented when the interface is updated in a way that is potentially
// ABI-compatible with older versions, if supported by the caller and/or
// implementation.
//
// Callers can implement forwards compatibility by using PJRT_Api_Version to
// check if the implementation is aware of newer interface additions.
//
// Implementations can implement backwards compatibility by using the
// `struct_size` fields to detect how many struct fields the caller is aware of.
//
// Changes include:
// * Adding a new field to the PJRT_Api or argument structs
// * Renaming a method or argument (doesn't affect ABI)
#define PJRT_API_MINOR 60

// The plugin should set the major_version and minor_version of
// PJRT_Api.pjrt_api_version to be the `PJRT_API_MAJOR` and `PJRT_API_MINOR` in
// this header that the implementation was compiled with.
struct PJRT_Api_Version {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  int major_version;  // out
  int minor_version;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Api_Version, minor_version);

// ---------------------------------- Errors -----------------------------------

// PJRT C API methods generally return a PJRT_Error*, which is nullptr if there
// is no error and set if there is. The implementation allocates any returned
// PJRT_Errors, but the caller is always responsible for freeing them via
// PJRT_Error_Destroy.

typedef struct PJRT_Error PJRT_Error;

struct PJRT_Error_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Error* error;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_Destroy_Args, error);

// Frees `error`. `error` can be nullptr.
typedef void PJRT_Error_Destroy(PJRT_Error_Destroy_Args* args);

struct PJRT_Error_Message_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  const PJRT_Error* error;
  PJRT_Error_Code code;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Error_GetCode_Args, code);

typedef PJRT_Error* PJRT_Error_GetCode(PJRT_Error_GetCode_Args* args);

// Function for PJRT implementation to pass to callback functions provided by
// caller so the callback can create a PJRT_Error* on error (to return to the
// implementation). `message` is only required to live for the
// PJRT_CallbackError call, i.e. the PJRT_CallbackError implementation must copy
// `message` into the PJRT_Error.
typedef PJRT_Error* (*PJRT_CallbackError)(PJRT_Error_Code code,
                                          const char* message,
                                          size_t message_size);

// ---------------------------- Named Values -----------------------------------

typedef enum {
  PJRT_NamedValue_kString = 0,
  PJRT_NamedValue_kInt64,
  PJRT_NamedValue_kInt64List,
  PJRT_NamedValue_kFloat,
  PJRT_NamedValue_kBool,
} PJRT_NamedValue_Type;

// Named value for key-value pairs.
struct PJRT_NamedValue {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* name;
  size_t name_size;
  PJRT_NamedValue_Type type;
  union {
    const char* string_value;
    int64_t int64_value;
    const int64_t* int64_array_value;
    float float_value;
    bool bool_value;
  };
  // `value_size` is the number of elements for array/string and 1 for scalar
  // values.
  size_t value_size;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_NamedValue, value_size);

// ---------------------------------- Plugin -----------------------------------

struct PJRT_Plugin_Initialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Plugin_Initialize_Args, extension_start);

// One-time plugin setup. Must be called before any other functions are called.
typedef PJRT_Error* PJRT_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args);

struct PJRT_Plugin_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // Returned attributes have the lifetime of the process.
  const PJRT_NamedValue* attributes;  // out
  size_t num_attributes;              // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Plugin_Attributes_Args, attributes);

// Returns an array of plugin attributes which are key-value pairs. Common keys
// include `xla_version`, `stablehlo_current_version`, and
// `stablehlo_minimum_version`.
typedef PJRT_Error* PJRT_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args);

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
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_Destroy_Args, event);

// Frees `event`. `event` can be `nullptr`.
typedef PJRT_Error* PJRT_Event_Destroy(PJRT_Event_Destroy_Args* args);

struct PJRT_Event_IsReady_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Event* event;
  bool is_ready;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Event_IsReady_Args, is_ready);

// Returns true if this PJRT_Event has completed, including if an error has
// occurred.
typedef PJRT_Error* PJRT_Event_IsReady(PJRT_Event_IsReady_Args* args);

struct PJRT_Event_Error_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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

// ---------------------------------- Client -----------------------------------

typedef struct PJRT_Client PJRT_Client;
typedef struct PJRT_Device PJRT_Device;
typedef struct PJRT_Memory PJRT_Memory;
typedef struct PJRT_ShapeSpec PJRT_ShapeSpec;
typedef struct PJRT_DeviceDescription PJRT_DeviceDescription;
typedef struct PJRT_TopologyDescription PJRT_TopologyDescription;
typedef struct PJRT_Executable PJRT_Executable;
typedef struct PJRT_LoadedExecutable PJRT_LoadedExecutable;
typedef struct PJRT_Buffer PJRT_Buffer;
typedef struct PJRT_AsyncHostToDeviceTransferManager
    PJRT_AsyncHostToDeviceTransferManager;

// The caller of PJRT_Client_Create can optionally provide a key-value store
// accessible across nodes and/or processes. KV store access may be necessary to
// create some multi-node/multi-process clients. The caller can provide the two
// callbacks below to access the key-value store.

// A callback to delete the value returned by PJRT_KeyValueGetCallback.
typedef void (*PJRT_KeyValueGetCallback_ValueDeleter)(char* value);

struct PJRT_KeyValueGetCallback_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* key;
  size_t key_size;
  int timeout_in_ms;
  PJRT_CallbackError* callback_error;
  void* user_arg;
  char* value;        // out
  size_t value_size;  // out
  // The caller needs to set a PJRT_KeyValueGetCallback_ValueDeleter to delete
  // the value returned by PJRT_KeyValueGetCallback. The implementation is
  // responsible for copying `value` and then calling value_deleter_callback.
  PJRT_KeyValueGetCallback_ValueDeleter value_deleter_callback;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_KeyValueGetCallback_Args,
                          value_deleter_callback);

// Requirements for PJRT_KeyValueGetCallback implementation: (1) Thread-safe.
// (2) The caller that provides the two callbacks is responsible for avoiding
// key collisions between different users of key-value store (i.e. between
// different plugins, but not between different nodes in one plugin). (3)
// Blocking.
typedef PJRT_Error* (*PJRT_KeyValueGetCallback)(
    PJRT_KeyValueGetCallback_Args* args);

struct PJRT_KeyValuePutCallback_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* key;
  size_t key_size;
  // Only needs to stay alive for the duration of the PJRT_KeyValuePutCallback
  // call.
  const char* value;
  size_t value_size;
  PJRT_CallbackError* callback_error;
  void* user_arg;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_KeyValuePutCallback_Args, user_arg);

// Requirements for PJRT_KeyValuePutCallback implementation: (1) Thread-safe.
// (2) The caller that provides the two callbacks is responsible for avoiding
// key collisions between different users of key-value store (i.e. between
// different plugins, but not between different nodes in one plugin).
typedef PJRT_Error* (*PJRT_KeyValuePutCallback)(
    PJRT_KeyValuePutCallback_Args* args);

struct PJRT_Client_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // Extra platform-specific options to create a client.
  const PJRT_NamedValue* create_options;
  size_t num_options;
  // Key-value get/put callback provided by the caller of PJRT_Client_Create.
  // PJRT client can use these callbacks to share information between
  // processes/nodes.
  PJRT_KeyValueGetCallback kv_get_callback;
  // Will be passed to `kv_get_callback` as `user_arg` argument.
  void* kv_get_user_arg;
  PJRT_KeyValuePutCallback kv_put_callback;
  // Will be passed to `kv_put_callback` as `user_arg` argument.
  void* kv_put_user_arg;

  PJRT_Client* client;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Create_Args, client);

// Creates and initializes a new PJRT_Client and returns in `client`.
typedef PJRT_Error* PJRT_Client_Create(PJRT_Client_Create_Args* args);

struct PJRT_Client_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Destroy_Args, client);

// Shuts down and frees `client`. `client` can be nullptr.
typedef PJRT_Error* PJRT_Client_Destroy(PJRT_Client_Destroy_Args* args);

struct PJRT_Client_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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

struct PJRT_Client_TopologyDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  // Is owned by and has the same lifetime as `client`.
  PJRT_TopologyDescription* topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_TopologyDescription_Args, topology);

// Returns the topology description of the runtime topology. The returned
// topology is owned by the client and should not be deleted by the caller.
typedef PJRT_Error* PJRT_Client_TopologyDescription(
    PJRT_Client_TopologyDescription_Args* args);

struct PJRT_Client_Devices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* devices;  // out
  size_t num_devices;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_Devices_Args, num_devices);

// Returns a list of all devices visible to the runtime, including addressable
// and non-addressable devices.
typedef PJRT_Error* PJRT_Client_Devices(PJRT_Client_Devices_Args* args);

struct PJRT_Client_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Device* const* addressable_devices;  // out
  size_t num_addressable_devices;           // out
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
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int id;
  // `device` has the same lifetime as `client`. It is owned by `client`.
  PJRT_Device* device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupDevice_Args, device);

// Returns a PJRT_Device* with the specified ID as returned by
// PJRT_DeviceDescription_Id.
typedef PJRT_Error* PJRT_Client_LookupDevice(
    PJRT_Client_LookupDevice_Args* args);

struct PJRT_Client_LookupAddressableDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  int local_hardware_id;
  // `addressable_device` has the same lifetime as `client`. It is owned by
  // `client`.
  PJRT_Device* addressable_device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_LookupAddressableDevice_Args,
                          addressable_device);

// Returns an addressable PJRT_Device* with the specified ID as returned by
// PJRT_DeviceDescription_LocalHardwareId.
typedef PJRT_Error* PJRT_Client_LookupAddressableDevice(
    PJRT_Client_LookupAddressableDevice_Args* args);

struct PJRT_Client_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_Memory* const* addressable_memories;  // out
  size_t num_addressable_memories;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_AddressableMemories_Args,
                          num_addressable_memories);

// Returns a list of memories that are addressable from the client. Addressable
// memories are those that the client can directly transfer data to and from.
// All memories are addressable in a single-process environment.
typedef PJRT_Error* PJRT_Client_AddressableMemories(
    PJRT_Client_AddressableMemories_Args* args);

struct PJRT_Program {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  // Only needs to stay alive for the duration of the Compile call.
  // `program->format` and `program->format_size` are owned by the caller.
  const PJRT_Program* program;
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
  PJRT_Extension_Base* extension_start;
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

struct PJRT_AsyncHostToDeviceTransferManager_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_AsyncHostToDeviceTransferManager_Destroy_Args,
                          transfer_manager);

// Frees `transfer_manager`. `transfer_manager` can be nullptr.
typedef PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_Destroy(
    PJRT_AsyncHostToDeviceTransferManager_Destroy_Args* args);

struct PJRT_AsyncHostToDeviceTransferManager_TransferData_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_AsyncHostToDeviceTransferManager* transfer_manager;
  int buffer_index;
  const void* data;
  int64_t offset;
  int64_t transfer_size;
  bool is_last_transfer;
  PJRT_Event* done_with_h2d_transfer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args,
    done_with_h2d_transfer);
typedef PJRT_Error* PJRT_AsyncHostToDeviceTransferManager_TransferData(
    PJRT_AsyncHostToDeviceTransferManager_TransferData_Args* args);

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

  // Truncated 8 bit floating-point formats.
  PJRT_Buffer_Type_F8E5M2,
  PJRT_Buffer_Type_F8E4M3FN,
  PJRT_Buffer_Type_F8E4M3B11FNUZ,
  PJRT_Buffer_Type_F8E5M2FNUZ,
  PJRT_Buffer_Type_F8E4M3FNUZ,

  // 4-bit integer types
  PJRT_Buffer_Type_S4,
  PJRT_Buffer_Type_U4,

  PJRT_Buffer_Type_TOKEN,

  // 2-bit integer types
  PJRT_Buffer_Type_S2,
  PJRT_Buffer_Type_U2,

  // More truncated 8 bit floating-point formats.
  PJRT_Buffer_Type_F8E4M3,
  PJRT_Buffer_Type_F8E3M4,
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
  // `data` contents as long as the buffer is alive. The runtime promises not
  // to mutate contents of the buffer (i.e. it will not use it for aliased
  // output buffers). The caller promises to keep `data` alive and not to mutate
  // its contents as long as the buffer is alive; to notify the caller that the
  // buffer may be freed, the runtime will call `done_with_host_buffer` when the
  // PjRtBuffer is freed.
  PJRT_HostBufferSemantics_kImmutableZeroCopy,

  // The PjRtBuffer may alias `data` internally and the runtime may use the
  // `data` contents as long as the buffer is alive. The runtime is allowed
  // to mutate contents of the buffer (i.e. use it for aliased output
  // buffers). The caller promises to keep `data` alive and not to mutate its
  // contents as long as the buffer is alive (otherwise it could be a data
  // race with the runtime); to notify the caller that the buffer may be
  // freed, the runtime will call `on_done_with_host_buffer` when the
  // PjRtBuffer is freed. On non-CPU platforms this acts identically to
  // kImmutableUntilTransferCompletes.
  PJRT_HostBufferSemantics_kMutableZeroCopy,
} PJRT_HostBufferSemantics;

typedef enum {
  PJRT_Buffer_MemoryLayout_Type_Tiled = 0,
  PJRT_Buffer_MemoryLayout_Type_Strides,
} PJRT_Buffer_MemoryLayout_Type;

struct PJRT_Buffer_MemoryLayout_Tiled {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // A map from physical dimension numbers to logical dimension numbers.
  // The first element is the most minor physical dimension (fastest varying
  // index) and the last the most major (slowest varying index). The contents of
  // the vector are the indices of the *logical* dimensions in the shape. Must
  // be the same size as the number of dimensions of the buffer.
  const int64_t* minor_to_major;
  size_t minor_to_major_size;
  // A concatenated list of tile dimensions.
  const int64_t* tile_dims;
  // The list of tile dimension sizes. The size of this list is `num_tiles`.
  const size_t* tile_dim_sizes;
  size_t num_tiles;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout_Tiled, num_tiles);

struct PJRT_Buffer_MemoryLayout_Strides {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  // Number of bytes to traverse per dimension. Must be the same size as
  // the number of dimensions of the data. Caution: `byte_strides` are allowed
  // to be negative, in which case data may need to point to the interior of
  // the buffer, not necessarily its start.
  const int64_t* byte_strides;
  size_t num_byte_strides;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout_Strides, num_byte_strides);

// Describe the memory layout. It can be (1) a list of minor-to-major order and
// optional tilings (each tile is a list of dimensions), or (2) a list of
// strides.
struct PJRT_Buffer_MemoryLayout {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  union {
    PJRT_Buffer_MemoryLayout_Tiled tiled;
    PJRT_Buffer_MemoryLayout_Strides strides;
  };
  PJRT_Buffer_MemoryLayout_Type type;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_MemoryLayout, type);

struct PJRT_Client_BufferFromHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  // Pointer to the host buffer
  const void* data;
  // The type of the `data`, and the type of the resulting output `buffer`
  PJRT_Buffer_Type type;
  // The array dimensions of `data`.
  const int64_t* dims;
  size_t num_dims;

  // Number of bytes to traverse per dimension of the input data. Must be the
  // same size as `dims`, or empty. If empty, the array is assumed to have a
  // dense layout with dimensions in major-to-minor order
  // Caution: `byte_strides` are allowed to be negative, in which case `data`
  // may need to point to the interior of the buffer, not necessarily its start.
  const int64_t* byte_strides;
  size_t num_byte_strides;

  PJRT_HostBufferSemantics host_buffer_semantics;

  // Device to copy host data to.
  PJRT_Device* device;

  // If nullptr, host data will be copied to `device`, otherwise we copy data to
  // `memory`.
  PJRT_Memory* memory;

  // The caller is responsible to keep the data (tiled or strides) in the
  // device_layout alive during the call. If nullptr, the device layout is
  // assumed to be a dense layout with dimensions in major-to-minor order.
  PJRT_Buffer_MemoryLayout* device_layout;

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

struct PJRT_Client_CreateViewOfDeviceBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  // A pointer to a non-owned device buffer. A PJRT_Buffer that is a non-owned
  // view of this device buffer will be created.
  void* device_buffer_ptr;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Buffer_Type element_type;
  PJRT_Buffer_MemoryLayout* layout;
  // The device that `device_buffer_ptr` is on.
  PJRT_Device* device;
  // A callback to be performed when the PJRT_Buffer is done with the on-device
  // buffer. This callback is optional and can be a nullptr.
  void (*on_delete_callback)(void* device_buffer_ptr, void* user_arg);
  // `on_delete_callback_arg` will be passed to `on_delete_callback` as
  // `user_arg` argument.
  void* on_delete_callback_arg;
  // A platform-specific stream handle that should contain the work or events
  // needed to materialize the on-device buffer. It is optional and can be
  // casted from a nullptr. PJRT_Client_CreateViewOfDeviceBuffer_Args will
  // append an event to `stream` that indicates when the returned buffer is
  // ready to use. This is intended to support dlpack on GPU and is not expected
  // to be supported on all hardware platforms.
  intptr_t stream;
  PJRT_Buffer* buffer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_CreateViewOfDeviceBuffer_Args, buffer);

// Creates a PJRT buffer that is a non-owned view of an on-device buffer
// (typically allocated by another library). The buffer may be mutated,
// for example, if the buffer is donated to an Execute operation. This method is
// not required on all hardware platforms.
typedef PJRT_Error* PJRT_Client_CreateViewOfDeviceBuffer(
    PJRT_Client_CreateViewOfDeviceBuffer_Args* args);

struct PJRT_ShapeSpec {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const int64_t* dims;
  size_t num_dims;
  PJRT_Buffer_Type element_type;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ShapeSpec, element_type);

struct PJRT_Client_CreateBuffersForAsyncHostToDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Client* client;
  PJRT_ShapeSpec* shape_specs;
  size_t num_shape_specs;
  PJRT_Buffer_MemoryLayout** device_layouts;  // optional
  size_t num_device_layouts;
  PJRT_Memory* memory;
  PJRT_AsyncHostToDeviceTransferManager* transfer_manager;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_CreateBuffersForAsyncHostToDevice_Args,
                          transfer_manager);
typedef PJRT_Error* PJRT_Client_CreateBuffersForAsyncHostToDevice(
    PJRT_Client_CreateBuffersForAsyncHostToDevice_Args* args);

// -------------------------- Device Descriptions ------------------------------

// Device descriptions may be associated with an actual device
// (via PJRT_Device_GetDescription), but they can also be used to describe a
// device that isn't currently available to the plugin. This is useful for
// compiling executables without hardware available, which can then be
// serialized and written somewhere durable, and then loaded and run on actual
// hardware later.

struct PJRT_DeviceDescription_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Id_Args, id);

// The ID of this device. IDs are unique among devices of this type
// (e.g. CPUs, GPUs). On multi-host platforms, this will be unique across all
// hosts' devices.
typedef PJRT_Error* PJRT_DeviceDescription_Id(
    PJRT_DeviceDescription_Id_Args* args);

struct PJRT_DeviceDescription_ProcessIndex_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  int process_index;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_ProcessIndex_Args,
                          process_index);

// The index of the process that this device belongs to, i.e. is addressable
// from. This is not always identical to PJRT_Client_ProcessIndex in a
// multi-process setting, where each client can see devices from all
// processes, but only a subset of them are addressable and have the same
// process_index as the client.
typedef PJRT_Error* PJRT_DeviceDescription_ProcessIndex(
    PJRT_DeviceDescription_ProcessIndex_Args* args);

struct PJRT_DeviceDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  size_t num_attributes;              // out
  const PJRT_NamedValue* attributes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Attributes_Args, attributes);

// Returns an array of device specific attributes with attribute name, value
// and value type.
typedef PJRT_Error* PJRT_DeviceDescription_Attributes(
    PJRT_DeviceDescription_Attributes_Args* args);

struct PJRT_DeviceDescription_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  // `device_kind` string is owned by `device` and has same lifetime as
  // `device`.
  const char* device_kind;  // out
  size_t device_kind_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_Kind_Args, device_kind_size);

// A vendor-dependent string that uniquely identifies the kind of device,
// e.g., "Tesla V100-SXM2-16GB".
typedef PJRT_Error* PJRT_DeviceDescription_Kind(
    PJRT_DeviceDescription_Kind_Args* args);

struct PJRT_DeviceDescription_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* debug_string;  // out
  size_t debug_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_DebugString_Args,
                          debug_string_size);

// Debug string suitable for logging when errors occur. Should be verbose
// enough to describe the current device unambiguously.
typedef PJRT_Error* PJRT_DeviceDescription_DebugString(
    PJRT_DeviceDescription_DebugString_Args* args);

struct PJRT_DeviceDescription_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_DeviceDescription* device_description;
  const char* to_string;  // out
  size_t to_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_DeviceDescription_ToString_Args, to_string_size);

// Debug string suitable for reading by end users, should be reasonably terse,
// for example: "CpuDevice(id=0)".
typedef PJRT_Error* PJRT_DeviceDescription_ToString(
    PJRT_DeviceDescription_ToString_Args* args);

// --------------------------------- Devices -----------------------------------

struct PJRT_Device_GetDescription_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  PJRT_DeviceDescription* device_description;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_GetDescription_Args, device_description);

// Fetch the DeviceDescription associated with this device.
typedef PJRT_Error* PJRT_Device_GetDescription(
    PJRT_Device_GetDescription_Args* args);

struct PJRT_Device_IsAddressable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  bool is_addressable;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_IsAddressable_Args, is_addressable);

// Whether client can issue command to this device.
typedef PJRT_Error* PJRT_Device_IsAddressable(
    PJRT_Device_IsAddressable_Args* args);

struct PJRT_Device_LocalHardwareId_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  int local_hardware_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_LocalHardwareId_Args, local_hardware_id);

// Opaque hardware ID, e.g., the CUDA device number. In general, not guaranteed
// to be dense, and -1 if undefined.
typedef PJRT_Error* PJRT_Device_LocalHardwareId(
    PJRT_Device_LocalHardwareId_Args* args);

struct PJRT_Device_AddressableMemories_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  // Has the lifetime of `device`.
  PJRT_Memory* const* memories;  // out
  size_t num_memories;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_AddressableMemories_Args, num_memories);

// Returns the memories that a device can address.
typedef PJRT_Error* PJRT_Device_AddressableMemories(
    PJRT_Device_AddressableMemories_Args* args);

struct PJRT_Device_DefaultMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;
  // `memory` has the same lifetime as `device`.
  PJRT_Memory* memory;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_DefaultMemory_Args, memory);

// Returns the default memory of a device, i.e. which memory data processed by
// this device should be stored in by default.
typedef PJRT_Error* PJRT_Device_DefaultMemory(
    PJRT_Device_DefaultMemory_Args* args);

struct PJRT_Device_MemoryStats_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Device* device;

  // Number of bytes in use.
  int64_t bytes_in_use;  // out

  // The peak bytes in use.
  int64_t peak_bytes_in_use;      // out
  bool peak_bytes_in_use_is_set;  // out
  // Number of allocations.
  int64_t num_allocs;      // out
  bool num_allocs_is_set;  // out
  // The largest single allocation seen.
  int64_t largest_alloc_size;      // out
  bool largest_alloc_size_is_set;  // out
  // The upper limit of user-allocatable device memory in bytes.
  int64_t bytes_limit;      // out
  bool bytes_limit_is_set;  // out

  // Number of bytes reserved.
  int64_t bytes_reserved;      // out
  bool bytes_reserved_is_set;  // out
  // The peak number of bytes reserved.
  int64_t peak_bytes_reserved;      // out
  bool peak_bytes_reserved_is_set;  // out
  // The upper limit on the number bytes of reservable memory.
  int64_t bytes_reservable_limit;      // out
  bool bytes_reservable_limit_is_set;  // out

  // Largest free block size in bytes.
  int64_t largest_free_block_bytes;      // out
  bool largest_free_block_bytes_is_set;  // out

  // Number of bytes of memory held by the allocator.  This may be higher than
  // bytes_in_use if the allocator holds a pool of memory (e.g. BFCAllocator).
  int64_t pool_bytes;           // out
  bool pool_bytes_is_set;       // out
  int64_t peak_pool_bytes;      // out
  bool peak_pool_bytes_is_set;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Device_MemoryStats_Args, peak_pool_bytes_is_set);

// Device memory/allocator statistics. All returned stats except `bytes_in_use`
// are optional and may not be returned by all platforms. Implementations may
// also return PJRT_Error_Code_UNIMPLEMENTED. Intended for diagnostic purposes.
typedef PJRT_Error* PJRT_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args);

//-------------------------------- Memory --------------------------------------

struct PJRT_Memory_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Id_Args, id);

// The ID of this memory. IDs are unique among memories of this type.
typedef PJRT_Error* PJRT_Memory_Id(PJRT_Memory_Id_Args* args);

struct PJRT_Memory_Kind_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  // `memory_kind` has same lifetime as `memory`.
  const char* kind;  // out
  size_t kind_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Kind_Args, kind_size);

// A platform-dependent string that uniquely identifies the kind of the memory.
typedef PJRT_Error* PJRT_Memory_Kind(PJRT_Memory_Kind_Args* args);

struct PJRT_Memory_Kind_Id_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  int kind_id;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_Kind_Id_Args, kind_id);

// A platform-dependent ID that uniquely identifies the kind of the memory.
typedef PJRT_Error* PJRT_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args);

struct PJRT_Memory_DebugString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* debug_string;  // out
  size_t debug_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_DebugString_Args, debug_string_size);

// Debug string suitable for logging when errors occur. Should be verbose
// enough to describe the current memory unambiguously.
typedef PJRT_Error* PJRT_Memory_DebugString(PJRT_Memory_DebugString_Args* args);

struct PJRT_Memory_ToString_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  const char* to_string;  // out
  size_t to_string_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_ToString_Args, to_string_size);

// Debug string suitable for reading by end users, should be reasonably terse.
typedef PJRT_Error* PJRT_Memory_ToString(PJRT_Memory_ToString_Args* args);

struct PJRT_Memory_AddressableByDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Memory* memory;
  PJRT_Device* const* devices;  // out
  size_t num_devices;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Memory_AddressableByDevices_Args, num_devices);

// Returns the devices that can address this memory.
typedef PJRT_Error* PJRT_Memory_AddressableByDevices(
    PJRT_Memory_AddressableByDevices_Args* args);

// ------------------------------- Execute Context -----------------------------

// An opaque context passed to an execution that may be used to supply
// additional arguments to a derived class of PJRT_Executable. It is a caller
// responsibility to ensure that the context is valid for the duration of the
// execution.
typedef struct PJRT_ExecuteContext PJRT_ExecuteContext;

struct PJRT_ExecuteContext_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteContext* context;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteContext_Create_Args, context);

// Creates an execute context.
typedef PJRT_Error* PJRT_ExecuteContext_Create(
    PJRT_ExecuteContext_Create_Args* args);

struct PJRT_ExecuteContext_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteContext* context;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteContext_Destroy_Args, context);

// Frees an execute context. `context` can be nullptr.
typedef PJRT_Error* PJRT_ExecuteContext_Destroy(
    PJRT_ExecuteContext_Destroy_Args* args);

// ------------------------------- Executables ---------------------------------

struct PJRT_Executable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Destroy_Args, executable);

// Frees `executable`. `executable` can be nullptr.
typedef PJRT_Error* PJRT_Executable_Destroy(PJRT_Executable_Destroy_Args* args);

struct PJRT_LoadedExecutable_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Destroy_Args, executable);

// Frees `executable` and deletes the underlying runtime object as if
// `PJRT_LoadedExecutable_Delete` were called. `executable` can be nullptr.
typedef PJRT_Error* PJRT_LoadedExecutable_Destroy(
    PJRT_LoadedExecutable_Destroy_Args* args);

struct PJRT_LoadedExecutable_GetExecutable_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_replicas;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumReplicas_Args, num_replicas);

// Returns the number of replicas of the executable.
typedef PJRT_Error* PJRT_Executable_NumReplicas(
    PJRT_Executable_NumReplicas_Args* args);

struct PJRT_Executable_NumPartitions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_partitions;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumPartitions_Args, num_partitions);

// Returns the number of partitions of the executable.
typedef PJRT_Error* PJRT_Executable_NumPartitions(
    PJRT_Executable_NumPartitions_Args* args);

struct PJRT_LoadedExecutable_AddressableDevices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  PJRT_Device* const* addressable_devices;  // out
  size_t num_addressable_devices;           // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_AddressableDevices_Args,
                          num_addressable_devices);

// Returns a list of devices this executable will run on.
typedef PJRT_Error* PJRT_LoadedExecutable_AddressableDevices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args);

struct PJRT_Executable_OptimizedProgram_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  bool is_deleted;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_IsDeleted_Args, is_deleted);

// True if and only if PJRT_LoadedExecutable_Delete has previously been called.
typedef PJRT_Error* PJRT_LoadedExecutable_IsDeleted(
    PJRT_LoadedExecutable_IsDeleted_Args* args);

typedef struct PJRT_Chunk {
  void* data;
  size_t size;
  void (*deleter)(void* data, void* deleter_arg);
  // `deleter_arg` will be passed to `deleter` as `deleter_arg` argument.
  void* deleter_arg;
} PJRT_Chunk;

// TODO(b/263390934) implement C API that calls `AddChunk` and other
// `xla::CopyToDeviceStream`.
typedef struct PJRT_CopyToDeviceStream PJRT_CopyToDeviceStream;

struct PJRT_TransferMetadata;

// Returns PJRT_Error* created by PJRT_CallbackError in case of error.
// Otherwise, returns nullptr. The callback must call
// `chunk->deleter(chunk->data, chunk->deleter_arg)` when it's finished with
// `chunk`.
typedef PJRT_Error* (*PJRT_SendCallback)(PJRT_Chunk* chunk,
                                         PJRT_CallbackError* callback_error,
                                         size_t total_size_in_bytes, bool done,
                                         void* user_arg);
// The callback takes the ownership of the stream object. The callback must call
// `PJRT_CopyToDeviceStream_Destroy` when it is done with the stream.
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
  PJRT_Extension_Base* extension_start;
  // Callbacks for when send/recv ops are executed. The outer lists correspond
  // to each device returned by `PJRT_Executable_AddressableDevices` for
  // `executable` (i.e. they will have length `num_devices`). Each inner list
  // contains callback info for each send/recv op in `executable`; the order
  // doesn't matter as the channel IDs are used instead. The callbacks can be
  // stateful and the user code is responsible for managing state. The callback
  // functions must outlive the execution (but not the info structs or lists).
  PJRT_SendCallbackInfo** send_callbacks;
  PJRT_RecvCallbackInfo** recv_callbacks;
  size_t num_send_ops;
  size_t num_recv_ops;
  // If non-zero, identifies this execution as part of a potentially
  // multi-device launch. This can be used to detect scheduling errors, e.g. if
  // multi-host programs are launched in different orders on different hosts,
  // the launch IDs may be used by the runtime to detect the mismatch.
  int launch_id;
  // A list of indices denoting the input buffers that should not be donated.
  // An input buffer may be non-donable, for example, if it is referenced more
  // than once. Since such runtime information is not available at compile time,
  // the compiler might mark the input as `may-alias`, which could lead PjRt to
  // donate the input buffer when it should not. By defining this list of
  // indices, a higher-level PJRT caller can instruct PJRT client not to donate
  // specific input buffers. The caller needs to make sure to keep it alive
  // during the call.
  const int64_t* non_donatable_input_indices;
  size_t num_non_donatable_input_indices;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteOptions, num_non_donatable_input_indices);

struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  // Only needs to stay alive for the duration of the Execute call.
  PJRT_ExecuteOptions* options;
  // Execution input of size [`num_devices`, `num_args`].
  PJRT_Buffer* const* const* argument_lists;
  size_t num_devices;
  size_t num_args;
  // Execution output of size [`num_devices`, num_outputs`], where `num_outputs`
  // is the number of outputs returned by this executable per device. Both the
  // outer (`PJRT_Buffer***`) and inner lists (`PJRT_Buffer**`) must be
  // allocated and deallocated by the caller. PJRT_Buffer_Destroy must be called
  // on the output PJRT_Buffer*.
  PJRT_Buffer** const* output_lists;  // in/out
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
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_NumOutputs_Args, num_outputs);

// Gets the number of outputs per device produced by `executable`.
typedef PJRT_Error* PJRT_Executable_NumOutputs(
    PJRT_Executable_NumOutputs_Args* args);

struct PJRT_Executable_SizeOfGeneratedCodeInBytes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  int64_t size_in_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_SizeOfGeneratedCodeInBytes_Args,
                          size_in_bytes);  // last field in the struct

typedef PJRT_Error* PJRT_Executable_SizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args);

struct PJRT_Executable_Fingerprint_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  // Has the lifetime of `executable`
  const char* executable_fingerprint;  // out
  size_t executable_fingerprint_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Fingerprint_Args,
                          executable_fingerprint_size);

// A unique fingerprint for `executable`. Two executables that were produced by
// compiling with identical inputs (same program, compile options, compiler
// version, etc.) should have the same fingerprint. May not be implemented by
// all platforms.
typedef PJRT_Error* PJRT_Executable_Fingerprint(
    PJRT_Executable_Fingerprint_Args* args);

struct PJRT_Executable_GetCostAnalysis_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_properties;  // out
  // `properties` and any embedded data are owned by and have the same lifetime
  // as `executable`.
  const PJRT_NamedValue* properties;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_GetCostAnalysis_Args, properties);

// Get the cost properties for the executable. Different platforms may return
// different properties; for example, some platforms may return the number of
// operations, or memory size of the input/output of the executable, based on
// program analysis.
typedef PJRT_Error* PJRT_Executable_GetCostAnalysis(
    PJRT_Executable_GetCostAnalysis_Args* args);

struct PJRT_Executable_GetCompiledMemoryStats_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;

  // Mirrors xla::CompiledMemoryStats.
  // Device default memory (e.g., HBM for GPU/TPU) usage stats.
  int64_t generated_code_size_in_bytes;  // out
  int64_t argument_size_in_bytes;        // out
  int64_t output_size_in_bytes;          // out
  // How much argument is reused for output.
  int64_t alias_size_in_bytes;  // out
  int64_t temp_size_in_bytes;   // out

  // Host memory usage stats.
  int64_t host_generated_code_size_in_bytes;  // out
  int64_t host_argument_size_in_bytes;        // out
  int64_t host_output_size_in_bytes;          // out
  int64_t host_alias_size_in_bytes;           // out
  int64_t host_temp_size_in_bytes;            // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_GetCompiledMemoryStats_Args,
                          host_temp_size_in_bytes);

// Return memory stats that allow callers to estimate memory usage when running
// this executable. The memory stats could contain usage info from different
// memory spaces, like default memory (e.g., HBM for GPU/TPU) and host memory.
typedef PJRT_Error* PJRT_Executable_GetCompiledMemoryStats(
    PJRT_Executable_GetCompiledMemoryStats_Args* args);

struct PJRT_Executable_OutputElementTypes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  PJRT_Buffer_Type* output_types;  // out
  size_t num_output_types;         // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputElementTypes_Args,
                          num_output_types);

// Returns a list of element types for outputs.
typedef PJRT_Error* PJRT_Executable_OutputElementTypes(
    PJRT_Executable_OutputElementTypes_Args* args);

struct PJRT_Executable_OutputDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;
  // Has length: sum of all elements in the list `dim_sizes`.
  const int64_t* dims;  // out
  // Has length `num_outputs`.
  const size_t* dim_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputDimensions_Args, dim_sizes);

// Returns a list of dimensions for outputs. Each output has an array shape,
// which is represented by a list of dimensions. The array shapes of all outputs
// are concatenated into a single list of dimensions.
typedef PJRT_Error* PJRT_Executable_OutputDimensions(
    PJRT_Executable_OutputDimensions_Args* args);

struct PJRT_Executable_OutputMemoryKinds_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Executable* executable;
  size_t num_outputs;
  // Has length `num_outputs`.
  const char* const* memory_kinds;  // out
  // Has length `num_outputs`.
  const size_t* memory_kind_sizes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_OutputMemoryKinds_Args,
                          memory_kind_sizes);

// Returns a list of memory kind strings for outputs.
typedef PJRT_Error* PJRT_Executable_OutputMemoryKinds(
    PJRT_Executable_OutputMemoryKinds_Args* args);

typedef struct PJRT_SerializedExecutable PJRT_SerializedExecutable;

struct PJRT_Executable_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_Executable* executable;

  // Lives only as long as serialized_executable
  const char* serialized_bytes;  // out
  size_t serialized_bytes_size;  // out

  PJRT_SerializedExecutable* serialized_executable;  // backs serialized_bytes.
  // cleanup fn must be called to free the backing memory for serialized_bytes.
  // Should only be called once on serialized_executable.
  void (*serialized_executable_deleter)(
      PJRT_SerializedExecutable* exec);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Executable_Serialize_Args,
                          serialized_executable_deleter);

// Returns a platform-specific serialization of `executable`. The serialization
// is not guaranteed to be stable over time.
typedef PJRT_Error* PJRT_Executable_Serialize(
    PJRT_Executable_Serialize_Args* args);

struct PJRT_Executable_DeserializeAndLoad_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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

struct PJRT_LoadedExecutable_Fingerprint_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  // Has the lifetime of `executable`
  const char* executable_fingerprint;  // out
  size_t executable_fingerprint_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_Fingerprint_Args,
                          executable_fingerprint_size);
// DEPRECATED. Will be removed in PJRT version 2.0. Please use
// PJRT_Executable_Fingerprint instead. A unique fingerprint for `executable`.
// Two executables that were produced by compiling with identical inputs (same
// program, compile options, compiler version, etc.) should have the same
// fingerprint. May not be implemented by all platforms.
typedef PJRT_Error* PJRT_LoadedExecutable_Fingerprint(
    PJRT_LoadedExecutable_Fingerprint_Args* args);

// ---------------------------------- Buffers ----------------------------------

struct PJRT_Buffer_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Destroy_Args, buffer);

// Deletes the underlying runtime objects as if 'PJRT_Buffer_Delete' were
// called and frees `buffer`. `buffer` can be nullptr.
typedef PJRT_Error* PJRT_Buffer_Destroy(PJRT_Buffer_Destroy_Args* args);

struct PJRT_Buffer_ElementType_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Buffer_Type type;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_ElementType_Args, type);

// Returns the type of the array elements of a buffer.
typedef PJRT_Error* PJRT_Buffer_ElementType(PJRT_Buffer_ElementType_Args* args);

struct PJRT_Buffer_Dimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  // Has the lifetime of `buffer` and length `num_dims`.
  const int64_t* dims;  // out
  size_t num_dims;      // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Dimensions_Args, num_dims);

// Returns the array shape of `buffer`, i.e. the size of each dimension.
typedef PJRT_Error* PJRT_Buffer_Dimensions(PJRT_Buffer_Dimensions_Args* args);

struct PJRT_Buffer_UnpaddedDimensions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  // Has the lifetime of `buffer` and length `num_dims`.
  const int64_t* unpadded_dims;  // out
  size_t num_dims;               // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_UnpaddedDimensions_Args, num_dims);

// Returns the unpadded array shape of `buffer`. This usually is equivalent to
// PJRT_Buffer_Dimensions, but for implementations that support
// dynamically-sized dimensions via padding to a fixed size, any dynamic
// dimensions may have a smaller unpadded size than the padded size reported by
// PJRT_Buffer_Dimensions. ("Dynamic" dimensions are those whose length is
// only known at runtime, vs. "static" dimensions whose size is fixed at compile
// time.)
typedef PJRT_Error* PJRT_Buffer_UnpaddedDimensions(
    PJRT_Buffer_UnpaddedDimensions_Args* args);

struct PJRT_Buffer_DynamicDimensionIndices_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  // Has the lifetime of `buffer` and length `num_dynamic_dims`.
  const size_t* dynamic_dim_indices;  // out
  size_t num_dynamic_dims;            // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_DynamicDimensionIndices_Args,
                          num_dynamic_dims);

// Returns the indices of dynamically-sized dimensions, or an empty list if all
// dimensions are static. ("Dynamic" dimensions are those whose length is
// only known at runtime, vs. "static" dimensions whose size is fixed at compile
// time.)
typedef PJRT_Error* PJRT_Buffer_DynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args* args);

struct PJRT_Buffer_GetMemoryLayout_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  // Layout data is owned by and has the lifetime of `buffer`.
  PJRT_Buffer_MemoryLayout layout;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_GetMemoryLayout_Args, layout);

// DEPRECATED. Please use layout extension instead.
// https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api_layouts_extension.h
// Returns the memory layout of the data in this buffer.
typedef PJRT_Error* PJRT_Buffer_GetMemoryLayout(
    PJRT_Buffer_GetMemoryLayout_Args* args);

struct PJRT_Buffer_ToHostBuffer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* src;

  // The caller can specify an optional host layout. If nullptr, the layout of
  // the src buffer will be used. The caller is responsible to keep the data
  // (tiled or strides) in the host_layout alive during the call.
  PJRT_Buffer_MemoryLayout* host_layout;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_deleted;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsDeleted_Args, is_deleted);

// True if and only if PJRT_Buffer_Delete has previously been called.
typedef PJRT_Error* PJRT_Buffer_IsDeleted(PJRT_Buffer_IsDeleted_Args* args);

struct PJRT_Buffer_CopyRawToHost_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  void* dst;
  int64_t offset;
  int64_t transfer_size;
  PJRT_Event* event;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyRawToHost_Args, event);

typedef PJRT_Error* PJRT_Buffer_CopyRawToHost(
    PJRT_Buffer_CopyRawToHost_Args* args);

struct PJRT_Buffer_CopyToDevice_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* dst_device;
  PJRT_Buffer* dst_buffer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyToDevice_Args, dst_buffer);

// Copies the buffer to device `dst_device` within the same client. Caller is
// responsible for freeing returned `dst_buffer` with PJRT_Buffer_Destroy.
// Returns an error if the buffer is already on `dst_device`.
typedef PJRT_Error* PJRT_Buffer_CopyToDevice(
    PJRT_Buffer_CopyToDevice_Args* args);

struct PJRT_Buffer_CopyToMemory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Memory* dst_memory;
  PJRT_Buffer* dst_buffer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_CopyToMemory_Args, dst_buffer);

// Copies the buffer to memory `dst_memory` within the same client. Caller is
// responsible for freeing returned `dst_buffer` with PJRT_Buffer_Destroy.
// Returns an error if the buffer is already on `dst_memory`.
typedef PJRT_Error* PJRT_Buffer_CopyToMemory(
    PJRT_Buffer_CopyToMemory_Args* args);

struct PJRT_Buffer_IsOnCpu_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  bool is_on_cpu;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IsOnCpu_Args, is_on_cpu);

// Whether this buffer is on CPU and thus allows for certain optimizations.
typedef PJRT_Error* PJRT_Buffer_IsOnCpu(PJRT_Buffer_IsOnCpu_Args* args);

struct PJRT_Buffer_Device_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Device* device;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Device_Args, device);

// Returns this buffer's storage device.
typedef PJRT_Error* PJRT_Buffer_Device(PJRT_Buffer_Device_Args* args);

struct PJRT_Buffer_Memory_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  PJRT_Memory* memory;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_Memory_Args, memory);

// Returns this buffer's storage memory.
typedef PJRT_Error* PJRT_Buffer_Memory(PJRT_Buffer_Memory_Args* args);

struct PJRT_Buffer_ReadyEvent_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  uintptr_t buffer_pointer;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_UnsafePointer_Args, buffer_pointer);

// Returns platform-dependent address for the given buffer that is often but
// not guaranteed to be the physical/device address.
typedef PJRT_Error* PJRT_Buffer_UnsafePointer(
    PJRT_Buffer_UnsafePointer_Args* args);

struct PJRT_Buffer_IncreaseExternalReferenceCount_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_IncreaseExternalReferenceCount_Args,
                          buffer);

// Increments the reference count for the buffer. The reference count indicates
// the raw buffer data is being shared with another framework (e.g. NumPy,
// dlpack) and should not be deleted or moved by the PJRT implementation (e.g.
// for memory compaction). TODO(b/295230663): document more API contract
// details, e.g. does this block, can the buffer be modified in-place.
typedef PJRT_Error* PJRT_Buffer_IncreaseExternalReferenceCount(
    PJRT_Buffer_IncreaseExternalReferenceCount_Args* args);

struct PJRT_Buffer_DecreaseExternalReferenceCount_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_DecreaseExternalReferenceCount_Args,
                          buffer);

// Decrements the reference count for the buffer. Returns an error if the
// reference count is zero (i.e. PJRT_Buffer_IncreaseExternalReferenceCount is
// not called beforehand).
typedef PJRT_Error* PJRT_Buffer_DecreaseExternalReferenceCount(
    PJRT_Buffer_DecreaseExternalReferenceCount_Args* args);

struct PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_Buffer* buffer;
  void* device_memory_ptr;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args,
                          device_memory_ptr);

// Returns the opaque device memory data pointer of the buffer. The returned
// data pointer may become invalid at any point unless the external reference
// count is greater than 0 via PJRT_Buffer_IncreaseExternalReferenceCount.
typedef PJRT_Error* PJRT_Buffer_OpaqueDeviceMemoryDataPointer(
    PJRT_Buffer_OpaqueDeviceMemoryDataPointer_Args* args);

// ---------------------------- CopyToDeviceStream -----------------------------

struct PJRT_CopyToDeviceStream_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_Destroy_Args, stream);

// Frees `stream`. `stream` can be nullptr.
typedef PJRT_Error* PJRT_CopyToDeviceStream_Destroy(
    PJRT_CopyToDeviceStream_Destroy_Args* args);

struct PJRT_CopyToDeviceStream_AddChunk_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  // Takes ownership of `chunk` (i.e. implementation will call chunk.deleter).
  PJRT_Chunk* chunk;
  PJRT_Event* transfer_complete;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_AddChunk_Args,
                          transfer_complete);

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
  PJRT_Extension_Base* extension_start;
  PJRT_CopyToDeviceStream* stream;
  int64_t total_bytes;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_CopyToDeviceStream_TotalBytes_Args, total_bytes);

// Returns the total amount of data the stream expects to be transferred.
typedef PJRT_Error* PJRT_CopyToDeviceStream_TotalBytes(
    PJRT_CopyToDeviceStream_TotalBytes_Args* args);

struct PJRT_CopyToDeviceStream_GranuleSize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
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
  PJRT_Extension_Base* extension_start;
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

struct PJRT_TopologyDescription_Create_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const char* topology_name;
  size_t topology_name_size;
  // Extra platform-specific options to create a client.
  const PJRT_NamedValue* create_options;
  size_t num_options;
  PJRT_TopologyDescription* topology;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Create_Args, topology);

// Creates and initializes a new PJRT_TopologyDescription and returns in
// `topology`.
typedef PJRT_Error* PJRT_TopologyDescription_Create(
    PJRT_TopologyDescription_Create_Args* args);

struct PJRT_TopologyDescription_Destroy_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Destroy_Args, topology);

// Frees `topology`. `topology` can be nullptr.
typedef PJRT_Error* PJRT_TopologyDescription_Destroy(
    PJRT_TopologyDescription_Destroy_Args* args);

struct PJRT_TopologyDescription_PlatformVersion_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  // `platform_version` has the same lifetime as `topology`. It's owned by
  // `topology`.
  const char* platform_version;  // out
  size_t platform_version_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_PlatformVersion_Args,
                          platform_version_size);

// Returns a string containing human-readable, platform-specific version info
// (e.g. the CUDA version on GPU or libtpu version on Cloud TPU).
typedef PJRT_Error* PJRT_TopologyDescription_PlatformVersion(
    PJRT_TopologyDescription_PlatformVersion_Args* args);

struct PJRT_TopologyDescription_PlatformName_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  // `platform_name` has the same lifetime as `topology`. It is owned by
  // `topology`.
  const char* platform_name;  // out
  size_t platform_name_size;  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_PlatformName_Args,
                          platform_name_size);

// Returns a string that identifies the platform (e.g. "cpu", "gpu", "tpu").
typedef PJRT_Error* PJRT_TopologyDescription_PlatformName(
    PJRT_TopologyDescription_PlatformName_Args* args);

struct PJRT_TopologyDescription_GetDeviceDescriptions_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;
  // Has the same lifetime as topology.
  PJRT_DeviceDescription* const* descriptions;  // out
  size_t num_descriptions;                      // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_GetDeviceDescriptions_Args,
                          num_descriptions);

// Returns descriptions for all devices in this topology. The device
// descriptions can be returned in any order, but will be in the same order
// across calls within a process.
typedef PJRT_Error* PJRT_TopologyDescription_GetDeviceDescriptions(
    PJRT_TopologyDescription_GetDeviceDescriptions_Args* args);

typedef struct PJRT_SerializedTopology PJRT_SerializedTopology;

struct PJRT_TopologyDescription_Serialize_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;

  // Lives only as long as serialized_topology.
  const char* serialized_bytes;  // out
  size_t serialized_bytes_size;  // out

  PJRT_SerializedTopology* serialized_topology;  // out
  // Must be called exactly once to free the backing memory for
  // serialized_bytes.
  void (*serialized_topology_deleter)(
      PJRT_SerializedTopology* serialized_topology);  // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Serialize_Args,
                          serialized_topology_deleter);

// Serializes the TopologyDescription to a string for use in cache keys.
typedef PJRT_Error* PJRT_TopologyDescription_Serialize(
    PJRT_TopologyDescription_Serialize_Args* args);

struct PJRT_TopologyDescription_Attributes_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_TopologyDescription* topology;

  // Only lives as long as topology.
  const PJRT_NamedValue* attributes;  // out
  size_t num_attributes;              // out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_TopologyDescription_Attributes_Args,
                          num_attributes);

// Returns platform-specific topology attributes.
typedef PJRT_Error* PJRT_TopologyDescription_Attributes(
    PJRT_TopologyDescription_Attributes_Args* args);

struct PJRT_Compile_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  const PJRT_TopologyDescription* topology;
  // Only needs to stay alive for the duration of the Compile call.
  // `program->format` and `program->format_size` are owned by the caller.
  const PJRT_Program* program;
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
typedef struct PJRT_Api {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_Api_Version pjrt_api_version;

  _PJRT_API_STRUCT_FIELD(PJRT_Error_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_Message);
  _PJRT_API_STRUCT_FIELD(PJRT_Error_GetCode);

  _PJRT_API_STRUCT_FIELD(PJRT_Plugin_Initialize);
  _PJRT_API_STRUCT_FIELD(PJRT_Plugin_Attributes);

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
  _PJRT_API_STRUCT_FIELD(PJRT_Client_AddressableMemories);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_Compile);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_DefaultDeviceAssignment);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_BufferFromHostBuffer);

  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_ProcessIndex);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Attributes);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_Kind);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_DebugString);
  _PJRT_API_STRUCT_FIELD(PJRT_DeviceDescription_ToString);

  _PJRT_API_STRUCT_FIELD(PJRT_Device_GetDescription);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_IsAddressable);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_LocalHardwareId);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_AddressableMemories);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_DefaultMemory);
  _PJRT_API_STRUCT_FIELD(PJRT_Device_MemoryStats);

  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Id);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Kind);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_DebugString);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_ToString);
  _PJRT_API_STRUCT_FIELD(PJRT_Memory_AddressableByDevices);

  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Name);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumReplicas);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumPartitions);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_NumOutputs);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_SizeOfGeneratedCodeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_GetCostAnalysis);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputMemoryKinds);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OptimizedProgram);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Serialize);

  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_GetExecutable);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_AddressableDevices);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Execute);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_DeserializeAndLoad);
  _PJRT_API_STRUCT_FIELD(PJRT_LoadedExecutable_Fingerprint);

  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ElementType);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Dimensions);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_UnpaddedDimensions);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_DynamicDimensionIndices);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_GetMemoryLayout);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OnDeviceSizeInBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Device);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Memory);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_Delete);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsDeleted);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToDevice);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ToHostBuffer);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IsOnCpu);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_ReadyEvent);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_UnsafePointer);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_IncreaseExternalReferenceCount);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_DecreaseExternalReferenceCount);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_OpaqueDeviceMemoryDataPointer);

  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_AddChunk);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_TotalBytes);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_GranuleSize);
  _PJRT_API_STRUCT_FIELD(PJRT_CopyToDeviceStream_CurrentBytes);

  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_PlatformName);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_PlatformVersion);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_GetDeviceDescriptions);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Serialize);
  _PJRT_API_STRUCT_FIELD(PJRT_TopologyDescription_Attributes);

  _PJRT_API_STRUCT_FIELD(PJRT_Compile);

  // Always add new fields to the end of the struct. Move fields below to their
  // corresponding places after each major version bump.
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputElementTypes);
  _PJRT_API_STRUCT_FIELD(PJRT_Executable_OutputDimensions);

  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyToMemory);

  _PJRT_API_STRUCT_FIELD(PJRT_Client_CreateViewOfDeviceBuffer);

  _PJRT_API_STRUCT_FIELD(PJRT_Executable_Fingerprint);

  _PJRT_API_STRUCT_FIELD(PJRT_Client_TopologyDescription);

  _PJRT_API_STRUCT_FIELD(PJRT_Executable_GetCompiledMemoryStats);

  _PJRT_API_STRUCT_FIELD(PJRT_Memory_Kind_Id);

  _PJRT_API_STRUCT_FIELD(PJRT_ExecuteContext_Create);
  _PJRT_API_STRUCT_FIELD(PJRT_ExecuteContext_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_Buffer_CopyRawToHost);
  _PJRT_API_STRUCT_FIELD(PJRT_AsyncHostToDeviceTransferManager_Destroy);
  _PJRT_API_STRUCT_FIELD(PJRT_AsyncHostToDeviceTransferManager_TransferData);
  _PJRT_API_STRUCT_FIELD(PJRT_Client_CreateBuffersForAsyncHostToDevice);
} PJRT_Api;

enum {
  PJRT_Api_STRUCT_SIZE =
      PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_CreateBuffersForAsyncHostToDevice)
};

#undef _PJRT_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_H_
