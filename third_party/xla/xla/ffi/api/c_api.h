/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_FFI_API_C_API_H_
#define XLA_FFI_API_C_API_H_

#include <stddef.h>
#include <stdint.h>

// XLA FFI C API follows PJRT API style for consistency. See `pjrt_c_api.h`.
// More details on versioning strategy and example version checks:
// https://github.com/tensorflow/community/blob/master/rfcs/20200612-stream-executor-c-api/C_API_versioning_strategy.md

// Every struct passed across the C API boundary has its size as a member, and
// we use it as a sanity check for API compatibility.
#define XLA_FFI_STRUCT_SIZE(struct_type, last_field) \
  (offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field))

// Must update XLA_FFI_DEFINE_STRUCT_TRAITS with the new `last_field` after
// adding a new member to a struct.
#define XLA_FFI_DEFINE_STRUCT_TRAITS(sname, last_field) \
  typedef struct sname sname;                           \
  enum { sname##_STRUCT_SIZE = XLA_FFI_STRUCT_SIZE(sname, last_field) }

#ifdef __cplusplus
extern "C" {
#endif

typedef struct XLA_FFI_Api XLA_FFI_Api;                  // Forward declare
typedef struct XLA_FFI_InternalApi XLA_FFI_InternalApi;  // Forward declare

//===----------------------------------------------------------------------===//
// Version
//===----------------------------------------------------------------------===//

// Incremented when an ABI-incompatible change is made to the interface.
//
// Major changes include:
// * Deleting a method or argument
// * Changing the type of an argument
// * Rearranging fields in the XLA_FFI_Api or argument structs
#define XLA_FFI_API_MAJOR 0

// Incremented when the interface is updated in a way that is potentially
// ABI-compatible with older versions, if supported by the caller and/or
// implementation.
//
// Callers can implement forwards compatibility by using XLA_FFI_Api_Version to
// check if the implementation is aware of newer interface additions.
//
// Implementations can implement backwards compatibility by using the
// `struct_size` fields to detect how many struct fields the caller is aware of.
//
// Minor changes include:
// * Adding a new field to the XLA_FFI_Api or argument structs
// * Renaming a method or argument (doesn't affect ABI)
#define XLA_FFI_API_MINOR 0

struct XLA_FFI_Api_Version {
  size_t struct_size;
  void* priv;
  int major_version;  // out
  int minor_version;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Api_Version, minor_version);

//===----------------------------------------------------------------------===//
// Error codes
//===----------------------------------------------------------------------===//

// XLA FFI error is a mechanism to communicate errors between XLA and XLA FFI
// via a set of C APIs. This is somewhat similar to type-erased version of
// absl::Status exposed via API with opaque pointers.
//
// Returning NULL error is equivalent to returning absl::OkStatus().
//
// Ownership of an XLA_FFI_Error is always transferred to the caller, and the
// caller is responsible for destroying it:
//
// (1) If the error is returned from an XLA FFI handler, the XLA runtime will
//     destroy it (XLA is the caller who calls into the handler implementation).
//
// (2) If the error is returned from an XLA FFI API call, the caller is
//     responsible for destroying it.
typedef struct XLA_FFI_Error XLA_FFI_Error;

// Codes are based on https://abseil.io/docs/cpp/guides/status-codes
typedef enum {
  XLA_FFI_Error_Code_OK = 0,
  XLA_FFI_Error_Code_CANCELLED = 1,
  XLA_FFI_Error_Code_UNKNOWN = 2,
  XLA_FFI_Error_Code_INVALID_ARGUMENT = 3,
  XLA_FFI_Error_Code_DEADLINE_EXCEEDED = 4,
  XLA_FFI_Error_Code_NOT_FOUND = 5,
  XLA_FFI_Error_Code_ALREADY_EXISTS = 6,
  XLA_FFI_Error_Code_PERMISSION_DENIED = 7,
  XLA_FFI_Error_Code_RESOURCE_EXHAUSTED = 8,
  XLA_FFI_Error_Code_FAILED_PRECONDITION = 9,
  XLA_FFI_Error_Code_ABORTED = 10,
  XLA_FFI_Error_Code_OUT_OF_RANGE = 11,
  XLA_FFI_Error_Code_UNIMPLEMENTED = 12,
  XLA_FFI_Error_Code_INTERNAL = 13,
  XLA_FFI_Error_Code_UNAVAILABLE = 14,
  XLA_FFI_Error_Code_DATA_LOSS = 15,
  XLA_FFI_Error_Code_UNAUTHENTICATED = 16
} XLA_FFI_Error_Code;

//===----------------------------------------------------------------------===//
// Error reporting APIs
//===----------------------------------------------------------------------===//

struct XLA_FFI_Error_Create_Args {
  size_t struct_size;
  void* priv;
  const char* message;
  XLA_FFI_Error_Code errc;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Error_Create_Args, errc);

typedef XLA_FFI_Error* XLA_FFI_Error_Create(XLA_FFI_Error_Create_Args* args);

struct XLA_FFI_Error_GetMessage_Args {
  size_t struct_size;
  void* priv;
  XLA_FFI_Error* error;
  const char* message;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Error_GetMessage_Args, message);

typedef void XLA_FFI_Error_GetMessage(XLA_FFI_Error_GetMessage_Args* args);

struct XLA_FFI_Error_Destroy_Args {
  size_t struct_size;
  void* priv;
  XLA_FFI_Error* error;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Error_Destroy_Args, error);

typedef void XLA_FFI_Error_Destroy(XLA_FFI_Error_Destroy_Args* args);

//===----------------------------------------------------------------------===//
// DataType
//===----------------------------------------------------------------------===//

// This enum corresponds to xla::PrimitiveType enum defined in `xla_data.proto`.
// LINT.IfChange
typedef enum {
  XLA_FFI_DataType_INVALID = 0,
  XLA_FFI_DataType_PRED = 1,
  XLA_FFI_DataType_S8 = 2,
  XLA_FFI_DataType_S16 = 3,
  XLA_FFI_DataType_S32 = 4,
  XLA_FFI_DataType_S64 = 5,
  XLA_FFI_DataType_U8 = 6,
  XLA_FFI_DataType_U16 = 7,
  XLA_FFI_DataType_U32 = 8,
  XLA_FFI_DataType_U64 = 9,
  XLA_FFI_DataType_F16 = 10,
  XLA_FFI_DataType_F32 = 11,
  XLA_FFI_DataType_F64 = 12,
  XLA_FFI_DataType_BF16 = 16,
  XLA_FFI_DataType_C64 = 15,
  XLA_FFI_DataType_C128 = 18,
  XLA_FFI_DataType_TOKEN = 17,
  XLA_FFI_DataType_F8E5M2 = 19,
  XLA_FFI_DataType_F8E4M3FN = 20,
  XLA_FFI_DataType_F8E4M3B11FNUZ = 23,
  XLA_FFI_DataType_F8E5M2FNUZ = 24,
  XLA_FFI_DataType_F8E4M3FNUZ = 25,
} XLA_FFI_DataType;
// LINT.ThenChange(ffi_test.cc)

//===----------------------------------------------------------------------===//
// Builtin argument types
//===----------------------------------------------------------------------===//

struct XLA_FFI_Buffer {
  size_t struct_size;
  void* priv;

  XLA_FFI_DataType dtype;
  void* data;
  int64_t rank;
  int64_t* dims;  // length == rank
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Buffer, dims);

typedef enum {
  XLA_FFI_ArgType_BUFFER = 1,
} XLA_FFI_ArgType;

//===----------------------------------------------------------------------===//
// Builtin result types
//===----------------------------------------------------------------------===//

typedef enum {
  XLA_FFI_RetType_BUFFER = 1,
} XLA_FFI_RetType;

//===----------------------------------------------------------------------===//
// Builtin attribute types
//===----------------------------------------------------------------------===//

typedef enum {
  XLA_FFI_AttrType_ARRAY = 1,
  XLA_FFI_AttrType_DICTIONARY = 2,
  XLA_FFI_AttrType_SCALAR = 3,
  XLA_FFI_AttrType_STRING = 4,
} XLA_FFI_AttrType;

//===----------------------------------------------------------------------===//
// Execution context
//===----------------------------------------------------------------------===//

// Execution context provides access to per-invocation state.
typedef struct XLA_FFI_ExecutionContext XLA_FFI_ExecutionContext;

//===----------------------------------------------------------------------===//
// Primitives.
//===----------------------------------------------------------------------===//

// TypeId uniquely identifies a user-defined type in a given XLA FFI instance.
struct XLA_FFI_TypeId {
  int64_t type_id;
};

// We use byte spans to pass strings to handlers because strings might not be
// null terminated, and even if they are, looking for a null terminator can
// become very expensive in tight loops.
struct XLA_FFI_ByteSpan {
  const char* ptr;
  size_t len;
};

// A struct to pass a scalar value to FFI handler.
struct XLA_FFI_Scalar {
  XLA_FFI_DataType dtype;
  void* value;
};

// A struct to pass a dense array to FFI handler.
struct XLA_FFI_Array {
  XLA_FFI_DataType dtype;
  size_t size;
  void* data;
};

//===----------------------------------------------------------------------===//
// Call frame
//===----------------------------------------------------------------------===//

// XLA runtime has multiple execution stages and it is possible to run
// different handlers for each stage:
//
// (1) Prepare - called before the execution to let FFI handlers to prepare
//     for the execution and request resources from runtime, i.e. in XLA:GPU
//     we use prepare stage to request collective cliques.
//
// (2) Initialize - called before the execution after acquiring all the
//     resources requested in the prepare stage.
//
// (3) Execute - called when FFI handler is executed. Note that FFI handler
//     can be called as a part of command buffer capture (CUDA graph capture
//     on GPU backend) and argument buffers might contain uninitialized
//     values in this case.
//
// It is undefined behavior to access argument buffers in prepare and
// initialize stages as they might not be initialized yet. However it is safe
// to use memory address as it is assigned ahead of time by buffer assignment.
typedef enum {
  XLA_FFI_ExecutionStage_PREPARE = 0,
  XLA_FFI_ExecutionStage_INITIALIZE = 1,
  XLA_FFI_ExecutionStage_EXECUTE = 2,
} XLA_FFI_ExecutionStage;

struct XLA_FFI_Args {
  size_t struct_size;
  void* priv;

  int64_t size;
  XLA_FFI_ArgType* types;  // length == size
  void** args;             // length == size
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Args, args);

struct XLA_FFI_Rets {
  size_t struct_size;
  void* priv;

  int64_t size;
  XLA_FFI_RetType* types;  // length == size
  void** rets;             // length == size
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Rets, rets);

// FFI handler attributes are always sorted by name, so that the handler can
// rely on binary search to look up attributes by name.
struct XLA_FFI_Attrs {
  size_t struct_size;
  void* priv;

  int64_t size;
  XLA_FFI_AttrType* types;   // length == size
  XLA_FFI_ByteSpan** names;  // length == size
  void** attrs;              // length == size
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Attrs, attrs);

struct XLA_FFI_CallFrame {
  size_t struct_size;
  void* priv;

  const XLA_FFI_Api* api;
  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_ExecutionStage stage;
  XLA_FFI_Args args;
  XLA_FFI_Rets rets;
  XLA_FFI_Attrs attrs;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_CallFrame, attrs);

//===----------------------------------------------------------------------===//
// FFI handler
//===----------------------------------------------------------------------===//

// External functions registered with XLA as FFI handlers.
typedef XLA_FFI_Error* XLA_FFI_Handler(XLA_FFI_CallFrame* call_frame);

// XLA FFI handlers for execution stages (see XLA_FFI_ExecutionStage).
struct XLA_FFI_Handler_Bundle {
  XLA_FFI_Handler* prepare;     // optional
  XLA_FFI_Handler* initialize;  // optional
  XLA_FFI_Handler* execute;     // required
};

enum XLA_FFI_Handler_TraitsBits {
  // Calls to FFI handler are safe to trace into the command buffer. It means
  // that calls to FFI handler always launch exactly the same device operations
  // (can depend on attribute values) that can be captured and then replayed.
  XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1u << 0,
};

typedef uint32_t XLA_FFI_Handler_Traits;

struct XLA_FFI_Handler_Register_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ByteSpan name;
  XLA_FFI_ByteSpan platform;
  XLA_FFI_Handler_Bundle bundle;
  XLA_FFI_Handler_Traits traits;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Handler_Register_Args, traits);

typedef XLA_FFI_Error* XLA_FFI_Handler_Register(
    XLA_FFI_Handler_Register_Args* args);

//===----------------------------------------------------------------------===//
// TypeId
//===----------------------------------------------------------------------===//

struct XLA_FFI_TypeId_Register_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ByteSpan name;
  XLA_FFI_TypeId* type_id;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_TypeId_Register_Args, type_id);

// Registers user type `name` and returns a unique `type_id`.
typedef XLA_FFI_Error* XLA_FFI_TypeId_Register(
    XLA_FFI_TypeId_Register_Args* args);

//===----------------------------------------------------------------------===//
// ExecutionContext
//===----------------------------------------------------------------------===//

struct XLA_FFI_ExecutionContext_Get_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_TypeId* type_id;
  void* data;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_ExecutionContext_Get_Args, data);

// Returns an opaque data from the execution context for a given name.
typedef XLA_FFI_Error* XLA_FFI_ExecutionContext_Get(
    XLA_FFI_ExecutionContext_Get_Args* args);

//===----------------------------------------------------------------------===//
// Stream
//===----------------------------------------------------------------------===//

struct XLA_FFI_Stream_Get_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ExecutionContext* ctx;
  void* stream;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Stream_Get_Args, stream);

// Returns an underling platform-specific stream via out argument, i.e. for CUDA
// platform it returns `CUstream` (same as `cudaStream`).
typedef XLA_FFI_Error* XLA_FFI_Stream_Get(XLA_FFI_Stream_Get_Args* args);

//===----------------------------------------------------------------------===//
// Device memory allocation
//===----------------------------------------------------------------------===//

struct XLA_FFI_DeviceMemory_Allocate_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ExecutionContext* ctx;
  size_t size;
  size_t alignment;
  void* data;  // out
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_DeviceMemory_Allocate_Args, data);

// Allocates a block of memory on the device bound to the execution context.
typedef XLA_FFI_Error* XLA_FFI_DeviceMemory_Allocate(
    XLA_FFI_DeviceMemory_Allocate_Args* args);

struct XLA_FFI_DeviceMemory_Free_Args {
  size_t struct_size;
  void* priv;

  XLA_FFI_ExecutionContext* ctx;
  size_t size;
  void* data;
};

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_DeviceMemory_Free_Args, data);

// Frees previously allocated device memory.
typedef XLA_FFI_Error* XLA_FFI_DeviceMemory_Free(
    XLA_FFI_DeviceMemory_Free_Args* args);

//===----------------------------------------------------------------------===//
// API access
//===----------------------------------------------------------------------===//

#define _XLA_FFI_API_STRUCT_FIELD(fn_type) fn_type* fn_type

struct XLA_FFI_Api {
  size_t struct_size;
  void* priv;

  XLA_FFI_InternalApi* internal_api;

  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_Create);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_GetMessage);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_Destroy);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Handler_Register);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Stream_Get);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_TypeId_Register);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ExecutionContext_Get);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_DeviceMemory_Allocate);
  _XLA_FFI_API_STRUCT_FIELD(XLA_FFI_DeviceMemory_Free);
};

#undef _XLA_FFI_API_STRUCT_FIELD

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_Api, XLA_FFI_Stream_Get);

const XLA_FFI_Api* XLA_FFI_GetApi();

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_H_
