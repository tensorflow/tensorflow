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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_C_API_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_C_API_H_

#include <stddef.h>
#include <stdint.h>

// Every struct passed across the C API boundary has its size as a member, and
// we use it as a sanity check for API compatibility.
#define XLA_FFI_STRUCT_SIZE(struct_type, last_field) \
  offsetof(struct_type, last_field) + sizeof(((struct_type*)0)->last_field)

#ifdef __cplusplus
extern "C" {
#endif

// Forward declare.
typedef struct XLA_FFI_Api XLA_FFI_Api;

//===----------------------------------------------------------------------===//
// XLA FFI Type checking.
//===----------------------------------------------------------------------===//

// XLA FFI passes type ids along with all arguments and attributes so that it
// should be possible to check types at run time inside the FFI handler.
typedef const void* XLA_FFI_TypeId;

typedef XLA_FFI_TypeId XLA_FFI_Get_TypeId();

//===----------------------------------------------------------------------===//
// XLA FFI Error handling.
//===----------------------------------------------------------------------===//

// XLA FFI handler must return a XLA_FFI_Error*, which is NULL if there is no
// error and set if there is. Caller allocates any returned XLA_FFI_Errors, and
// the XLA FFI is responsible for freeing them.
typedef struct XLA_FFI_Error XLA_FFI_Error;

// Codes are based on https://abseil.io/docs/cpp/guides/status-codes
typedef enum {
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
// XLA FFI module defines a set of exported FFI functions and their state.
//===----------------------------------------------------------------------===//

// XLA FFI module is a way to structure FFI functions together with a state
// required for calling them. XLA runtime executable can be linked with multiple
// of such modules at run time.
typedef struct XLA_FFI_Module XLA_FFI_Module;

// XLA FFI module state can be instantiated once for each XLA executable, and
// its life time will be bound to the executable iself, or it can be
// instantiated for each separate execution.
typedef enum {
  XLA_FFI_Module_State_PER_EXECUTABLE = 1,
  XLA_FFI_Module_State_PER_EXECUTION = 2,
} XLA_FFI_Module_StateType;

// Exported FFI functions will have access to the state object that they can
// use to share data between different function invocations. If the state is
// instantiated for each executable, it is the user's responsibility to
// guarantee that it is thread-safe to use from multiple concurrent executions.
typedef struct XLA_FFI_Module_State XLA_FFI_Module_State;

// Creates a new per-executable module state.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_Module* module;
  XLA_FFI_Module_State* state;  // out
} XLA_FFI_Module_CreateState_Args;

const size_t XLA_FFI_Module_CreateState_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Module_CreateState_Args, state);

typedef XLA_FFI_Error* XLA_FFI_Module_CreateState(
    XLA_FFI_Module_CreateState_Args* args);

// Destroys a module state.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_Module* module;
  XLA_FFI_Module_State* state;
} XLA_FFI_Module_DestroyState_Args;

const size_t XLA_FFI_Module_DestroyState_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Module_DestroyState_Args, state);

typedef void XLA_FFI_Module_DestroyState(
    XLA_FFI_Module_DestroyState_Args* args);

//===----------------------------------------------------------------------===//
// XLA FFI Error Reporting APIs.
//===----------------------------------------------------------------------===//

typedef struct {
  size_t struct_size;
  void* priv;
  const char* message;
  XLA_FFI_Error_Code errc;
} XLA_FFI_Error_Create_Args;

const size_t XLA_FFI_Error_Create_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Error_Create_Args, message);

typedef XLA_FFI_Error* XLA_FFI_Error_Create(XLA_FFI_Error_Create_Args* args);

//===----------------------------------------------------------------------===//
// XLA FFI Stream.
//===----------------------------------------------------------------------===//

// XLA FFI stream is an opaque handle to the underlying stream executor `Stream`
// implementation. In XLA:GPU it is `se::gpu::GpuStreamHandle` (when running on
// CUDA platform it is a `CUstream`).
typedef struct XLA_FFI_Stream XLA_FFI_Stream;

//===----------------------------------------------------------------------===//
// XLA FFI Execution Context.
//===----------------------------------------------------------------------===//

typedef struct XLA_FFI_ExecutionContext XLA_FFI_ExecutionContext;

// Get `XLA_FFI_Module_State` from the execution context.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_ExecutionContext* ctx;
} XLA_FFI_ExecutionContext_GetModuleState_Args;

const size_t XLA_FFI_ExecutionContext_GetModuleState_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_ExecutionContext_GetModuleState_Args, ctx);

typedef XLA_FFI_Module_State* XLA_FFI_ExecutionContext_GetModuleState(
    XLA_FFI_ExecutionContext_GetModuleState_Args* args);

// Get `XLA_FFI_Stream` from the execution context.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_ExecutionContext* ctx;
} XLA_FFI_ExecutionContext_GetStream_Args;

const size_t XLA_FFI_ExecutionContext_GetStream_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_ExecutionContext_GetStream_Args, ctx);

typedef XLA_FFI_Stream* XLA_FFI_ExecutionContext_GetStream(
    XLA_FFI_ExecutionContext_GetStream_Args* args);

//===----------------------------------------------------------------------===//
// XLA FFI Function API.
//===----------------------------------------------------------------------===//

// Arguments passed to an FFI function.
typedef struct {
  size_t struct_size;
  void* priv;
  const XLA_FFI_Api* api;
  XLA_FFI_ExecutionContext* ctx;
  void** args;
  void** attrs;
  void** rets;
} XLA_FFI_Function_Args;

const size_t XLA_FFI_Function_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Function_Args, rets);

// XLA FFI function type that can be exported to a runtime.
typedef XLA_FFI_Error* XLA_FFI_Function(XLA_FFI_Function_Args* args);

//===----------------------------------------------------------------------===//
// XLA FFI Api.
//===----------------------------------------------------------------------===//

// Register FFI module with an XLA runtime.
typedef struct {
  size_t struct_size;
  void* priv;
  const char* name;
  XLA_FFI_Module* module;
  XLA_FFI_Module_StateType state_type;
  XLA_FFI_Module_CreateState* create_state;
  XLA_FFI_Module_DestroyState* destroy_state;
  int64_t num_exported_functions;
  const char** exported_names;            // length == num_exported_functions
  XLA_FFI_Function** exported_functions;  // length == num_exported_functions
} XLA_FFI_Module_Register_Args;

const size_t XLA_FFI_Module_Register_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Module_Register_Args, exported_functions);

typedef void XLA_FFI_Module_Register(XLA_FFI_Module_Register_Args* args);

#define XLA_FFI_API_STRUCT_FIELD(fn_type) fn_type* fn_type

#define XLA_FFI_API_TYPEID_FIELD(type) \
  XLA_FFI_Get_TypeId* XLA_FFI_Get_##type##_TypeId

typedef struct XLA_FFI_Api {
  size_t struct_size;
  void* priv;

  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Module_Register);

  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ExecutionContext_GetModuleState);
  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ExecutionContext_GetStream);

  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_Error_Create);

  XLA_FFI_API_TYPEID_FIELD(String);
  XLA_FFI_API_TYPEID_FIELD(Float);
  XLA_FFI_API_TYPEID_FIELD(Double);
  XLA_FFI_API_TYPEID_FIELD(Int1);
  XLA_FFI_API_TYPEID_FIELD(Int32);
  XLA_FFI_API_TYPEID_FIELD(Int64);

  XLA_FFI_API_TYPEID_FIELD(FloatArray);
  XLA_FFI_API_TYPEID_FIELD(DoubleArray);
  XLA_FFI_API_TYPEID_FIELD(Int32Array);
  XLA_FFI_API_TYPEID_FIELD(Int64Array);

  XLA_FFI_API_TYPEID_FIELD(FloatTensor);
  XLA_FFI_API_TYPEID_FIELD(DoubleTensor);
  XLA_FFI_API_TYPEID_FIELD(Int32Tensor);
  XLA_FFI_API_TYPEID_FIELD(Int64Tensor);

  XLA_FFI_API_TYPEID_FIELD(BufferArg);
  XLA_FFI_API_TYPEID_FIELD(StridedBufferArg);
  XLA_FFI_API_TYPEID_FIELD(Dictionary);
} XLA_FFI_Api;

#undef XLA_FFI_API_STRUCT_FIELD
#undef XLA_FFI_API_TYPEID_FIELD

const size_t XLA_FFI_Api_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Api, XLA_FFI_Get_StridedBufferArg_TypeId);

// Does not pass ownership of returned XLA_FFI_Api* to caller.
const XLA_FFI_Api* GetXlaFfiApi();

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_C_API_H_
