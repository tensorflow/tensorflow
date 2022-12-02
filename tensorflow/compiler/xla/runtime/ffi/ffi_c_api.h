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
// XLA FFI registry for exporting FFI functions to runtime.
//===----------------------------------------------------------------------===//

typedef struct XLA_FFI_Registry XLA_FFI_Registry;

//===----------------------------------------------------------------------===//
// XLA FFI module defines a set of exported FFI functions and their state.
//===----------------------------------------------------------------------===//

// XLA FFI module is a way to structure FFI functions together with a state
// required for calling them. XLA runtime executable can be linked with multiple
// of such modules at run time.
typedef struct XLA_FFI_Module XLA_FFI_Module;

// When a module is instantiated for each executable it can optionally create a
// state object that can be used to implement stateful functions, to keep
// state between FFI functions invocations. State can be accessed from different
// executable invocations running concurrently.
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

// Exports module functions to the given FFI registry.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_Module* module;
  XLA_FFI_Registry* registry;
} XLA_FFI_Module_ExportFunctions_Args;

const size_t XLA_FFI_Module_ExportFunctions_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Module_ExportFunctions_Args, registry);

typedef void XLA_FFI_Module_ExportFunctions(
    XLA_FFI_Module_ExportFunctions_Args* args);

//===----------------------------------------------------------------------===//
// Execution context passed to FFI functions.
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

#define XLA_FFI_EXECUTION_CONTEXT_STRUCT_FIELD(fn_type) fn_type* fn_type

#define XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD(type) \
  XLA_FFI_Get_TypeId* XLA_FFI_Get_##type##_TypeId

typedef struct {
  XLA_FFI_EXECUTION_CONTEXT_STRUCT_FIELD(XLA_FFI_Error_Create);

  // Type ids for supported scalar types.
  XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD(Float);
  XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD(Int32);

  // Type ids for supported buffer types.
  XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD(BufferArg);
  XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD(StridedBufferArg);
} XLA_FFI_ExecutionContext;

#undef XLA_FFI_EXECUTION_CONTEXT_STRUCT_FIELD
#undef XLA_FFI_EXECUTION_CONTEXT_TYPEID_FIELD

//===----------------------------------------------------------------------===//
// XLA FFI Api.
//===----------------------------------------------------------------------===//

// Arguments passed to an FFI function.
typedef struct {
  size_t struct_size;
  void* priv;
  XLA_FFI_ExecutionContext* ctx;
  XLA_FFI_Module_State* state;
  void** args;
  void** attrs;
  void** rets;
} XLA_FFI_Function_Args;

const size_t XLA_FFI_Function_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_Function_Args, rets);

// XLA FFI function type that can be exported to a runtime.
typedef XLA_FFI_Error* XLA_FFI_Function(XLA_FFI_Function_Args* args);

// Register FFI module with an XLA runtime.
typedef struct {
  size_t struct_size;
  void* priv;
  const char* name;
  XLA_FFI_Module* module;
  XLA_FFI_Module_CreateState* create_state;
  XLA_FFI_Module_DestroyState* destroy_state;
  XLA_FFI_Module_ExportFunctions* export_functions;
} XLA_FFI_RegisterModule_Args;

const size_t XLA_FFI_RegisterModule_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_RegisterModule_Args, export_functions);

typedef void XLA_FFI_RegisterModule(XLA_FFI_RegisterModule_Args* args);

// Export FFI function from the module to an XLA runtime.
typedef struct {
  size_t struct_size;
  void* priv;
  const char* target;
  XLA_FFI_Function* function;
  XLA_FFI_Registry* registry;
} XLA_FFI_ExportFunction_Args;

const size_t XLA_FFI_ExportFunction_Args_STRUCT_SIZE =
    XLA_FFI_STRUCT_SIZE(XLA_FFI_ExportFunction_Args, registry);

typedef void XLA_FFI_ExportFunction(XLA_FFI_ExportFunction_Args* args);

#define XLA_FFI_API_STRUCT_FIELD(fn_type) fn_type* fn_type

typedef struct {
  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_RegisterModule);
  XLA_FFI_API_STRUCT_FIELD(XLA_FFI_ExportFunction);
} XLA_FFI_Api;

#undef XLA_FFI_API_STRUCT_FIELD

// Does not pass ownership of returned XLA_FFI_Api* to caller.
const XLA_FFI_Api* GetXlaFfiApi();

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_FFI_FFI_C_API_H_
