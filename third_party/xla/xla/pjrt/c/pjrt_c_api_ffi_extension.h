/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_FFI_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_FFI_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// PJRT FFI extension provides capabilities for integrating with a
// backend-specific FFI (foreign function interface) library, i.e. for XLA CPU
// and GPU backends it gives access to the XLA FFI internals.
//
// See: https://en.wikipedia.org/wiki/Foreign_function_interface
#define PJRT_API_FFI_EXTENSION_VERSION 2

struct PJRT_FFI_TypeID_Register_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  const char* type_name;
  size_t type_name_size;
  int64_t type_id;  // in-out
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_FFI_TypeID_Register_Args, type_id);

// Registers external type in a static type registry. If `type_id` is set to `0`
// XLA will assign a unique type id to it and return via out argument, otherwise
// it will verify that user-provided type id matches previously registered type
// id for the given type name.
typedef PJRT_Error* PJRT_FFI_TypeID_Register(
    PJRT_FFI_TypeID_Register_Args* args);

// User-data that will be forwarded to the FFI handlers. Deleter is optional,
// and can be nullptr. Deleter will be called when the context is destroyed.
typedef struct PJRT_FFI_UserData {
  int64_t type_id;
  void* data;
  void (*deleter)(void* data);
} PJRT_FFI_UserData;

struct PJRT_FFI_UserData_Add_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  PJRT_ExecuteContext* context;
  PJRT_FFI_UserData user_data;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_FFI_UserData_Add_Args, user_data);

// Adds a user data to the execute context.
typedef PJRT_Error* PJRT_FFI_UserData_Add(PJRT_FFI_UserData_Add_Args* args);

typedef enum PJRT_FFI_Handler_TraitsBits {
  PJRT_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE = 1u << 0,
} PJRT_FFI_Handler_TraitsBits;

struct PJRT_FFI_Register_Handler_Args {
  size_t struct_size;
  const char* target_name;
  size_t target_name_size;
  void* handler;  // XLA_FFI_Handler* for typed FFI calls
  const char* platform_name;
  size_t platform_name_size;
  PJRT_FFI_Handler_TraitsBits traits;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_FFI_Register_Handler_Args, traits);

// Registers an FFI call handler for a specific platform.
// Only supports typed FFI handlers (XLA_FFI_Handler*).
typedef PJRT_Error* PJRT_FFI_Register_Handler(
    PJRT_FFI_Register_Handler_Args* args);

typedef struct PJRT_FFI_Extension {
  PJRT_Extension_Base base;
  PJRT_FFI_TypeID_Register* type_id_register;
  PJRT_FFI_UserData_Add* user_data_add;
  PJRT_FFI_Register_Handler* register_handler;
} PJRT_FFI;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_FFI_Extension, register_handler);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_FFI_EXTENSION_H_
