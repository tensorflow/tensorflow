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

#ifndef XLA_PJRT_C_PJRT_C_API_GPU_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_GPU_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_GPU_EXTENSION_VERSION 4

struct PJRT_Gpu_Register_Custom_Call_Args {
  size_t struct_size;
  const char* function_name;
  size_t function_name_size;
  int api_version;  // 0 for an untyped call, 1 -- for typed
  void* handler_instantiate;
  void* handler_prepare;
  void* handler_initialize;
  void* handler_execute;
  // XLA_FFI_Handler_TraitsBits, honored by custom_call when the extension's
  // custom_call_handles_traits is set (both added in version 3).
  uint32_t traits;
  // Optional XLA_FFI_Handler for RECORD. Honored if custom_call_handles_record
  // is set (both added in version 4).
  void* handler_record;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Register_Custom_Call_Args, handler_record);

// Registers a custom call. Honors args->traits and args->handler_record iff the
// respective custom_call_handles_* capability (below) is set.
typedef PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args);

typedef struct PJRT_Gpu_Custom_Call {
  PJRT_Extension_Base base;
  PJRT_NO_DISCARD PJRT_Gpu_Register_Custom_Call* custom_call;
  // True if custom_call reads args->traits. Added in version 3; callers detect
  // support by checking base.struct_size covers this member and that it is set.
  bool custom_call_handles_traits;
  // True if custom_call reads args->handler_record. Added in version 4; callers
  // check that base.struct_size covers this member and that it is set.
  bool custom_call_handles_record;
} PJRT_Gpu_Custom_Call;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Custom_Call, custom_call_handles_record);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_GPU_EXTENSION_H_
