/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_GPU_EXTENSION_VERSION 0

struct PJRT_Gpu_Register_Custom_Call_Args {
  size_t struct_size;
  const char* function_name;
  size_t function_name_size;
  void* custom_call_function;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Gpu_Register_Custom_Call_Args,
                          custom_call_function);

// Registers a custom call.
typedef PJRT_Error* PJRT_Gpu_Register_Custom_Call(
    PJRT_Gpu_Register_Custom_Call_Args* args);

typedef struct PJRT_Gpu_Custom_Call {
  PJRT_Structure_Type type;
  const void* next;
  PJRT_Gpu_Register_Custom_Call* custom_call;
} PJRT_Gpu_Custom_Call;

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_GPU_EXTENSION_H_
