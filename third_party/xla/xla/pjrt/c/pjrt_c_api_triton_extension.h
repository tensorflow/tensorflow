/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_TRITON_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_TRITON_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_TRITON_EXTENSION_VERSION 1

struct PJRT_Triton_Compile_Args {
  size_t struct_size;
  const char* module;
  size_t module_size;
  const char* arch_name;
  size_t arch_name_size;
  int num_warps;
  int num_ctas;
  int num_stages;
  const char* out_asm;  // owned
  size_t out_asm_size;
  int64_t out_smem_bytes;
  int out_cluster_dim_x;
  int out_cluster_dim_y;
  int out_cluster_dim_z;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Triton_Compile_Args, out_cluster_dim_z);

// Compiles a given Triton kernel.
typedef PJRT_Error* PJRT_Triton_Compile(PJRT_Triton_Compile_Args* args);

typedef struct PJRT_Triton_Extension {
  PJRT_Extension_Base base;
  PJRT_Triton_Compile* compile;
} PJRT_Triton;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Triton_Extension, compile);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_TRITON_EXTENSION_H_
