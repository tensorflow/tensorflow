/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_FFI_API_KERNEL_C_API_H_
#define XLA_FFI_API_KERNEL_C_API_H_

// IMPORTANT: Any ABI-breaking change to a struct, enum, or type definition in
// this file (such as changing struct field order/layout or removing/renumbering
// enumerators) requires bumping the version macros of ALL FFI extensions that
// include and use these shared kernel definitions (e.g.,
// XLA_FFI_Extension_Record in record_c_api.h).

#include <stddef.h>
#include <stdint.h>

#include "xla/ffi/api/c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Shared Kernel Types
//===----------------------------------------------------------------------===//

typedef enum XLA_FFI_SourceFormat {
  XLA_FFI_SourceFormat_FUNCTION_PTR = 2,
} XLA_FFI_SourceFormat;

typedef struct XLA_FFI_KernelSpec {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_ByteSpan name;
  const void* data;
  size_t size;
  XLA_FFI_SourceFormat format;
} XLA_FFI_KernelSpec;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_KernelSpec, format);

typedef struct XLA_FFI_Dim3 {
  int32_t x;
  int32_t y;
  int32_t z;
} XLA_FFI_Dim3;

typedef struct XLA_FFI_LaunchDims {
  XLA_FFI_Dim3 grid;
  XLA_FFI_Dim3 block;
  // Cluster dims are optional, 0, 0, 0 is considered as no cluster dims.
  XLA_FFI_Dim3 cluster;
} XLA_FFI_LaunchDims;

typedef enum XLA_FFI_KernelArgType {
  XLA_FFI_KernelArgType_DevicePtr = 0,
  XLA_FFI_KernelArgType_HostValue = 1,
} XLA_FFI_KernelArgType;

// A kernel argument is either a pointer to a device buffer or a host value.
// For a pointer to a device buffer, the size must be 0.
// For a host value, the size must be non-zero. In this case, XLA will pass the
// argument by value to the kernel.
typedef struct XLA_FFI_KernelArg {
  const void* arg_address;
  int64_t size;
  XLA_FFI_KernelArgType type;
} XLA_FFI_KernelArg;

typedef struct XLA_FFI_KernelArgs {
  const XLA_FFI_KernelArg* args;
  int64_t num_args;
} XLA_FFI_KernelArgs;

typedef struct XLA_FFI_KernelLaunchParams {
  size_t struct_size;
  XLA_FFI_InternalExtension* extension_start;

  XLA_FFI_LaunchDims launch_dims;
  uint32_t shared_mem_bytes;
  int32_t uses_pdl;
  const XLA_FFI_KernelArgs* args;
} XLA_FFI_KernelLaunchParams;

XLA_FFI_DEFINE_STRUCT_TRAITS(XLA_FFI_KernelLaunchParams, args);

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_KERNEL_C_API_H_
