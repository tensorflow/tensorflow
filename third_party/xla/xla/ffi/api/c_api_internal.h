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

#ifndef XLA_FFI_API_C_API_INTERNAL_H_
#define XLA_FFI_API_C_API_INTERNAL_H_

#include "xla/ffi/api/c_api.h"

// Internal XLA FFI API that gives access to XLA implementation details that
// should be used only for implementing FFI handlers statically linked into
// the binary. This API should be used only by XLA itself (to implement builtin
// custom calls), or libraries tightly coupled to XLA and built from exact same
// commit and using the same toolchain (e.g. jaxlib). Trying to use this API
// from a dynamically loaded shared library can lead to undefined behavior and
// likely impossible to debug run time crashes.

#ifdef __cplusplus
extern "C" {
#endif

// Because this is an internal XLA FFI API we use a slightly relaxed C API
// style and do not track the struct size, as we expect this API to be used
// only in statically linked binaries, and we do not need any backward or
// forward compatibility.

// Forwards `absl::Status` object pointed to by `status` to XLA FFI error
// (status left in moved-from state). Pointer ownership stays with the
// caller.
typedef XLA_FFI_Error* XLA_FFI_Error_Forward(void* status);

// Returns a pointer to `xla::ServiceExecutableRunOptions`.
typedef void* XLA_FFI_ServiceExecutableRunOptions_Get(
    XLA_FFI_ExecutionContext* ctx);

//===----------------------------------------------------------------------===//
// API access
//===----------------------------------------------------------------------===//

#define _XLA_FFI_INTERNAL_API_STRUCT_FIELD(fn_type) fn_type* fn_type

struct XLA_FFI_InternalApi {
  _XLA_FFI_INTERNAL_API_STRUCT_FIELD(XLA_FFI_Error_Forward);
  _XLA_FFI_INTERNAL_API_STRUCT_FIELD(XLA_FFI_ServiceExecutableRunOptions_Get);
};

#undef _XLA_FFI_INTERNAL_API_STRUCT_FIELD

#ifdef __cplusplus
}
#endif

#endif  // XLA_FFI_API_C_API_INTERNAL_H_
