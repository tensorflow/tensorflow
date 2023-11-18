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

#ifndef XLA_FFI_API_FFI_H_
#define XLA_FFI_API_FFI_H_

#ifdef TENSORFLOW_COMPILER_XLA_FFI_FFI_H_
#error Two different XLA FFI implementations cannot be included together
#endif  // XLA_FFI_API_H_

#include <optional>
#include <type_traits>

#include "xla/ffi/api/c_api.h"

// IWYU pragma: begin_exports
#include "xla/ffi/api/api.h"
// IWYU pragma: end_exports

namespace xla::ffi {

namespace internal {
// TODO(ezhulenev): We need to log error message somewhere, currently we
// silently destroy it.
inline void DestroyError(const XLA_FFI_Api* api, XLA_FFI_Error* error) {
  XLA_FFI_Error_Destroy_Args destroy_args;
  destroy_args.struct_size = XLA_FFI_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.priv = nullptr;
  destroy_args.error = error;
  api->XLA_FFI_Error_Destroy(&destroy_args);
}
}  // namespace internal

//===----------------------------------------------------------------------===//
// PlatformStream
//===----------------------------------------------------------------------===//

template <typename T>
struct PlatformStream {};

template <typename T>
struct CtxDecoding<PlatformStream<T>> {
  using Type = T;

  static_assert(std::is_pointer_v<T>, "stream type must be a pointer");

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx) {
    XLA_FFI_Stream_Get_Args args;
    args.struct_size = XLA_FFI_Stream_Get_Args_STRUCT_SIZE;
    args.priv = nullptr;
    args.ctx = ctx;
    args.stream = nullptr;

    if (XLA_FFI_Error* error = api->XLA_FFI_Stream_Get(&args); error) {
      internal::DestroyError(api, error);
      return std::nullopt;
    }

    return reinterpret_cast<T>(args.stream);
  }
};

}  // namespace xla::ffi

#endif  // XLA_FFI_API_FFI_H_
