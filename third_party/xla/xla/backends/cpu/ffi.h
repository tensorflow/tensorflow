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

#ifndef XLA_BACKENDS_CPU_FFI_H_
#define XLA_BACKENDS_CPU_FFI_H_

#include <optional>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/c_api_internal.h"  // IWYU pragma: keep
#include "xla/ffi/ffi.h"  // IWYU pragma: export

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// Type tags to bind parameters passed via execution context to FFI handler
//===----------------------------------------------------------------------===//

struct IntraOpThreadPool {};  // binds `const Eigen::ThreadPoolDevice*`

//===----------------------------------------------------------------------===//
// Context decoding
//===----------------------------------------------------------------------===//

template <>
struct CtxDecoding<IntraOpThreadPool> {
  using Type = const Eigen::ThreadPoolDevice*;

  static std::optional<Type> Decode(const XLA_FFI_Api* api,
                                    XLA_FFI_ExecutionContext* ctx,
                                    DiagnosticEngine& diagnostic) {
    return internal::DecodeInternalCtx<Type>(
        api, ctx, diagnostic,
        api->internal_api->XLA_FFI_INTERNAL_IntraOpThreadPool_Get,
        "intra op thread pool");
  }
};

}  // namespace xla::ffi

#endif  // XLA_BACKENDS_CPU_FFI_H_
