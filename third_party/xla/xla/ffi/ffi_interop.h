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

#ifndef XLA_FFI_FFI_INTEROP_H_
#define XLA_FFI_FFI_INTEROP_H_

#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

namespace xla::ffi {

// A library for XLA:FFI/XLA interop: converting XLA:FFI C structs into more
// user-friendly C++ types used in XLA. This library hides XLA:FFI details and
// C API structs memory layout from the users.

// Takes ownership of the XLA FFI error and returns underlying status. Frees
// `error` if it's not nullptr. If `error` is nullptr, returns OkStatus.
absl::Status TakeStatus(XLA_FFI_Error* error);

// Takes ownership of the XLA FFI future and returns underlying AsyncValue.
// Frees `future` if it's not nullptr. If `future` is nullptr, returns available
// async value.
tsl::AsyncValueRef<tsl::Chain> TakeFuture(XLA_FFI_Future* future);

}  // namespace xla::ffi

#endif  // XLA_FFI_FFI_INTEROP_H_
