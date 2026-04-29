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

#include "xla/ffi/ffi_interop.h"

#include <utility>

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/ffi_structs.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

namespace xla::ffi {

absl::Status TakeStatus(XLA_FFI_Error* error) {
  if (ABSL_PREDICT_TRUE(error == nullptr)) {
    return absl::OkStatus();
  }
  absl::Status status = std::move(error->status);
  delete error;
  return status;
}

tsl::AsyncValueRef<tsl::Chain> TakeFuture(XLA_FFI_Future* future) {
  // Non-reference-counted async value ref for synchronous FFI handlers.
  static tsl::AsyncValueOwningRef<tsl::Chain>* chain = [] {
    auto* storage = new tsl::internal::AsyncValueStorage<tsl::Chain>();
    return new tsl::AsyncValueOwningRef<tsl::Chain>(
        tsl::MakeAvailableAsyncValueRef<tsl::Chain>(*storage));
  }();

  if (ABSL_PREDICT_TRUE(future == nullptr)) {
    return chain->AsRef();
  }

  // If the future is already completed, immediately return the underlying
  // async value and delete the XLA_FFI_Future.
  if (ABSL_PREDICT_TRUE(future->async_value.IsAvailable())) {
    tsl::AsyncValueRef<tsl::Chain> async_value = std::move(future->async_value);
    delete future;
    return async_value;
  }

  // If the future is not completed, return a copy of the underlying async
  // value and keep XLA_FFI_Future alive until it is completed.
  tsl::AsyncValueRef<tsl::Chain> async_value = future->async_value;
  async_value.AndThen([future] { delete future; });
  return async_value;
}

}  // namespace xla::ffi
