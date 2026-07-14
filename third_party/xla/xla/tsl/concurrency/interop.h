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

#ifndef XLA_TSL_CONCURRENCY_INTEROP_H_
#define XLA_TSL_CONCURRENCY_INTEROP_H_

#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace tsl {

// In all APIs below, the returned `Future<>` will become ready on a thread that
// makes the last pending async value available. To avoid running `OnReady`
// callbacks on the wrong thread use `Detach` to jump into a separate executor.

// Returns a future that becomes ready when `value` becomes available. If the
// async value becomes an error, the returned future will also be an error.
[[nodiscard]] Future<> MakeFutureWhenReady(AsyncValue* value);
[[nodiscard]] Future<> MakeFutureWhenReady(
    const RCReference<AsyncValue>& value);

// Returns a future that becomes ready when all `values` become available. If
// any of the async values become an error, returned future will also be an
// error. The first encountered async value error will be returned via the
// future.
[[nodiscard]] Future<> MakeFutureWhenReady(
    absl::Span<AsyncValue* const> values);
[[nodiscard]] Future<> MakeFutureWhenReady(
    absl::Span<const RCReference<AsyncValue>> values);

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_INTEROP_H_
