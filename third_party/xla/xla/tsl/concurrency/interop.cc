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

#include "xla/tsl/concurrency/interop.h"

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace tsl {

static Future<> MakeFutureWhenValuesReady(
    std::vector<RCReference<AsyncValue>> values) {
  auto [promise, future] = MakePromise<>();

  absl::Span<const RCReference<AsyncValue>> span(values);
  RunWhenReady(span, [promise = std::move(promise),
                      values = std::move(values)]() mutable {
    for (const auto& value : values) {
      if (value->IsError()) {
        promise.Set(value->GetError());
        return;
      }
    }
    promise.Set(absl::OkStatus());
  });

  return std::move(future);
}

Future<> MakeFutureWhenReady(AsyncValue* value) {
  return MakeFutureWhenReady(absl::MakeSpan(&value, 1));
}

Future<> MakeFutureWhenReady(const RCReference<AsyncValue>& value) {
  return MakeFutureWhenReady(value.get());
}

Future<> MakeFutureWhenReady(absl::Span<AsyncValue* const> values) {
  std::vector<RCReference<AsyncValue>> captured;
  captured.reserve(values.size());
  for (auto* v : values) {
    captured.push_back(FormRef(v));
  }
  return MakeFutureWhenValuesReady(std::move(captured));
}

Future<> MakeFutureWhenReady(absl::Span<const RCReference<AsyncValue>> values) {
  return MakeFutureWhenValuesReady({values.begin(), values.end()});
}

}  // namespace tsl
