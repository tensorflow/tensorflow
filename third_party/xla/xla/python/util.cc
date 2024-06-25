/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/util.h"

#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/value.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"

namespace xla {

absl::Status AwaitBuffersReady(absl::Span<ifrt::Array* const> ifrt_arrays) {
  if (ifrt_arrays.empty()) {
    return absl::OkStatus();
  }

  ifrt::Future<> future;
  if (ifrt_arrays.size() == 1) {
    future = ifrt_arrays[0]->GetReadyFuture();
  } else {
    std::vector<tsl::RCReference<ifrt::Value>> values;
    values.reserve(ifrt_arrays.size());
    for (ifrt::Array* const ifrt_array : ifrt_arrays) {
      values.push_back(tsl::FormRef(ifrt_array));
    }
    ifrt::Client* const client = ifrt_arrays.front()->client();
    future = client->GetReadyFuture(values);
  }

  absl::Status s = future.Await();
  if (!s.ok()) {
    // Fix up error string because some clients rely on it.
    if (s.message() == "GetReadyFuture() called on deleted or donated buffer") {
      s = InvalidArgument(
          "BlockHostUntilReady() called on deleted or donated buffer");
    }
  }
  return s;
}

}  // namespace xla
