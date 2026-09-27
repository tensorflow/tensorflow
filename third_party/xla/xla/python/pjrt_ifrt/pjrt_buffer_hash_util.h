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

#ifndef XLA_PYTHON_PJRT_IFRT_PJRT_BUFFER_HASH_UTIL_H_
#define XLA_PYTHON_PJRT_IFRT_PJRT_BUFFER_HASH_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/future.h"

namespace xla {
namespace ifrt {

// Computes the hash of `PjRtBuffer`s.
//
// `index_domains` should be the index domains of the buffers in the same order
// as `buffers`.
//
// `max_inflight_memory` limits the total amount of host buffer memory that can
// be used by parallel fetch and hash operations. Currently, the fetches are
// done on a per-buffer basis; if a single buffer is larger than this limit, the
// actual memory use will be larger. This limit is applied only within a single
// call to `HashPjRtBuffers()`.
//
// Once it returns, `buffers` may be donated or deleted.
tsl::Future<std::vector<uint64_t>> HashPjRtBuffers(
    tsl::Executor& executor, absl::Span<PjRtBuffer* const> buffers,
    absl::Span<const IndexDomain> index_domains, Client::HashMode mode,
    int64_t max_inflight_memory = 128 * 1024 * 1024);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_PJRT_IFRT_PJRT_BUFFER_HASH_UTIL_H_
