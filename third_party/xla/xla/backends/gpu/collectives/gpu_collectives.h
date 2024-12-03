/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_

#include <cstdint>

#include "xla/core/collectives/collectives.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// In XLA:GPU we use different streams for different kinds of collective
// operations, and include the async stream kind into the GPU clique key.
//
// We carefully isolate different kinds of collectives using separate
// communicators and guarantee that all collective operations have a total order
// that will not create a deadlock.
enum class AsyncStreamKind : int64_t {
  kCollective = 0,  // Stream for asynchronous collective ops.
  kP2P0 = 1,        // One Stream for P2P Send and Recv ops.
  kP2P1 = 2,        // Another Stream for P2P Send and Recv ops.
  kMemCpyP2P = 3,   // Stream for MemCpyP2P
};

inline constexpr int64_t kAsyncStreamTotal =
    static_cast<int64_t>(AsyncStreamKind::kMemCpyP2P) + 1;

// Strongly-typed wrapper to represent collective stream ID.
TSL_LIB_GTL_DEFINE_INT_TYPE(CollectiveStreamId, uint64_t);

// Assigns a unique ID to a stream for asynchronous or synchronous execution.
// These IDs can be used, for example, to look up the NCCL communicator.
CollectiveStreamId GetCollectiveStreamId(
    bool is_async, AsyncStreamKind stream_kind = AsyncStreamKind::kCollective);

// XLA:GPU extension of the Collectives interface with GPU-specific APIs.
class GpuCollectives : public Collectives {
 public:
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_COLLECTIVES_H_
