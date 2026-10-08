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

#ifndef XLA_BACKENDS_CPU_COLLECTIVES_GLOO_REDUCE_SCATTER_H_
#define XLA_BACKENDS_CPU_COLLECTIVES_GLOO_REDUCE_SCATTER_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "gloo/context.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/stream_executor/device_address.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Gloo reserves slot prefixes 0x01..0x08 for its built-in collectives
// (gloo/types.h). XLA uses 0x40+ for collectives implemented in XLA.
inline constexpr uint8_t kCollectivePermuteSlotPrefix = 0x40;
inline constexpr uint8_t kReduceScatterSlotPrefix = 0x41;

// Performs a halving-doubling reduce-scatter collective operation.
// We implement this ourselves because Gloo's built-in halving-doubling
// reduce-scatter is buggy and does not support timeouts.
//
// Arguments:
//   context: The Gloo communicator context (defines world size and rank).
//   send_buffer: Input buffer containing `count * context->size` elements of
//     type `dtype`. Reduction is not performed in-place.
//   recv_buffer: Output buffer receiving `count` reduced elements for this
//     rank. May alias or partially overlap `send_buffer`.
//   dtype: Element primitive type.
//   count: Number of output elements per rank (total input elements is
//     `count * context->size`).
//   reduction_kind: Reduction operator (SUM, PRODUCT, MIN, MAX).
//   timeout: Maximum duration allowed for the collective before failing with a
//     timeout error.
//   tag: Caller-supplied tag used to disambiguate concurrent collectives on the
//     same `context`.
absl::Status GlooReduceScatter(const std::shared_ptr<gloo::Context>& context,
                               se::DeviceAddressBase send_buffer,
                               se::DeviceAddressBase recv_buffer,
                               PrimitiveType dtype, size_t count,
                               ReductionKind reduction_kind,
                               absl::Duration timeout, uint32_t tag = 0);

namespace internal {
// Reverses the lowest `n` bits of `ctr`. Exposed for testing.
uint32_t ReverseLastNBits(uint32_t ctr, uint32_t n);
}  // namespace internal

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_COLLECTIVES_GLOO_REDUCE_SCATTER_H_
