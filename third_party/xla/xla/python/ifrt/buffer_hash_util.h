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

#ifndef XLA_PYTHON_IFRT_BUFFER_HASH_UTIL_H_
#define XLA_PYTHON_IFRT_BUFFER_HASH_UTIL_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index_domain.h"

namespace xla {

class Layout;

namespace ifrt {

// Computes the physical hash of a host buffer representing a shard, using raw
// buffer bytes and layout.
//
// `buffer` can be either a buffer with the original layout or a relayouted
// buffer with some canonical layout. For the latter, `layout` should be the
// original layout of the buffer before it is relayouted. Either way, `buffer`
// should have deterministic data (e.g., zeros) for padding if any because this
// function will not skip the padding when computing the hash.
absl::StatusOr<uint64_t> HashBufferPhysical(absl::Span<const char> buffer,
                                            const xla::Layout& layout);

// Computes the logical hash of a host buffer representing a shard.
//
// `buffer` is a dense buffer whose element is major-to-minor and unpacked (any
// sub-byte element occupies the full byte), and has no tiling or padding.
absl::StatusOr<uint64_t> HashBufferLogical(absl::Span<const char> buffer,
                                           int64_t element_byte_size,
                                           const IndexDomain& index_domain);

// Maps replica group IDs (0, 1, ..., M-1) for each `IndexDomain`, where
// identical domains are assigned the same ID.
//
// `index_domains` should use the same order as the corresponding shards of the
// original `Array` (i.e., the result from `Sharding::IndexDomains()`).
//
// Replica group IDs do not have a particular semantics and are currently only
// used for `AggregateShardHashes()`.
std::vector<int> GetReplicaGroupIds(
    absl::Span<const IndexDomain> index_domains);

// Aggregates hash values computed for all shards of a single `Array` into a
// single hash value.
//
// When there are multiple replicas, it will verify that all replicas have the
// same hash value and avoid duplicate counting.
//
// `shard_hashes` and `replica_group_ids` should be the same size and parallel.
// In general, it is expected that the order of their elements matches the
// corresponding shard of the original `Array` even though the aggregation
// method of some hash mode may not be order-dependent.
absl::StatusOr<uint64_t> AggregateShardHashes(
    absl::Span<const uint64_t> shard_hashes,
    absl::Span<const int> replica_group_ids, Client::HashMode mode);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_BUFFER_HASH_UTIL_H_
