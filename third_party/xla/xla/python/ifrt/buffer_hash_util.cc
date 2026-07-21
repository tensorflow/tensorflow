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

#include "xla/python/ifrt/buffer_hash_util.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "highwayhash/arch_specific.h"
#include "highwayhash/hh_types.h"
#include "highwayhash/highwayhash.h"
#include "xla/layout.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/fingerprint.h"

namespace xla {
namespace ifrt {

namespace {

// A HighwayHash function with a randomly generated fixed key.
uint64_t HighwayHash64(const char* data, size_t size) {
  static constexpr highwayhash::HHKey kKey = {
      0x82b924c02d5409baULL,
      0x3b2e1849380974a9ULL,
      0xac79a5b64f5559aeULL,
      0xf5d1bfa11e88f03eULL,
  };
  highwayhash::HHStateT<HH_TARGET> state(kKey);
  highwayhash::HHResult64 result;
  highwayhash::HighwayHashT(&state, data, size, &result);
  return result;
}

// Hashes a span of buffer that contains elements of integral type `T`.
template <typename T>
uint64_t HashBuffer(absl::Span<const T> buffer) {
  static_assert(std::is_integral_v<T>, "T must be an integral type.");
  return HighwayHash64(reinterpret_cast<const char*>(buffer.data()),
                       buffer.size() * sizeof(T));
}
}  // namespace

absl::StatusOr<uint64_t> HashBufferPhysical(absl::Span<const char> buffer,
                                            const xla::Layout& layout) {
  // Take the whole buffer and compute the hash.
  uint64_t hash = HashBuffer(buffer);

  // Combine with the layout hash.
  xla::LayoutProto layout_proto = layout.ToProto();
  std::string serialized_layout;
  TF_RET_CHECK(
      tsl::SerializeToStringDeterministic(layout_proto, &serialized_layout));
  uint64_t layout_hash = tsl::Fingerprint64(serialized_layout);
  return tsl::FingerprintCat64(hash, layout_hash);
}

absl::StatusOr<uint64_t> HashBufferLogical(absl::Span<const char> buffer,
                                           int64_t element_byte_size,
                                           const IndexDomain& index_domain) {
  // Each element is hashed with its coordinate. The resulting per-element hash
  // values can be combined using XOR in a commutative and associative way.
  const Shape& shard_shape = index_domain.shape();
  const int64_t num_elements = shard_shape.num_elements();
  if (buffer.size() != element_byte_size * num_elements) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Buffer byte size does not match the expected size. buffer byte size: ",
        buffer.size(), " vs. element byte size: ", element_byte_size,
        " and index domain: ", index_domain));
  }

  const Index& origin = index_domain.origin();
  const int num_dims = shard_shape.dims().size();

  uint64_t hash = 0;
  // Coordinates in the array.
  std::vector<int64_t> dims(origin.elements().begin(), origin.elements().end());
  std::vector<int64_t> limits(num_dims);
  for (int d = 0; d < num_dims; ++d) {
    limits[d] = origin.elements()[d] + shard_shape.dims()[d];
  }

  for (int64_t k = 0; k < num_elements; ++k) {
    uint64_t coord_hash = HashBuffer(absl::MakeConstSpan(dims));
    uint64_t element_hash =
        HashBuffer(buffer.subspan(k * element_byte_size, element_byte_size));
    hash ^= tsl::FingerprintCat64(coord_hash, element_hash);

    // Increment the multi-dimensional index in the array.
    for (int d = num_dims - 1; d >= 0; --d) {
      ++dims[d];
      if (dims[d] < limits[d]) {
        break;
      }
      dims[d] = origin.elements()[d];
    }
  }
  return hash;
}

std::vector<int> GetReplicaGroupIds(
    absl::Span<const IndexDomain> index_domains) {
  std::vector<int> replica_group_ids;
  replica_group_ids.reserve(index_domains.size());
  absl::flat_hash_map<IndexDomain, int> domain_to_group_id;
  for (const IndexDomain& index_domain : index_domains) {
    auto it =
        domain_to_group_id.try_emplace(index_domain, domain_to_group_id.size())
            .first;
    replica_group_ids.push_back(it->second);
  }
  return replica_group_ids;
}

absl::StatusOr<uint64_t> AggregateShardHashes(
    absl::Span<const uint64_t> shard_hashes,
    absl::Span<const int> replica_group_ids, Client::HashMode mode) {
  if (shard_hashes.size() != replica_group_ids.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Shard hashes and replica group IDs must have the same size, but have ",
        shard_hashes.size(), " vs. ", replica_group_ids.size()));
  }

  absl::btree_map<int, uint64_t> replica_group_id_to_hash;
  for (int i = 0; i < shard_hashes.size(); ++i) {
    const int replica_group_id = replica_group_ids[i];
    auto it =
        replica_group_id_to_hash.try_emplace(replica_group_id, shard_hashes[i])
            .first;
    if (it->second != shard_hashes[i]) {
      for (int j = 0; j < i; ++j) {
        if (replica_group_ids[j] == replica_group_id) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Hash mismatch across replicas for shards ", j, " and ", i,
              " (replica group ID ", replica_group_id,
              "). hashes: ", shard_hashes[j], " vs. ", shard_hashes[i]));
        }
      }
      // This should not happen unless there is memory corruption.
      CHECK(false) << "No previous shard found for replica group ID "
                   << replica_group_id;
    }
  }

  switch (mode) {
    case Client::HashMode::kPhysical: {
      std::vector<uint64_t> hashes;
      hashes.reserve(replica_group_id_to_hash.size());
      for (const auto& [unused_replica_group_id, shard_hash] :
           replica_group_id_to_hash) {
        hashes.push_back(shard_hash);
      }
      return HashBuffer(absl::MakeConstSpan(hashes));
    }
    case Client::HashMode::kLogical: {
      uint64_t aggregated_hash = 0;
      for (const auto& [_, shard_hash] : replica_group_id_to_hash) {
        aggregated_hash ^= shard_hash;
      }
      return aggregated_hash;
    }
  }
}

}  // namespace ifrt
}  // namespace xla
