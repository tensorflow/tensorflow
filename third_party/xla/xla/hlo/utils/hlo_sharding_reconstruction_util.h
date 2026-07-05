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

#ifndef XLA_HLO_UTILS_HLO_SHARDING_RECONSTRUCTION_UTIL_H_
#define XLA_HLO_UTILS_HLO_SHARDING_RECONSTRUCTION_UTIL_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/literal.h"
#include "xla/shape.h"

namespace xla {

struct ShardTensor {
  // The logical shard ID.
  int64_t logical_shard_id;
  // The actual tensor data for this shard.
  std::shared_ptr<Literal> data;
};

struct ManualShardingInfo {
  // Groups shards by manual shard ID.
  // manual_shard_id is 0 if no manual sharding is present.
  absl::flat_hash_map<int64_t, std::vector<ShardTensor>> manual_shard_groups;
  // True if manual sharding is present.
  bool has_manual_sharding = false;
  // The adjusted sharding to use for unsharding within each manual group.
  HloSharding unshard_sharding;

  explicit ManualShardingInfo(HloSharding unshard_sharding)
      : unshard_sharding(std::move(unshard_sharding)) {}
};

// Factors out manual sharding from the given sharding and groups shards by
// manual shard ID.
absl::StatusOr<ManualShardingInfo> FactorManualSharding(
    absl::Span<const ShardTensor> shards, const HloSharding& sharding);

// Reconstructs a full XLA Literal from an array of ShardTensors according to
// its HloSharding.
//
// Assumptions:
// 1. Sharding is simple. We currently don't support tuple sharding.
// 2. Tensors sizes perfectly divide by `partitions` on each dimension. (Or
//    the shard data fits well in padded tiles. The logic handles some boundary
//    conditions using std::min).
// 3. For manual subgroups, it is assumed that data is replicated across the
//    manual dimension, and only the first shard in the manual subgroup is
//    processed for reconstruction.
absl::StatusOr<xla::Literal> UnshardLiteral(
    absl::Span<const ShardTensor> shards, const HloSharding& sharding,
    const Shape& unsharded_shape);

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_SHARDING_RECONSTRUCTION_UTIL_H_
