/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char XlaCompatibleShardingSpec::ID = 0;  // NOLINT
char HloShardingSpec::ID = 0;            // NOLINT

namespace {

// Generates IndexDomains for an HloShardingSpec, using XLA HloSharding APIs.
std::vector<IndexDomain> IndexDomainsSlowPath(
    const xla::HloSharding& hlo_sharding, int num_shards, const Shape& shape) {
  // Only shape dimensions are used.
  auto xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S32, shape.dims());
  if (num_shards > 8) {
    LOG_FIRST_N(WARNING, 1) << "Taking a slow path for "
                               "HloShardingSpec::IndexDomains(). This will not "
                               "scale for a large number of devices.";
  }

  std::vector<IndexDomain> result;
  result.reserve(num_shards);

  Index::Elements origin(shape.dims().size());
  Shape::Dimensions shard_shape(shape.dims().size());
  for (int device_idx = 0; device_idx < num_shards; ++device_idx) {
    auto tile_offset = hlo_sharding.TileOffsetForDevice(xla_shape, device_idx);
    auto tile_limit = hlo_sharding.TileLimitForDevice(xla_shape, device_idx);
    for (int i = 0; i < shape.dims().size(); ++i) {
      origin[i] = tile_offset[i];
      shard_shape[i] = tile_limit[i] - tile_offset[i];
    }
    result.push_back(IndexDomain(Index(origin), Shape(shard_shape)));
  }
  return result;
}

}  // namespace

std::unique_ptr<HloShardingSpec> HloShardingSpec::Create(
    int num_shards, xla::HloSharding xla_hlo_sharding) {
  if (!xla_hlo_sharding.IsReplicated() && !xla_hlo_sharding.IsUnreduced() &&
      xla_hlo_sharding.IsTiled()) {
    CHECK_EQ(num_shards, xla_hlo_sharding.num_devices())
        << "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not "
           "match: "
        << num_shards << " vs. " << xla_hlo_sharding.num_devices()
        << "; sharding=" << xla_hlo_sharding.ToString();
  }
  return std::unique_ptr<HloShardingSpec>(
      new HloShardingSpec(num_shards, std::move(xla_hlo_sharding)));
}

HloShardingSpec::HloShardingSpec(int num_shards,
                                 xla::HloSharding xla_hlo_sharding)
    : RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      xla_hlo_sharding_(std::move(xla_hlo_sharding)) {
  is_fully_replicated_ =
      xla_hlo_sharding_.IsReplicated() ||
      ((xla_hlo_sharding_.IsTiled() || xla_hlo_sharding_.IsSingleDevice()) &&
       num_shards_ == 1);
}

HloShardingSpec::HloShardingSpec(const HloShardingSpec& other)
    : RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec>(other),
      xla_hlo_sharding_(other.xla_hlo_sharding_),
      hash_(other.hash_.load(std::memory_order_relaxed)) {}

absl::StatusOr<ShardingRef> HloShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices->size() != num_shards()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec requires %d devices, but received %d devices; "
        "sharding=%s",
        num_shards(), devices->size(), xla_hlo_sharding_.ToString()));
  }
  std::shared_ptr<const HloShardingSpec> spec =
      std::static_pointer_cast<const HloShardingSpec>(weak_from_this().lock());
  if (spec == nullptr) {
    spec = HloShardingSpec::Create(num_shards(), xla_hlo_sharding());
  }
  return std::unique_ptr<HloSharding>(
      new HloSharding(std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> HloShardingSpec::GetShardShape(const Shape& shape) const {
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice() ||
      xla_hlo_sharding_.IsManual() || xla_hlo_sharding_.IsUnreduced() ||
      xla_hlo_sharding_.IsUnknown()) {
    return shape;
  }
  if (shape.dims().size() != xla_hlo_sharding_.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %v vs from "
        "HloSharding %s",
        shape, xla_hlo_sharding_.ToString()));
  }
  const absl::Span<const int64_t> sharding_dims =
      xla_hlo_sharding_.dimensions();
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(xla::CeilOfRatio(shape.dims()[i], sharding_dims[i]));
  }
  return Shape(std::move(tile_shape));
}

bool HloShardingSpec::HasSamePartitioning(const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  if (num_shards() != other.num_shards()) {
    return false;
  }
  const auto* other_hlo_sharding_spec = dyn_cast<HloShardingSpec>(&other);
  if (!other_hlo_sharding_spec) {
    return false;
  }
  return xla_hlo_sharding_ == other_hlo_sharding_spec->xla_hlo_sharding_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
HloShardingSpec::Disassemble(const Shape& shape) const {
  bool is_even_sharding = false;
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice() ||
      xla_hlo_sharding_.IsUnreduced()) {
    is_even_sharding = true;
  } else if (xla_hlo_sharding_.IsTiled()) {
    const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%v, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape,
          xla_hlo_sharding_.ToString()));
    }

    is_even_sharding = true;
    for (int i = 0; i < tiled_data_rank; ++i) {
      if (shape.dims()[i] % xla_hlo_sharding_.dimension(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (xla_hlo_sharding_.IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    is_even_sharding = true;
  }

  if (is_even_sharding) {
    ABSL_ASSIGN_OR_RETURN(Shape shard_shape, GetShardShape(shape));
    std::vector<std::pair<Shape, ShardingSpecRef>> result;
    result.reserve(num_shards_);
    for (int i = 0; i < num_shards_; ++i) {
      result.push_back({
          shard_shape,
          SingleDeviceShardingSpec::Create(),
      });
    }
    return result;
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains, IndexDomains(shape));
  CHECK_EQ(index_domains.size(), num_shards_);
  std::vector<std::pair<Shape, ShardingSpecRef>> result;
  result.reserve(num_shards_);
  for (int i = 0; i < index_domains.size(); ++i) {
    result.push_back({
        index_domains[i].shape(),
        SingleDeviceShardingSpec::Create(),
    });
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
HloShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return absl::InvalidArgumentError(absl::StrFormat(
      "HloShardingSpec can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::StatusOr<std::vector<IndexDomain>> HloShardingSpec::IndexDomains(
    const Shape& shape) const {
  std::vector<IndexDomain> result;

  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support IndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unreduced sharding does not support IndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    result.resize(/*count=*/num_shards_, /*value=*/element);
    return result;
  }
  if (!xla_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, num_shards_, shape);
  }
  if (xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, num_shards_, shape);
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%v, sharding=%s",
                        tiled_data_rank, shape.dims().size(), shape,
                        xla_hlo_sharding_.ToString()));
  }

  ABSL_ASSIGN_OR_RETURN(Shape tile_shape, GetShardShape(shape));

  const absl::Span<const int64_t> shape_dims = shape.dims();
  std::vector<std::optional<IndexDomain>> all(num_shards_);
  ABSL_RETURN_IF_ERROR(xla_hlo_sharding_.EachTile(
      shape_dims, [shape_dims, &all](int device_index,
                                     absl::Span<const int64_t> tile_offset,
                                     absl::Span<const int64_t> tile_limit) {
        Shape::Dimensions tile_shape;
        tile_shape.reserve(shape_dims.size());
        for (int i = 0; i < shape_dims.size(); ++i) {
          tile_shape.push_back(tile_limit[i] - tile_offset[i]);
        }
        all[device_index] =
            IndexDomain(Index(tile_offset), Shape(std::move(tile_shape)));
      }));

  result.reserve(num_shards_);
  for (int device_idx = 0; device_idx < num_shards_; ++device_idx) {
    result.push_back(*std::move(all[device_idx]));
  }

  return result;
}

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
HloShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support UniqueIndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unreduced sharding does not support UniqueIndexDomains: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Non-replicated subgroup (e.g., manual or unreduced subgroup) sharding "
        "does not support UniqueIndexDomains: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    absl::call_once(unique_shard_indices_once_, [this] {
      cached_shard_indices_.reserve(num_shards_);
      for (int i = 0; i < num_shards_; ++i) {
        cached_shard_indices_.push_back(i);
      }
    });
    return absl::InlinedVector<IndexDomainAndShardIndices, 1>{
        IndexDomainAndShardIndices{
            /*index_domain=*/IndexDomain(shape),
            /*shard_indices=*/absl::MakeConstSpan(cached_shard_indices_),
        },
    };
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%v, sharding=%s",
                        tiled_data_rank, shape.dims().size(), shape,
                        xla_hlo_sharding_.ToString()));
  }

  absl::call_once(unique_shard_indices_once_, [this] {
    const int64_t* flat_tile_assignment =
        xla_hlo_sharding_.tile_assignment().array().data();
    cached_shard_indices_.reserve(num_shards_);
    for (int64_t i = 0; i < num_shards_; ++i) {
      cached_shard_indices_.push_back(
          static_cast<int>(flat_tile_assignment[i]));
    }
  });

  const int64_t num_unique_tiles = xla_hlo_sharding_.NumTiles();
  if (num_shards_ % num_unique_tiles != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec has %d shards, but HloSharding has %d unique tiles, "
        "which is not a divisor of the number of shards; sharding=%s",
        num_shards_, num_unique_tiles, xla_hlo_sharding_.ToString()));
  }
  const int64_t num_replicas = num_shards_ / num_unique_tiles;

  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S32, shape.dims());
  absl::InlinedVector<IndexDomainAndShardIndices, 1> unique_domains;
  unique_domains.reserve(num_unique_tiles);
  for (int64_t tile_idx = 0; tile_idx < num_unique_tiles; ++tile_idx) {
    const int first_shard = cached_shard_indices_[tile_idx * num_replicas];
    std::vector<int64_t> tile_offset =
        xla_hlo_sharding_.TileOffsetForDevice(xla_shape, first_shard);
    std::vector<int64_t> tile_limit =
        xla_hlo_sharding_.TileLimitForDevice(xla_shape, first_shard);
    Index::Elements origin(shape.dims().size());
    Shape::Dimensions shard_shape(shape.dims().size());
    for (int i = 0; i < shape.dims().size(); ++i) {
      origin[i] = tile_offset[i];
      shard_shape[i] = tile_limit[i] - tile_offset[i];
    }
    unique_domains.push_back(IndexDomainAndShardIndices{
        /*index_domain=*/
        IndexDomain(Index(std::move(origin)), Shape(std::move(shard_shape))),
        /*shard_indices=*/
        absl::MakeConstSpan(cached_shard_indices_)
            .subspan(tile_idx * num_replicas, num_replicas),
    });
  }

  return unique_domains;
}

absl::StatusOr<absl::Span<const int>>
HloShardingSpec::ShardToUniqueIndexDomainIndex() const {
  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support "
                        "ShardToUniqueIndexDomainIndex: sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unreduced sharding does not support "
                        "ShardToUniqueIndexDomainIndex: sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Non-replicated subgroup (e.g., manual or unreduced subgroup) sharding "
        "does not support ShardToUniqueIndexDomainIndex: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    absl::call_once(shard_to_unique_index_domain_index_once_, [this] {
      cached_shard_to_unique_index_domain_index_.assign(num_shards_, 0);
    });
    return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
  }

  const int64_t num_unique_tiles = xla_hlo_sharding_.NumTiles();
  if (num_shards_ % num_unique_tiles != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec has %d shards, but HloSharding has %d unique tiles, "
        "which is not a divisor of the number of shards; sharding=%s",
        num_shards_, num_unique_tiles, xla_hlo_sharding_.ToString()));
  }
  const int64_t num_replicas = num_shards_ / num_unique_tiles;

  absl::call_once(shard_to_unique_index_domain_index_once_, [&, this] {
    cached_shard_to_unique_index_domain_index_.resize(num_shards_);
    const int64_t* flat_tile_assignment =
        xla_hlo_sharding_.tile_assignment().array().data();
    for (int64_t tile_idx = 0; tile_idx < num_unique_tiles; ++tile_idx) {
      const int64_t offset = tile_idx * num_replicas;
      for (int64_t i = 0; i < num_replicas; ++i) {
        const int device_idx =
            static_cast<int>(flat_tile_assignment[offset + i]);
        cached_shard_to_unique_index_domain_index_[device_idx] = tile_idx;
      }
    }
  });
  return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
}

std::string HloShardingSpec::DebugString() const {
  return absl::StrFormat("HloShardingSpec(num_shards: %d, hlo_sharding: %s)",
                         num_shards_, xla_hlo_sharding_.ToString());
}

void HloShardingSpec::Hash(absl::HashState state) const {
  uint64_t hash = hash_.load(std::memory_order_relaxed);
  if (hash == kUnsetHash) {
    hash = absl::HashOf(num_shards_, xla_hlo_sharding_);
    if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
      ++hash;
    }
    hash_.store(hash, std::memory_order_relaxed);
  }
  absl::HashState::combine(std::move(state), hash);
}

std::vector<IndexDomain> TEST_HloShardingSpecIndexDomainsSlowPath(
    const HloShardingSpec& sharding_spec, const Shape& shape) {
  return IndexDomainsSlowPath(sharding_spec.xla_hlo_sharding(),
                              sharding_spec.num_shards(), shape);
}

}  // namespace ifrt
}  // namespace xla
