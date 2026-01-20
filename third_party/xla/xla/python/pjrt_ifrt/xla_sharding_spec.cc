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

#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char XlaCompatibleShardingSpec::ID = 0;  // NOLINT
char HloShardingSpec::ID = 0;            // NOLINT

namespace {

// Generates IndexDomains for an HloShardingSpec, using XLA HloSharding APIs.
// Note that this is O(N^2) where N is the number of devices (shards).
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
        << num_shards << " vs. " << xla_hlo_sharding.num_devices();
  }
  return std::unique_ptr<HloShardingSpec>(
      new HloShardingSpec(num_shards, std::move(xla_hlo_sharding)));
}

HloShardingSpec::HloShardingSpec(int num_shards,
                                 xla::HloSharding xla_hlo_sharding)
    : llvm::RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      xla_hlo_sharding_(std::move(xla_hlo_sharding)) {
  is_fully_replicated_ =
      xla_hlo_sharding_.IsReplicated() ||
      ((xla_hlo_sharding_.IsTiled() || xla_hlo_sharding_.IsTileMaximal()) &&
       num_shards_ == 1);
}

absl::StatusOr<Shape> HloShardingSpec::GetShardShape(const Shape& shape) const {
  if (xla_hlo_sharding_.IsTileMaximal() || xla_hlo_sharding_.IsManual() ||
      xla_hlo_sharding_.IsUnreduced() || xla_hlo_sharding_.IsUnknown()) {
    return shape;
  }
  if (shape.dims().size() != xla_hlo_sharding_.TiledDataRank()) {
    return InvalidArgument(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.dims().size(), xla_hlo_sharding_.TiledDataRank());
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
  const auto* other_hlo_sharding_spec = llvm::dyn_cast<HloShardingSpec>(&other);
  if (!other_hlo_sharding_spec) {
    return false;
  }
  return xla_hlo_sharding_ == other_hlo_sharding_spec->xla_hlo_sharding_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
HloShardingSpec::Disassemble(const Shape& shape) const {
  bool is_even_sharding = false;
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal() ||
      xla_hlo_sharding_.IsUnreduced()) {
    is_even_sharding = true;
  } else if (xla_hlo_sharding_.IsTiled()) {
    const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%s, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape.DebugString(),
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
    TF_ASSIGN_OR_RETURN(Shape shard_shape, GetShardShape(shape));
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

  TF_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                      IndexDomains(shape));
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
  return InvalidArgument(
      "HloShardingSpec can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> HloShardingSpec::IndexDomains(
    const Shape& shape) const {
  std::vector<IndexDomain> result;

  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        "Manual sharding does not support IndexDomains");
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(
        "Unreduced sharding does not support IndexDomains");
  }
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    result.resize(/*count=*/num_shards_, /*value=*/element);
    return result;
  }
  if (!xla_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, num_shards_, shape);
  }
  for (const xla::OpSharding::Type subgroup_type :
       xla_hlo_sharding_.subgroup_types()) {
    if (subgroup_type != xla::OpSharding::REPLICATED) {
      return IndexDomainsSlowPath(xla_hlo_sharding_, num_shards_, shape);
    }
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%s, sharding=%s",
                        tiled_data_rank, shape.dims().size(),
                        shape.DebugString(), xla_hlo_sharding_.ToString()));
  }

  TF_ASSIGN_OR_RETURN(Shape tile_shape, GetShardShape(shape));

  const absl::Span<const int64_t> shape_dims = shape.dims();
  std::vector<std::optional<IndexDomain>> all(num_shards_);
  TF_RETURN_IF_ERROR(xla_hlo_sharding_.EachTile(
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
