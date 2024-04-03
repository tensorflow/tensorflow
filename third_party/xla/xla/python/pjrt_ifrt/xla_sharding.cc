/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/pjrt_ifrt/xla_sharding.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char XlaCompatibleSharding::ID = 0;  // NOLINT
char HloSharding::ID = 0;            // NOLINT

namespace {

// Advances the specified set of indexes and returns true if we haven't
// wrapped around (i.e. result isn't {0, 0, ...}).
bool NextIndex(Index::Elements* index, absl::Span<const int64_t> limit) {
  DCHECK_LE(index->size(), limit.size());
  for (int64_t i = index->size() - 1; i >= 0; --i) {
    ++(*index)[i];
    if ((*index)[i] < limit[i]) {
      return true;
    }
    (*index)[i] = 0;
  }
  return false;
}

// Generates IndexDomains for an HloSharding, using XLA HloSharding APIs.
// Note that this is O(N^2) where N is the number of devices (shards).
std::vector<IndexDomain> IndexDomainsSlowPath(
    const xla::HloSharding& hlo_sharding, const DeviceList& devices,
    const Shape& shape) {
  // Only shape dimensions are used.
  auto xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S32, shape.dims());
  if (devices.size() > 8) {
    LOG_FIRST_N(WARNING, 1)
        << "Taking a slow path for HloSharding::IndexDomains(). This will not "
           "scale for a large number of devices.";
  }

  std::vector<IndexDomain> result;
  result.reserve(devices.size());

  Index::Elements origin(shape.dims().size());
  Shape::Dimensions shard_shape(shape.dims().size());
  for (int device_idx = 0; device_idx < devices.size(); ++device_idx) {
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

std::unique_ptr<HloSharding> HloSharding::Create(
    DeviceList devices, MemoryKind memory_kind,
    xla::HloSharding xla_hlo_sharding) {
  return std::unique_ptr<HloSharding>(new HloSharding(
      std::move(devices), memory_kind, std::move(xla_hlo_sharding)));
}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(const Shape& shape) const {
  TF_ASSIGN_OR_RETURN(auto index_domains, IndexDomains(shape));
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    result.push_back({index_domains[i].shape(),
                      SingleDeviceSharding::Create(devices_[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "HloSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape) const {
  auto format_shape = [&] {
    return absl::StrCat("[", absl::StrJoin(shape.dims(), ","), "]");
  };

  std::vector<IndexDomain> result;
  const int num_devices = devices_.size();

  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    result.resize(/*count=*/num_devices, /*value=*/element);
    return result;
  }
  if (!xla_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape);
  }
  for (const xla::OpSharding::Type subgroup_type :
       xla_hlo_sharding_.subgroup_types()) {
    if (subgroup_type != xla::OpSharding::REPLICATED) {
      return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape);
    }
  }
  if (xla_hlo_sharding_.tile_assignment().num_elements() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile_assignment_devices and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        xla_hlo_sharding_.tile_assignment().num_elements(), num_devices,
        format_shape(), DebugString()));
  }
  if (xla_hlo_sharding_.TotalNumTiles() != num_devices) {
    return absl::InvalidArgumentError(
        absl::StrFormat("sharding's tile count and device count does not "
                        "match: %d vs. %d; shape=%s, sharding=%s",
                        xla_hlo_sharding_.TotalNumTiles(), num_devices,
                        format_shape(), xla_hlo_sharding_.ToString()));
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%s, sharding=%s",
                        tiled_data_rank, shape.dims().size(), format_shape(),
                        xla_hlo_sharding_.ToString()));
  }

  // At the high-level, tile_assignment_dims[i] describes the number of ways the
  // shape is partitioned along i-th dimension. Note that
  // tile_assignment_dims[i] with i >= shape.size() encodes other information
  // such as subgroups to express partial replication/sharding and other
  // semantics.  They do not participate in determining the tile origin and
  // shape.
  const absl::Span<const int64_t> tile_assignment_dims =
      xla_hlo_sharding_.tile_assignment().dimensions();

  // Get the tile shape. This shape represents the shape of all per-shard
  // buffers.
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(
        xla::CeilOfRatio(shape.dims()[i], tile_assignment_dims[i]));
  }

  const int64_t replication_dim = xla_hlo_sharding_.SubgroupReplicationDim();
  int64_t num_replicas;
  if (replication_dim == -1) {
    num_replicas = 1;
  } else {
    num_replicas = tile_assignment_dims[replication_dim];
  }

  // Enumerate over all indices of tiles. For instance, if tile_assignment_dims
  // is [3, 2], iterate over [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]].
  // If tile_assignment_dims includes replication, we only enumerate over the
  // sharding portion, and copy the same indices multiple times.
  Index::Elements unique_tile_index(shape.dims().size());
  std::vector<Index::Elements> origins(num_devices);
  Index::Elements origin(shape.dims().size());
  int64_t device_assignment_index = 0;
  do {
    for (int64_t i = 0; i < shape.dims().size(); ++i) {
      origin[i] =
          std::min(tile_shape[i] * unique_tile_index[i], shape.dims()[i]);
    }
    for (int64_t i = 0; i < num_replicas; ++i) {
      CHECK_LT(device_assignment_index, num_devices);
      const int64_t device_id = xla_hlo_sharding_.tile_assignment()
                                    .array()
                                    .data()[device_assignment_index];
      if (device_id < 0 || device_id >= num_devices) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Out of range device id in device_assignment: %d; "
                            "valid range: [0, %d)",
                            device_id, num_devices));
      }
      origins[device_id] = origin;
      ++device_assignment_index;
    }
  } while (NextIndex(&unique_tile_index, tile_assignment_dims));

  result.reserve(num_devices);
  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    Shape::Dimensions actual_tile_shape;
    actual_tile_shape.reserve(tile_shape.size());
    for (int i = 0; i < tile_shape.size(); ++i) {
      actual_tile_shape.push_back(
          std::min(tile_shape[i], shape.dims()[i] - origins[device_idx][i]));
    }
    result.push_back(IndexDomain(Index(origins[device_idx]),
                                 Shape(std::move(actual_tile_shape))));
  }
  return result;
}

std::string HloSharding::DebugString() const {
  return absl::StrFormat("HloSharding(memory_kind: %s, hlo_sharding: %s)",
                         memory_kind_.DebugString(),
                         xla_hlo_sharding_.ToString());
}

std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& hlo_sharding, const Shape& shape) {
  return IndexDomainsSlowPath(hlo_sharding.xla_hlo_sharding(),
                              hlo_sharding.devices(), shape);
}

}  // namespace ifrt
}  // namespace xla
