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
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

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
    const xla::HloSharding& hlo_sharding,
    const tsl::RCReference<DeviceList>& devices, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  // Only shape dimensions are used.
  auto xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla::PrimitiveType::S32, shape.dims());
  if (devices->size() > 8) {
    LOG_FIRST_N(WARNING, 1)
        << "Taking a slow path for HloSharding::IndexDomains(). This will not "
           "scale for a large number of devices.";
  }

  std::vector<IndexDomain> result;
  result.reserve(devices->size());

  Index::Elements origin(shape.dims().size());
  Shape::Dimensions shard_shape(shape.dims().size());
  const absl::Span<Device* const> device_ptrs = devices->devices();
  for (int device_idx = 0; device_idx < device_ptrs.size(); ++device_idx) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        device_ptrs[device_idx]->IsAddressable()) {
      auto tile_offset =
          hlo_sharding.TileOffsetForDevice(xla_shape, device_idx);
      auto tile_limit = hlo_sharding.TileLimitForDevice(xla_shape, device_idx);
      for (int i = 0; i < shape.dims().size(); ++i) {
        origin[i] = tile_offset[i];
        shard_shape[i] = tile_limit[i] - tile_offset[i];
      }
      result.push_back(IndexDomain(Index(origin), Shape(shard_shape)));
    }
  }
  return result;
}

// Returns a canonicalized memory kind for the given devices.
// REQUIRES: !devices->devices().empty()
MemoryKind CanonicalizeMemoryKindWithDevices(
    const MemoryKind& memory_kind,
    const tsl::RCReference<DeviceList>& devices) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  return CanonicalizeMemoryKind(memory_kind, devices->devices().front());
}

}  // namespace

std::unique_ptr<HloSharding> HloSharding::Create(
    tsl::RCReference<DeviceList> devices, MemoryKind memory_kind,
    xla::HloSharding xla_hlo_sharding) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<HloSharding>(new HloSharding(
      std::move(devices), memory_kind, std::move(xla_hlo_sharding)));
}

HloSharding::HloSharding(tsl::RCReference<DeviceList> devices,
                         MemoryKind memory_kind,
                         xla::HloSharding xla_hlo_sharding)
    : llvm::RTTIExtends<HloSharding, XlaCompatibleSharding>(
          std::move(devices), memory_kind, xla_hlo_sharding.IsReplicated()),
      xla_hlo_sharding_(std::move(xla_hlo_sharding)) {}

absl::StatusOr<Shape> HloSharding::GetShardShape(const Shape& shape) const {
  if (xla_hlo_sharding_.IsTileMaximal() || xla_hlo_sharding_.IsManual() ||
      xla_hlo_sharding_.IsUnknown()) {
    return shape;
  }
  if (xla_hlo_sharding_.TotalNumTiles() != devices_->size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("sharding's tile count and device count does not "
                        "match: %d vs. %d; shape=%s, sharding=%s",
                        xla_hlo_sharding_.TotalNumTiles(), devices_->size(),
                        shape.DebugString(), xla_hlo_sharding_.ToString()));
  }
  if (shape.dims().size() != xla_hlo_sharding_.TiledDataRank()) {
    return InvalidArgument(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "HloSharding %d",
        shape.dims().size(), xla_hlo_sharding_.TiledDataRank());
  }
  const absl::Span<const int64_t> tile_assignment_dims =
      xla_hlo_sharding_.tile_assignment().dimensions();
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(
        xla::CeilOfRatio(shape.dims()[i], tile_assignment_dims[i]));
  }
  return Shape(std::move(tile_shape));
}

bool HloSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  if (devices()->size() != other.devices()->size()) {
    return false;
  }
  const auto* other_hlo_sharding = llvm::dyn_cast<HloSharding>(&other);
  if (!other_hlo_sharding) {
    return false;
  }
  return xla_hlo_sharding_ == other_hlo_sharding->xla_hlo_sharding_;
}

absl::StatusOr<std::unique_ptr<Sharding>> HloSharding::WithDeviceAssignment(
    std::optional<tsl::RCReference<DeviceList>> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return InvalidArgument(
        "HloSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size());
  }
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_),
                xla_hlo_sharding_);
}
absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  bool is_even_sharding = false;
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal()) {
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
      if (shape.dims()[i] % xla_hlo_sharding_.tile_assignment().dim(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (xla_hlo_sharding_.IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    is_even_sharding = true;
  }

  const absl::Span<Device* const> devices = devices_->devices();
  if (is_even_sharding) {
    // Fast path for even sharding.
    TF_ASSIGN_OR_RETURN(xla::ifrt::Shape shard_shape, GetShardShape(shape));
    std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.reserve(devices_->size());
    } else {
      result.reserve(devices_->AddressableDeviceList()->size());
    }
    for (int i = 0; i < devices_->size(); ++i) {
      if (single_device_shard_semantics ==
              SingleDeviceShardSemantics::kAllShards ||
          devices[i]->IsAddressable()) {
        result.push_back({
            shard_shape,
            SingleDeviceSharding::Create(devices[i], memory_kind_),
        });
      }
    }
    return result;
  } else {
    // Slow path that uses `IndexDomains()` to handle uneven sharding.
    TF_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                        IndexDomains(shape));
    CHECK_EQ(index_domains.size(), devices_->size());
    std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.reserve(devices_->size());
    } else {
      result.reserve(devices_->AddressableDeviceList()->size());
    }
    for (int i = 0; i < index_domains.size(); ++i) {
      if (single_device_shard_semantics ==
              SingleDeviceShardSemantics::kAllShards ||
          devices[i]->IsAddressable()) {
        result.push_back({
            index_domains[i].shape(),
            SingleDeviceSharding::Create(devices[i], memory_kind_),
        });
      }
    }
    return result;
  }
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
HloSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "HloSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  std::vector<IndexDomain> result;
  const int num_devices = devices_->size();

  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        "Manual sharding does not support IndexDomains");
  }
  if (xla_hlo_sharding_.IsReplicated() || xla_hlo_sharding_.IsTileMaximal()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.resize(/*count=*/num_devices, /*value=*/element);
    } else {
      result.resize(/*count=*/devices_->AddressableDeviceList()->size(),
                    /*value=*/element);
    }
    return result;
  }
  if (!xla_hlo_sharding_.IsTiled()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape,
                                single_device_shard_semantics);
  }
  for (const xla::OpSharding::Type subgroup_type :
       xla_hlo_sharding_.subgroup_types()) {
    if (subgroup_type != xla::OpSharding::REPLICATED) {
      return IndexDomainsSlowPath(xla_hlo_sharding_, devices_, shape,
                                  single_device_shard_semantics);
    }
  }
  if (xla_hlo_sharding_.tile_assignment().num_elements() != num_devices) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "sharding's tile_assignment_devices and device count does not "
        "match: %d vs. %d; shape=%s, sharding=%s",
        xla_hlo_sharding_.tile_assignment().num_elements(), num_devices,
        shape.DebugString(), DebugString()));
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%s, sharding=%s",
                        tiled_data_rank, shape.dims().size(),
                        shape.DebugString(), xla_hlo_sharding_.ToString()));
  }

  // Get the tile shape. This shape represents the shape of all per-shard
  // buffers.
  TF_ASSIGN_OR_RETURN(Shape tile_shape, GetShardShape(shape));
  const absl::Span<const int64_t> tile_shape_dims = tile_shape.dims();

  // At the high-level, tile_assignment_dims[i] describes the number of ways the
  // shape is partitioned along i-th dimension. Note that
  // tile_assignment_dims[i] with i >= shape.size() encodes other information
  // such as subgroups to express partial replication/sharding and other
  // semantics.  They do not participate in determining the tile origin and
  // shape.
  const absl::Span<const int64_t> tile_assignment_dims =
      xla_hlo_sharding_.tile_assignment().dimensions();

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
          std::min(tile_shape_dims[i] * unique_tile_index[i], shape.dims()[i]);
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

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(num_devices);
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  const absl::Span<Device* const> devices = devices_->devices();
  for (int device_idx = 0; device_idx < num_devices; ++device_idx) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        devices[device_idx]->IsAddressable()) {
      Shape::Dimensions actual_tile_shape;
      actual_tile_shape.reserve(tile_shape_dims.size());
      for (int i = 0; i < tile_shape_dims.size(); ++i) {
        actual_tile_shape.push_back(std::min(
            tile_shape_dims[i], shape.dims()[i] - origins[device_idx][i]));
      }
      result.push_back(IndexDomain(Index(origins[device_idx]),
                                   Shape(std::move(actual_tile_shape))));
    }
  }
  return result;
}

std::string HloSharding::DebugString() const {
  return absl::StrFormat("HloSharding(memory_kind: %v, hlo_sharding: %s)",
                         memory_kind_, xla_hlo_sharding_.ToString());
}

std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& hlo_sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  return IndexDomainsSlowPath(hlo_sharding.xla_hlo_sharding(),
                              hlo_sharding.devices(), shape,
                              single_device_shard_semantics);
}

}  // namespace ifrt
}  // namespace xla
