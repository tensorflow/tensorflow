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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"

namespace xla {
namespace ifrt {

char XlaCompatibleSharding::ID = 0;  // NOLINT
char HloSharding::ID = 0;            // NOLINT

std::unique_ptr<HloSharding> HloSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind,
    xla::HloSharding xla_hlo_sharding) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  int num_shards = devices->size();
  return std::unique_ptr<HloSharding>(new HloSharding(
      std::move(devices), memory_kind,
      HloShardingSpec::Create(num_shards, std::move(xla_hlo_sharding))));
}

HloSharding::HloSharding(DeviceListRef devices, MemoryKind memory_kind,
                         std::shared_ptr<const HloShardingSpec> sharding_spec)
    : RTTIExtends<HloSharding, XlaCompatibleSharding>(
          std::move(devices), memory_kind, sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef HloSharding::sharding_spec() const { return sharding_spec_; }

absl::StatusOr<Shape> HloSharding::GetShardShape(const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool HloSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_hlo_sharding = dyn_cast<HloSharding>(&other);
  if (!other_hlo_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_hlo_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>> HloSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(
      new HloSharding(devices.value_or(devices_),
                      memory_kind.value_or(memory_kind_), sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  bool is_even_sharding = false;
  if (xla_hlo_sharding().IsReplicatedOrSingleDevice() ||
      xla_hlo_sharding().IsUnreduced()) {
    is_even_sharding = true;
  } else if (xla_hlo_sharding().IsTiled()) {
    const int64_t tiled_data_rank = xla_hlo_sharding().TiledDataRank();
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%v, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape,
          xla_hlo_sharding().ToString()));
    }

    is_even_sharding = true;
    for (int i = 0; i < tiled_data_rank; ++i) {
      if (shape.dims()[i] % xla_hlo_sharding().dimension(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (xla_hlo_sharding().IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    is_even_sharding = true;
  }

  return is_even_sharding
             ? DisassembleEven(shape, single_device_shard_semantics)
             : DisassembleUneven(shape, single_device_shard_semantics);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::DisassembleEven(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  // Fast path for even sharding.
  ABSL_ASSIGN_OR_RETURN(xla::ifrt::Shape shard_shape, GetShardShape(shape));
  std::vector<std::pair<Shape, ShardingRef>> result;
  DeviceList* device_list;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    device_list = devices_.get();
  } else {
    device_list = devices_->AddressableDeviceList();
  }
  result.reserve(device_list->size());
  for (Device* device : device_list->devices()) {
    result.push_back({
        shard_shape,
        SingleDeviceSharding::Create(device, memory_kind_),
    });
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
HloSharding::DisassembleUneven(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  // Slow path that uses `IndexDomains()` to handle uneven sharding.
  ABSL_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                   IndexDomains(shape, SingleDeviceShardSemantics::kAllShards));
  CHECK_EQ(index_domains.size(), devices_->size());
  std::vector<std::pair<Shape, ShardingRef>> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(devices_->size());
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  const absl::Span<Device* const> devices = devices_->devices();
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

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
HloSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(absl::StrFormat(
      "HloSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::StatusOr<std::vector<IndexDomain>> HloSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  ABSL_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                   sharding_spec_->IndexDomains(shape));
  DCHECK_EQ(index_domains.size(), devices_->size());
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    return index_domains;
  }
  std::vector<IndexDomain> result;
  result.reserve(devices_->AddressableDeviceList()->size());
  const absl::Span<Device* const> devices = devices_->devices();
  for (int i = 0; i < index_domains.size(); ++i) {
    if (devices[i]->IsAddressable()) {
      result.push_back(std::move(index_domains[i]));
    }
  }
  return result;
}

std::string HloSharding::DebugString() const {
  return absl::StrFormat("HloSharding(memory_kind: %v, hlo_sharding: %s)",
                         memory_kind_, xla_hlo_sharding().ToString());
}

void HloSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

std::vector<IndexDomain> TEST_HloShardingIndexDomainsSlowPath(
    const HloSharding& hlo_sharding, const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) {
  std::vector<IndexDomain> index_domains =
      TEST_HloShardingSpecIndexDomainsSlowPath(
          *cast<const HloShardingSpec>(hlo_sharding.sharding_spec()), shape);
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    return index_domains;
  }
  std::vector<IndexDomain> result;
  result.reserve(hlo_sharding.devices()->AddressableDeviceList()->size());
  const absl::Span<Device* const> devices = hlo_sharding.devices()->devices();
  for (int i = 0; i < index_domains.size(); ++i) {
    if (devices[i]->IsAddressable()) {
      result.push_back(std::move(index_domains[i]));
    }
  }
  return result;
}

}  // namespace ifrt
}  // namespace xla
