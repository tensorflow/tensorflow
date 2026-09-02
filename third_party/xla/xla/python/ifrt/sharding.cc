/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/sharding.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.pb.h"
#include "xla/python/ifrt/sharding_spec.h"

namespace xla {
namespace ifrt {

char Sharding::ID = 0;
char SingleDeviceSharding::ID = 0;
char OpaqueSharding::ID = 0;
char ConcreteSharding::ID = 0;
char ConcreteEvenSharding::ID = 0;
char ShardingParamSharding::ID = 0;

char DeserializeShardingOptions::ID = 0;

Sharding::Sharding(DeviceListRef devices, MemoryKind memory_kind,
                   bool is_fully_replicated)
    : devices_(std::move(devices)),
      memory_kind_(memory_kind),
      is_fully_replicated_(is_fully_replicated) {}

bool Sharding::operator==(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  return HasSamePartitioning(other) && memory_kind_ == other.memory_kind_ &&
         *devices() == *other.devices();
}

absl::StatusOr<ShardingRef> Sharding::FromProto(
    Client* client, const ShardingProto& sharding_proto) {
  return Deserialize<Sharding>(
      sharding_proto.serialized_sharding(),
      std::make_unique<DeserializeShardingOptions>(client));
}

absl::Status Sharding::ToProto(ShardingProto& sharding_proto,
                               SerDesVersion version) const {
  // `ShardingProto` does not store its own version. It delegates the details to
  // SerDes of the `Sharding` subclasses.
  auto options = std::make_unique<SerializeOptions>(version);
  return Serialize(*this, std::move(options),
                   *sharding_proto.mutable_serialized_sharding());
}

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << absl::StrCat(sharding);
}

std::unique_ptr<SingleDeviceSharding> SingleDeviceSharding::Create(
    Device* device, MemoryKind memory_kind) {
  CHECK(device != nullptr);
  absl::StatusOr<DeviceListRef> device_list =
      device->client()->MakeDeviceList({device});
  CHECK_OK(device_list);
  return std::unique_ptr<SingleDeviceSharding>(
      new SingleDeviceSharding(*std::move(device_list), memory_kind,
                               SingleDeviceShardingSpec::Create()));
}

SingleDeviceSharding::SingleDeviceSharding(
    DeviceListRef device_list, MemoryKind memory_kind,
    std::shared_ptr<const SingleDeviceShardingSpec> sharding_spec)
    : RTTIExtends<SingleDeviceSharding, Sharding>(
          std::move(device_list), memory_kind,
          sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef SingleDeviceSharding::sharding_spec() const {
  return sharding_spec_;
}

absl::StatusOr<Shape> SingleDeviceSharding::GetShardShape(
    const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool SingleDeviceSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_single_device_sharding =
      dyn_cast<SingleDeviceSharding>(&other);
  if (!other_single_device_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_single_device_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>>
SingleDeviceSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SingleDeviceSharding can only have one device, but was asked to have "
        "%d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(new SingleDeviceSharding(
      devices.value_or(devices_), memory_kind.value_or(memory_kind_),
      sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
SingleDeviceSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  std::vector<std::pair<Shape, ShardingRef>> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards ||
      devices_->devices().front()->IsAddressable()) {
    result.reserve(1);
    result.push_back({shape, SingleDeviceSharding::Create(
                                 devices_->devices().front(), memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
SingleDeviceSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  std::vector<std::pair<DynamicShape, ShardingRef>> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards ||
      devices_->devices().front()->IsAddressable()) {
    result.reserve(1);
    result.push_back(
        {dynamic_shape, SingleDeviceSharding::Create(
                            devices_->devices().front(), memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<IndexDomain>> SingleDeviceSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  std::vector<IndexDomain> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards ||
      devices_->devices().front()->IsAddressable()) {
    result.reserve(1);
    result.push_back(IndexDomain(shape));
  }
  return result;
}

std::string SingleDeviceSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("SingleDeviceSharding(%s, memory_kind: %v)",
                         devices_->devices().front()->DebugString(),
                         memory_kind_);
}

void SingleDeviceSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

std::unique_ptr<OpaqueSharding> OpaqueSharding::Create(DeviceListRef devices,
                                                       MemoryKind memory_kind) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  int num_shards = devices->size();
  return std::unique_ptr<OpaqueSharding>(new OpaqueSharding(
      std::move(devices), memory_kind, OpaqueShardingSpec::Create(num_shards)));
}

OpaqueSharding::OpaqueSharding(
    DeviceListRef devices, MemoryKind memory_kind,
    std::shared_ptr<const OpaqueShardingSpec> sharding_spec)
    : RTTIExtends<OpaqueSharding, Sharding>(std::move(devices), memory_kind,
                                            sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef OpaqueSharding::sharding_spec() const { return sharding_spec_; }

absl::StatusOr<Shape> OpaqueSharding::GetShardShape(const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool OpaqueSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_opaque_sharding = dyn_cast<OpaqueSharding>(&other);
  if (!other_opaque_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_opaque_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>> OpaqueSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OpaqueSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(
      new OpaqueSharding(devices.value_or(devices_),
                         memory_kind.value_or(memory_kind_), sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
OpaqueSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
OpaqueSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(
      "OpaqueSharding does not have index domain information");
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("OpaqueSharding(devices: %v, memory_kind: %v)",
                         *devices_, memory_kind_);
}

void OpaqueSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, Shape shape,
    std::vector<Shape> shard_shapes,
    std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  return std::unique_ptr<ConcreteSharding>(new ConcreteSharding(
      std::move(devices), memory_kind,
      ConcreteShardingSpec::Create(std::move(shape), std::move(shard_shapes),
                                   std::move(index_domains))));
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  return std::unique_ptr<ConcreteSharding>(new ConcreteSharding(
      std::move(devices), memory_kind,
      ConcreteShardingSpec::Create(std::move(dynamic_shape),
                                   std::move(shard_dynamic_shapes))));
}

ConcreteSharding::ConcreteSharding(
    DeviceListRef devices, MemoryKind memory_kind,
    std::shared_ptr<const ConcreteShardingSpec> sharding_spec)
    : RTTIExtends<ConcreteSharding, Sharding>(
          std::move(devices), memory_kind, sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef ConcreteSharding::sharding_spec() const {
  return sharding_spec_;
}

absl::StatusOr<Shape> ConcreteSharding::GetShardShape(
    const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool ConcreteSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_sharding = dyn_cast<ConcreteSharding>(&other);
  if (!other_concrete_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_concrete_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>>
ConcreteSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(
      new ConcreteSharding(devices.value_or(devices_),
                           memory_kind.value_or(memory_kind_), sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!has_static_shape()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConcreteSharding holds dynamic shape, but was asked "
                        "to disassemble static shape %v",
                        shape));
  }
  if (shape != this->shape()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding can only disassemble shape %v, but was asked "
        "to disassemble shape %v",
        this->shape(), shape));
  }
  std::vector<std::pair<Shape, ShardingRef>> result;
  const std::vector<Shape>& shard_shapes = this->shard_shapes();

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != shard_shapes.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains non-addressable devices. Saw %d devices, "
        "with %d addressable devices.",
        devices_->size(), shard_shapes.size()));
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (shard_shapes.size() != addressable_devices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding must have the same number of "
        "shard shapes and addressable devices. Saw %d shard shapes, with %d "
        "addressable devices.",
        shard_shapes.size(), addressable_devices.size()));
  }

  result.reserve(addressable_devices.size());
  for (int i = 0; i < addressable_devices.size(); ++i) {
    result.push_back(
        {shard_shapes[i],
         SingleDeviceSharding::Create(addressable_devices[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ConcreteSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!has_dynamic_shape()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ConcreteSharding holds static shape, but was asked "
                        "to disassemble dynamic shape %v",
                        dynamic_shape));
  }
  if (dynamic_shape != this->dynamic_shape()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding can only disassemble dynamic shape %v, but was asked "
        "to disassemble dynamic shape %v",
        this->dynamic_shape(), dynamic_shape));
  }
  std::vector<std::pair<DynamicShape, ShardingRef>> result;
  const std::vector<DynamicShape>& shard_dynamic_shapes =
      this->shard_dynamic_shapes();

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != shard_dynamic_shapes.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains non-addressable devices. Saw %d devices, "
        "with %d addressable devices.",
        devices_->size(), shard_dynamic_shapes.size()));
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (shard_dynamic_shapes.size() != addressable_devices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding must have the same number of "
        "shard shapes and addressable devices. Saw %d shard shapes, with %d "
        "addressable devices.",
        shard_dynamic_shapes.size(), addressable_devices.size()));
  }

  result.reserve(addressable_devices.size());
  for (int i = 0; i < addressable_devices.size(); ++i) {
    result.push_back(
        {shard_dynamic_shapes[i],
         SingleDeviceSharding::Create(addressable_devices[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!index_domains().has_value()) {
    return absl::InvalidArgumentError(
        "ConcreteSharding does not have index domain information");
  }

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != index_domains()->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains index domains from non-addressable devices. "
        "Saw %d devices, with %d addressable devices.",
        devices_->size(), index_domains()->size()));
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (index_domains()->size() != addressable_devices.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteSharding must have the same number of "
        "index domains and addressable devices. Saw %d index domains, with %d "
        "addressable devices.",
        index_domains()->size(), addressable_devices.size()));
  }

  return *index_domains();
}

std::string ConcreteSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ConcreteSharding(devices: %v, shape: %v, shard_shapes: [%s], "
      "index_domains: %s, memory_kind: %v)",
      *devices_,
      has_static_shape() ? absl::StrCat(shape())
                         : absl::StrCat(dynamic_shape()),
      has_static_shape() ? absl::StrJoin(shard_shapes(), ",")
                         : absl::StrJoin(shard_dynamic_shapes(), ","),
      index_domains().has_value()
          ? absl::StrCat("[", absl::StrJoin(*index_domains(), ","), "]")
          : "<nullopt>",
      memory_kind_);
}

void ConcreteSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

std::unique_ptr<ConcreteEvenSharding> ConcreteEvenSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, Shape shape,
    Shape shard_shape, bool is_fully_replicated) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  int num_shards = devices->size();
  return std::unique_ptr<ConcreteEvenSharding>(new ConcreteEvenSharding(
      std::move(devices), memory_kind,
      ConcreteEvenShardingSpec::Create(num_shards, std::move(shape),
                                       std::move(shard_shape),
                                       is_fully_replicated)));
}

ConcreteEvenSharding::ConcreteEvenSharding(
    DeviceListRef devices, MemoryKind memory_kind,
    std::shared_ptr<const ConcreteEvenShardingSpec> sharding_spec)
    : RTTIExtends<ConcreteEvenSharding, Sharding>(
          std::move(devices), memory_kind, sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef ConcreteEvenSharding::sharding_spec() const {
  return sharding_spec_;
}

absl::StatusOr<Shape> ConcreteEvenSharding::GetShardShape(
    const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool ConcreteEvenSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_even_sharding =
      dyn_cast<ConcreteEvenSharding>(&other);
  if (!other_concrete_even_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_concrete_even_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>>
ConcreteEvenSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteEvenSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(new ConcreteEvenSharding(
      devices.value_or(devices_), memory_kind.value_or(memory_kind_),
      sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (shape != this->shape()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteEvenSharding can only disassemble shape %v, but was asked "
        "to disassemble shape %v",
        this->shape(), shape));
  }
  std::vector<std::pair<Shape, ShardingRef>> result;
  const absl::Span<Device* const> devices = devices_->devices();
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(devices_->size());
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  for (int i = 0; i < devices.size(); ++i) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        devices[i]->IsAddressable()) {
      result.push_back({shard_shape(), SingleDeviceSharding::Create(
                                           devices[i], memory_kind_)});
    }
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(absl::StrFormat(
      "ConcreteEvenSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (IsFullyReplicated() && this->shape() == shard_shape() &&
      this->shape() == shape) {
    std::vector<IndexDomain> result;
    if (single_device_shard_semantics ==
        SingleDeviceShardSemantics::kAllShards) {
      result.resize(devices_->size(), IndexDomain(shape));
    } else {
      result.resize(devices_->AddressableDeviceList()->size(),
                    IndexDomain(shape));
    }
    return result;
  }
  return absl::InvalidArgumentError(
      "ConcreteEvenSharding does not have index domain information");
}

std::string ConcreteEvenSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ConcreteEvenSharding(devices: %v, shape: %v, shard_shape: %v, "
      "memory_kind: %v, is_fully_replicated: %s)",
      *devices_, shape(), shard_shape(), memory_kind_,
      IsFullyReplicated() ? "true" : "false");
}

void ConcreteEvenSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

absl::StatusOr<std::unique_ptr<ShardingParamSharding>>
ShardingParamSharding::Create(ShardingParam sharding_param,
                              DeviceListRef devices, MemoryKind memory_kind) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  int64_t device_count =
      absl::c_accumulate(sharding_param.minor_to_major().axis_sizes, 1,
                         std::multiplies<int64_t>());
  if (device_count != devices->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Device counts don't match. From ShardingParam %d vs from DeviceList "
        "%d",
        device_count, devices->size()));
  }
  return std::unique_ptr<ShardingParamSharding>(new ShardingParamSharding(
      std::move(devices), memory_kind,
      ShardingParamShardingSpec::Create(std::move(sharding_param))));
}

ShardingParamSharding::ShardingParamSharding(
    DeviceListRef devices, MemoryKind memory_kind,
    std::shared_ptr<const ShardingParamShardingSpec> sharding_spec)
    : RTTIExtends<ShardingParamSharding, Sharding>(
          std::move(devices), memory_kind, sharding_spec->IsFullyReplicated()),
      sharding_spec_(std::move(sharding_spec)) {}

ShardingSpecRef ShardingParamSharding::sharding_spec() const {
  return sharding_spec_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ShardingParamSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  ABSL_ASSIGN_OR_RETURN(Shape local_shape, GetShardShape(shape));

  std::vector<std::pair<Shape, ShardingRef>> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(devices_->size());
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  for (Device* device : devices_->devices()) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        device->IsAddressable()) {
      result.push_back(
          {local_shape, SingleDeviceSharding::Create(device, memory_kind_)});
    }
  }

  return result;
}

absl::StatusOr<Shape> ShardingParamSharding::GetShardShape(
    const Shape& shape) const {
  return sharding_spec_->GetShardShape(shape);
}

bool ShardingParamSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_sharding_param_sharding =
      dyn_cast<ShardingParamSharding>(&other);
  if (!other_sharding_param_sharding) {
    return false;
  }
  return sharding_spec_->HasSamePartitioning(
      *other_sharding_param_sharding->sharding_spec());
}

absl::StatusOr<std::unique_ptr<Sharding>>
ShardingParamSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ShardingParamSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size()));
  }
  return std::unique_ptr<Sharding>(new ShardingParamSharding(
      devices.value_or(devices_), memory_kind.value_or(memory_kind_),
      sharding_spec_));
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ShardingParamSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return absl::InvalidArgumentError(absl::StrFormat(
      "ShardingParamSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
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

std::string ShardingParamSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ShardingParamSharding(%s, devices: %v, memory_kind: %v)",
      sharding_param().DebugString(), *devices_, memory_kind_);
}

void ShardingParamSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           *sharding_spec_);
}

}  // namespace ifrt
}  // namespace xla
