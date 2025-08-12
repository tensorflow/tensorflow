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
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

namespace {

// Returns a canonicalized memory kind for the given devices.
// REQUIRES: !devices->devices().empty()
MemoryKind CanonicalizeMemoryKindWithDevices(const MemoryKind& memory_kind,
                                             const DeviceListRef& devices) {
  CHECK(devices != nullptr);
  CHECK(!devices->devices().empty());
  return CanonicalizeMemoryKind(memory_kind, devices->devices().front());
}

// Returns if `sharding_param` indicates a fully replicated sharding.
bool ComputeIsFullyReplicated(const ShardingParam& sharding_param) {
  return llvm::all_of(sharding_param.dim_shards(),
                      [](auto shards) { return shards == 1; });
}

// Iterates the major-to-minor Cartesian product of a Span of containers of the
// same type.
//
// For example, for {1, 2, 3} x {4, 5}, it iterates in the order of
//   {1, 4}, {1, 5}, {2, 4}, {2, 5}, {3, 4}, {3, 5}
// The values are copied into the result vectors.
template <typename ContainerT>
class MajorToMinorIter {
 public:
  using IteratorT = typename ContainerT::const_iterator;
  using ValueT = typename ContainerT::value_type;

  // Returns the iterator at the begin of the Cartesian product.
  static MajorToMinorIter<ContainerT> cbegin(
      absl::Span<const ContainerT> containers) {
    std::vector<IteratorT> iters;
    iters.reserve(containers.size());
    for (const ContainerT& container : containers) {
      iters.push_back(container.cbegin());
    }
    return MajorToMinorIter(containers, std::move(iters));
  }

  // Returns the vector of values at the iteration point.
  std::vector<ValueT> operator*() const {
    std::vector<ValueT> result;
    result.reserve(iters_.size());
    for (const auto& iter : iters_) {
      result.push_back(*iter);
    }
    return result;
  }

  // Moves to the next.
  void operator++() {
    for (int i = iters_.size() - 1; i >= 0; --i) {
      ++iters_[i];
      if (iters_[i] != containers_[i].end()) {
        break;
      }
      if (i != 0) {
        // Carry over.
        iters_[i] = containers_[i].begin();
      }
    }
  }

  // Returns whether the iterator has reached the end.
  // Note: Due to the implementation of ++, not all iters_ is end().
  bool IsEnd() const {
    return iters_.empty() || iters_[0] == containers_[0].end();
  }

 private:
  MajorToMinorIter(absl::Span<const ContainerT> containers,
                   std::vector<IteratorT> iters)
      : containers_(containers), iters_(iters) {
    DCHECK_EQ(iters.size(), containers.size());
  }

  absl::Span<const ContainerT> containers_;
  std::vector<IteratorT> iters_;
};

// Returns the indices of the tiles.
//
// For example, when `dim_shards` is {2, 3}, the result is
//   {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}
std::vector<Index> GetTileIndices(absl::Span<const int64_t> dim_shards) {
  std::vector<std::vector<int64_t>> indices;
  indices.reserve(dim_shards.size());
  for (const int64_t dim_shard : dim_shards) {
    std::vector<int64_t> index(dim_shard);
    absl::c_iota(index, 0);
    indices.push_back(std::move(index));
  }

  std::vector<Index> result;
  int64_t shard_count =
      absl::c_accumulate(dim_shards, 1, std::multiplies<int64_t>());
  result.reserve(shard_count);
  for (auto iter = MajorToMinorIter<std::vector<int64_t>>::cbegin(indices);
       !iter.IsEnd(); ++iter) {
    result.push_back(Index(*iter));
  }
  return result;
}

}  // namespace

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

absl::StatusOr<ShardingProto> Sharding::ToProto(SerDesVersion version) const {
  ShardingProto sharding_proto;
  // `ShardingProto` does not store its own version. It delegates the details to
  // SerDes of the `Sharding` subclasses.
  auto options = std::make_unique<SerializeOptions>(version);
  TF_ASSIGN_OR_RETURN(*sharding_proto.mutable_serialized_sharding(),
                      Serialize(*this, std::move(options)));
  return sharding_proto;
}

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << sharding.DebugString();
}

std::unique_ptr<SingleDeviceSharding> SingleDeviceSharding::Create(
    Device* device, MemoryKind memory_kind) {
  CHECK(device != nullptr);
  memory_kind = CanonicalizeMemoryKind(memory_kind, device);
  return std::unique_ptr<SingleDeviceSharding>(
      new SingleDeviceSharding(device, memory_kind));
}

SingleDeviceSharding::SingleDeviceSharding(Device* device,
                                           MemoryKind memory_kind)
    : llvm::RTTIExtends<SingleDeviceSharding, Sharding>(
          device->client()->MakeDeviceList({device}), memory_kind,
          /*is_fully_replicated=*/true) {}

absl::StatusOr<Shape> SingleDeviceSharding::GetShardShape(
    const Shape& shape) const {
  return shape;
}

bool SingleDeviceSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  return llvm::isa<SingleDeviceSharding>(&other);
}

absl::StatusOr<std::unique_ptr<Sharding>>
SingleDeviceSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != 1) {
    return InvalidArgument(
        "SingleDeviceSharding can only have one device, but was asked to have "
        "%d devices",
        (*devices)->size());
  }
  return Create(devices.value_or(devices_)->devices().front(),
                memory_kind.value_or(memory_kind_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
SingleDeviceSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
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
SingleDeviceSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
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
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
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
  absl::HashState::combine(std::move(state), devices_, memory_kind_);
}

std::unique_ptr<OpaqueSharding> OpaqueSharding::Create(DeviceListRef devices,
                                                       MemoryKind memory_kind) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<OpaqueSharding>(
      new OpaqueSharding(std::move(devices), memory_kind));
}

OpaqueSharding::OpaqueSharding(DeviceListRef devices, MemoryKind memory_kind)
    : llvm::RTTIExtends<OpaqueSharding, Sharding>(
          std::move(devices), memory_kind, /*is_fully_replicated=*/false) {}

absl::StatusOr<Shape> OpaqueSharding::GetShardShape(const Shape& shape) const {
  return InvalidArgument(
      "OpaqueSharding does not have shard shape information");
}

bool OpaqueSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  // If the objects are not the same, we cannot tell whether the two
  // OpaqueShardings are using the same logical partitioning.
  return false;
}

absl::StatusOr<std::unique_ptr<Sharding>> OpaqueSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return InvalidArgument(
        "OpaqueSharding should have the same number of devices as the current "
        "sharding, but was asked to have %d devices",
        (*devices)->size());
  }
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
OpaqueSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
OpaqueSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
OpaqueSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
OpaqueSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have index domain information");
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("OpaqueSharding(devices: %v, memory_kind: %v)",
                         *devices_, memory_kind_);
}

void OpaqueSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_);
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, Shape shape,
    std::vector<Shape> shard_shapes,
    std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<ConcreteSharding>(
      new ConcreteSharding(std::move(devices), memory_kind, std::move(shape),
                           std::move(shard_shapes), std::move(index_domains)));
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<ConcreteSharding>(new ConcreteSharding(
      std::move(devices), memory_kind, std::move(dynamic_shape),
      std::move(shard_dynamic_shapes)));
}

ConcreteSharding::ConcreteSharding(
    DeviceListRef devices, MemoryKind memory_kind, Shape shape,
    std::vector<Shape> shard_shapes,
    std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains)
    : llvm::RTTIExtends<ConcreteSharding, Sharding>(
          std::move(devices), memory_kind, /*is_fully_replicated=*/false),
      shape_(std::move(shape)),
      shard_shapes_(std::move(shard_shapes)),
      index_domains_(std::move(index_domains)) {
  // If all per-shard shapes are the same, cache this shape for
  // `GetShardShape()`. Ideally, users should have used `ConcreteEvenSharding`
  // for such a case, but there are existing use cases that instantiate
  // `ConcreteSharding` from a list of per-shard shapes without checking for
  // identical per-shard shapes.
  const auto& static_shard_shapes = std::get<std::vector<Shape>>(shard_shapes_);
  bool identical = true;
  for (int i = 1; i < static_shard_shapes.size(); ++i) {
    if (static_shard_shapes[i] != static_shard_shapes[0]) {
      identical = false;
      break;
    }
  }
  if (identical && !static_shard_shapes.empty()) {
    shard_shape_ = static_shard_shapes[0];
  }
}

ConcreteSharding::ConcreteSharding(
    DeviceListRef devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes)
    : llvm::RTTIExtends<ConcreteSharding, Sharding>(
          std::move(devices), memory_kind, /*is_fully_replicated=*/false),
      shape_(std::move(dynamic_shape)),
      shard_shapes_(std::move(shard_dynamic_shapes)) {}

absl::StatusOr<Shape> ConcreteSharding::GetShardShape(
    const Shape& shape) const {
  if (shard_shape_.has_value()) {
    return *shard_shape_;
  }
  return InvalidArgument("ConcreteSharding does not have a fixed shard shape");
}

bool ConcreteSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_sharding =
      llvm::dyn_cast<ConcreteSharding>(&other);
  if (!other_concrete_sharding) {
    return false;
  }
  return shape_ == other_concrete_sharding->shape_ &&
         shard_shapes_ == other_concrete_sharding->shard_shapes_;
}

absl::StatusOr<std::unique_ptr<Sharding>>
ConcreteSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return InvalidArgument(
        "ConcreteSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size());
  }
  if (has_static_shape()) {
    return Create(devices.value_or(devices_),
                  memory_kind.value_or(memory_kind_), std::get<Shape>(shape_),
                  std::get<std::vector<Shape>>(shard_shapes_));
  }
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_),
                std::get<DynamicShape>(shape_),
                std::get<std::vector<DynamicShape>>(shard_shapes_));
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!has_static_shape()) {
    return InvalidArgument(
        "ConcreteSharding holds dynamic shape, but was asked "
        "to disassemble static shape %s",
        shape.DebugString());
  }
  if (shape != std::get<Shape>(shape_)) {
    return InvalidArgument(
        "ConcreteSharding can only disassemble shape %s, but was asked "
        "to disassemble shape %s",
        std::get<Shape>(shape_).DebugString(), shape.DebugString());
  }
  std::vector<std::pair<Shape, ShardingRef>> result;
  const std::vector<Shape>& shard_shapes =
      std::get<std::vector<Shape>>(shard_shapes_);

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != shard_shapes.size()) {
    return InvalidArgument(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains non-addressable devices. Saw %d devices, "
        "with %d addressable devices.",
        devices_->size(), shard_shapes.size());
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (shard_shapes.size() != addressable_devices.size()) {
    return InvalidArgument(
        "ConcreteSharding must have the same number of "
        "shard shapes and addressable devices. Saw %d shard shapes, with %d "
        "addressable devices.",
        shard_shapes.size(), addressable_devices.size());
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
ConcreteSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ConcreteSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!has_dynamic_shape()) {
    return InvalidArgument(
        "ConcreteSharding holds static shape, but was asked "
        "to disassemble dynamic shape %s",
        dynamic_shape.DebugString());
  }
  if (dynamic_shape != std::get<DynamicShape>(shape_)) {
    return InvalidArgument(
        "ConcreteSharding can only disassemble dynamic shape %s, but was asked "
        "to disassemble dynamic shape %s",
        std::get<DynamicShape>(shape_).DebugString(),
        dynamic_shape.DebugString());
  }
  std::vector<std::pair<DynamicShape, ShardingRef>> result;
  const std::vector<DynamicShape>& shard_dynamic_shapes =
      std::get<std::vector<DynamicShape>>(shard_shapes_);

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != shard_dynamic_shapes.size()) {
    return InvalidArgument(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains non-addressable devices. Saw %d devices, "
        "with %d addressable devices.",
        devices_->size(), shard_dynamic_shapes.size());
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (shard_dynamic_shapes.size() != addressable_devices.size()) {
    return InvalidArgument(
        "ConcreteSharding must have the same number of "
        "shard shapes and addressable devices. Saw %d shard shapes, with %d "
        "addressable devices.",
        shard_dynamic_shapes.size(), addressable_devices.size());
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
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (!index_domains_.has_value()) {
    return InvalidArgument(
        "ConcreteSharding does not have index domain information");
  }

  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards &&
      devices_->size() != index_domains_->size()) {
    return InvalidArgument(
        "SingleDeviceShardSemantics::kAllShards was requested, but the "
        "ConcreteSharding contains index domains from non-addressable devices. "
        "Saw %d devices, with %d addressable devices.",
        devices_->size(), index_domains_->size());
  }

  const absl::Span<Device* const> addressable_devices =
      devices_->AddressableDeviceList()->devices();
  if (index_domains_->size() != addressable_devices.size()) {
    return InvalidArgument(
        "ConcreteSharding must have the same number of "
        "index domains and addressable devices. Saw %d index domains, with %d "
        "addressable devices.",
        index_domains_->size(), addressable_devices.size());
  }

  return *index_domains_;
}

std::string ConcreteSharding::DebugString() const {
  DCHECK(this);
  return std::visit(
      [this](const auto& shape, const auto& shard_shapes) {
        return absl::StrFormat(
            "ConcreteSharding(devices: %v, shape: %s, shard_shapes: [%s], "
            "index_domains: %s, memory_kind: %v)",
            *devices_, shape.DebugString(),
            absl::StrJoin(shard_shapes, ",",
                          [](std::string* out, const auto& shard_shape) {
                            absl::StrAppend(out, shard_shape.DebugString());
                          }),
            index_domains_.has_value()
                ? absl::StrCat("[", absl::StrJoin(*index_domains_, ","), "]")
                : "<nullopt>",
            memory_kind_);
      },
      shape_, shard_shapes_);
}

void ConcreteSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_, shape_,
                           shard_shapes_);
}

std::unique_ptr<ConcreteEvenSharding> ConcreteEvenSharding::Create(
    DeviceListRef devices, MemoryKind memory_kind, Shape shape,
    Shape shard_shape, bool is_fully_replicated) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  return std::unique_ptr<ConcreteEvenSharding>(new ConcreteEvenSharding(
      std::move(devices), memory_kind, std::move(shape), std::move(shard_shape),
      is_fully_replicated));
}

ConcreteEvenSharding::ConcreteEvenSharding(DeviceListRef devices,
                                           MemoryKind memory_kind, Shape shape,
                                           Shape shard_shape,
                                           bool is_fully_replicated)
    : llvm::RTTIExtends<ConcreteEvenSharding, Sharding>(
          std::move(devices), memory_kind, is_fully_replicated),
      shape_(std::move(shape)),
      shard_shape_(std::move(shard_shape)) {}

absl::StatusOr<Shape> ConcreteEvenSharding::GetShardShape(
    const Shape& shape) const {
  if (shape != shape_) {
    return InvalidArgument(
        "ConcreteEvenSharding has a shard shape for shape %s, but was asked "
        "to get a shard shape for shape %s",
        shape_.DebugString(), shape.DebugString());
  }
  return shard_shape_;
}

bool ConcreteEvenSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_even_sharding =
      llvm::dyn_cast<ConcreteEvenSharding>(&other);
  if (!other_concrete_even_sharding) {
    return false;
  }
  return devices_->size() == other_concrete_even_sharding->devices_->size() &&
         shape_ == other_concrete_even_sharding->shape_ &&
         shard_shape_ == other_concrete_even_sharding->shard_shape_ &&
         is_fully_replicated_ ==
             other_concrete_even_sharding->is_fully_replicated_;
}

absl::StatusOr<std::unique_ptr<Sharding>>
ConcreteEvenSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return InvalidArgument(
        "ConcreteEvenSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size());
  }
  return Create(devices.value_or(devices_), memory_kind.value_or(memory_kind_),
                shape_, shard_shape_, is_fully_replicated_);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  if (shape != shape_) {
    return InvalidArgument(
        "ConcreteEvenSharding can only disassemble shape %s, but was asked "
        "to disassemble shape %s",
        shape_.DebugString(), shape.DebugString());
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
      result.push_back({shard_shape_, SingleDeviceSharding::Create(
                                          devices[i], memory_kind_)});
    }
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ConcreteEvenSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "ConcreteEvenSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}
absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "ConcreteEvenSharding does not have index domain information");
}

std::string ConcreteEvenSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ConcreteEvenSharding(devices: %v, shape: %s, shard_shape: %s, "
      "memory_kind: %v, is_fully_replicated: %s)",
      *devices_, shape_.DebugString(), shard_shape_.DebugString(), memory_kind_,
      is_fully_replicated_ ? "true" : "false");
}

void ConcreteEvenSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           is_fully_replicated_, shape_, shard_shape_);
}

absl::StatusOr<std::unique_ptr<ShardingParamSharding>>
ShardingParamSharding::Create(ShardingParam sharding_param,
                              DeviceListRef devices, MemoryKind memory_kind) {
  memory_kind = CanonicalizeMemoryKindWithDevices(memory_kind, devices);
  int64_t device_count =
      absl::c_accumulate(sharding_param.minor_to_major().axis_sizes, 1,
                         std::multiplies<int64_t>());
  if (device_count != devices->size()) {
    return InvalidArgument(
        "Device counts don't match. From ShardingParam %d vs from DeviceList "
        "%d",
        device_count, devices->size());
  }
  return std::unique_ptr<ShardingParamSharding>(new ShardingParamSharding(
      std::move(sharding_param), std::move(devices), memory_kind));
}

ShardingParamSharding::ShardingParamSharding(ShardingParam sharding_param,
                                             DeviceListRef devices,
                                             MemoryKind memory_kind)
    : llvm::RTTIExtends<ShardingParamSharding, Sharding>(
          std::move(devices), memory_kind,
          ComputeIsFullyReplicated(sharding_param)),
      sharding_param_(sharding_param) {}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ShardingParamSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return Disassemble(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingRef>>>
ShardingParamSharding::Disassemble(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(Shape local_shape, GetShardShape(shape));

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
  if (shape.dims().size() != sharding_param_.dim_shards().size()) {
    return InvalidArgument(
        "Numbers of dimensions don't match. From Shape %d vs from "
        "ShardingParam %d",
        shape.dims().size(), sharding_param_.dim_shards().size());
  }
  std::vector<int64_t> dims;
  dims.reserve(shape.dims().size());
  for (const auto [dim, dim_shards] :
       llvm::zip(shape.dims(), sharding_param_.dim_shards())) {
    if (dim % dim_shards != 0) {
      return InvalidArgument(
          "Uneven shard is not supported. dim: %d, dim_shards: %d", dim,
          dim_shards);
    }
    dims.push_back(dim / dim_shards);
  }
  return Shape(dims);
}

bool ShardingParamSharding::HasSamePartitioning(const Sharding& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_sharding_param_sharding =
      llvm::dyn_cast<ShardingParamSharding>(&other);
  if (!other_sharding_param_sharding) {
    return false;
  }
  return sharding_param_ == other_sharding_param_sharding->sharding_param_;
}

absl::StatusOr<std::unique_ptr<Sharding>>
ShardingParamSharding::WithDeviceAssignment(
    std::optional<DeviceListRef> devices,
    std::optional<MemoryKind> memory_kind) const {
  if (devices.has_value() && (*devices)->size() != devices_->size()) {
    return InvalidArgument(
        "ShardingParamSharding should have the same number of devices as the "
        "current sharding, but was asked to have %d devices",
        (*devices)->size());
  }
  return Create(sharding_param_, devices.value_or(devices_),
                memory_kind.value_or(memory_kind_));
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ShardingParamSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return Disassemble(dynamic_shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingRef>>>
ShardingParamSharding::Disassemble(
    const DynamicShape& dynamic_shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);
  return InvalidArgument(
      "ShardingParamSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return IndexDomains(shape, SingleDeviceShardSemantics::kAllShards);
}

absl::StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
    const Shape& shape,
    SingleDeviceShardSemantics single_device_shard_semantics) const {
  DCHECK(this);

  // Calculate the origins of tiles, ignoring device assignments.
  TF_ASSIGN_OR_RETURN(Shape local_shape, GetShardShape(shape));
  std::vector<Index> tile_indices =
      GetTileIndices(sharding_param_.dim_shards());
  std::vector<Index> origins;
  origins.reserve(tile_indices.size());
  for (const Index& tile_index : tile_indices) {
    origins.push_back(tile_index * local_shape.dims());
  }

  // Calculate the device assignments.
  // `origins[i]` should go to `device_list[i]`.
  static constexpr int kInvalidIndex = -1;
  llvm::SmallVector<int, 4> device_list;
  sharding_param_.minor_to_major().ToDeviceList(device_list);
  std::vector<int> device_to_index(device_list.size(), kInvalidIndex);
  for (int i = 0; i < device_list.size(); ++i) {
    device_to_index[device_list[i]] = i;
  }

  // Replication is the minor axis in `device_list`.
  DCHECK_EQ(device_to_index.size() % origins.size(), 0);
  int replication = device_to_index.size() / origins.size();

  DCHECK_EQ(device_to_index.size(), devices_->size());
  std::vector<IndexDomain> result;
  if (single_device_shard_semantics == SingleDeviceShardSemantics::kAllShards) {
    result.reserve(devices_->size());
  } else {
    result.reserve(devices_->AddressableDeviceList()->size());
  }
  const absl::Span<Device* const> devices = devices_->devices();
  for (int i = 0; i < device_to_index.size(); ++i) {
    if (single_device_shard_semantics ==
            SingleDeviceShardSemantics::kAllShards ||
        devices[i]->IsAddressable()) {
      int index = device_to_index[i];
      DCHECK_NE(index, kInvalidIndex);
      result.push_back(IndexDomain(origins[index / replication], local_shape));
    }
  }
  return result;
}

std::string ShardingParamSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ShardingParamSharding(%s, devices: %v, memory_kind: %v)",
      sharding_param_.DebugString(), *devices_, memory_kind_);
}

void ShardingParamSharding::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), devices_, memory_kind_,
                           is_fully_replicated_, sharding_param_);
}

}  // namespace ifrt
}  // namespace xla
