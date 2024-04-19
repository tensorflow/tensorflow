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
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/shape.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

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

// Returns the tile shape after disassembling `shape` with `sharding_param`.
//
// Fails if can't shard evenly.
absl::StatusOr<Shape> GetDisassembledShape(const ShardingParam& sharding_param,
                                           const Shape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.dims().size());
  for (const auto [dim, dim_shards] :
       llvm::zip(shape.dims(), sharding_param.dim_shards())) {
    if (dim % dim_shards != 0) {
      return FailedPrecondition(
          "Uneven shard is not supported. dim: %d, dim_shards: %d", dim,
          dim_shards);
    }
    dims.push_back(dim / dim_shards);
  }
  return Shape(dims);
}

}  // namespace

char Sharding::ID = 0;
char SingleDeviceSharding::ID = 0;
char OpaqueSharding::ID = 0;
char ConcreteSharding::ID = 0;
char ConcreteEvenSharding::ID = 0;
char ShardingParamSharding::ID = 0;

char DeserializeShardingOptions::ID = 0;

absl::StatusOr<std::unique_ptr<Sharding>> Sharding::FromProto(
    DeviceList::LookupDeviceFunc lookup_device,
    const ShardingProto& sharding_proto) {
  return Deserialize<Sharding>(
      sharding_proto.serialized_sharding(),
      std::make_unique<DeserializeShardingOptions>(std::move(lookup_device)));
}

absl::StatusOr<ShardingProto> Sharding::ToProto() const {
  ShardingProto sharding_proto;
  TF_ASSIGN_OR_RETURN(*sharding_proto.mutable_serialized_sharding(),
                      Serialize(const_cast<Sharding&>(*this)));
  return sharding_proto;
}

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << sharding.DebugString();
}

std::unique_ptr<SingleDeviceSharding> SingleDeviceSharding::Create(
    Device* device, MemoryKind memory_kind) {
  return std::unique_ptr<SingleDeviceSharding>(
      new SingleDeviceSharding(device, memory_kind));
}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
SingleDeviceSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>{
      {shape, SingleDeviceSharding::Create(devices_[0], memory_kind_)}};
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
SingleDeviceSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>{
      {dynamic_shape, SingleDeviceSharding::Create(devices_[0], memory_kind_)}};
}

absl::StatusOr<std::vector<IndexDomain>> SingleDeviceSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  std::vector<IndexDomain> result;
  result.reserve(1);
  result.push_back(IndexDomain(shape));
  return result;
}

std::string SingleDeviceSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("SingleDeviceSharding(%s, memory_kind: %s)",
                         devices_.front()->ToString(),
                         memory_kind_.DebugString());
}

std::unique_ptr<OpaqueSharding> OpaqueSharding::Create(DeviceList devices,
                                                       MemoryKind memory_kind) {
  return std::unique_ptr<OpaqueSharding>(
      new OpaqueSharding(std::move(devices), memory_kind));
}

OpaqueSharding::OpaqueSharding(DeviceList devices, MemoryKind memory_kind)
    : llvm::RTTIExtends<OpaqueSharding, Sharding>(std::move(devices),
                                                  memory_kind) {}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
OpaqueSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
OpaqueSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have shard shape information");
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have index domain information");
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "OpaqueSharding(devices: %s, memory_kind: %s)",
      absl::StrJoin(devices_, ",",
                    [](std::string* out, const Device* device) {
                      absl::StrAppend(out, device->ToString());
                    }),
      memory_kind_.DebugString());
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceList devices, MemoryKind memory_kind, Shape shape,
    std::vector<Shape> shard_shapes) {
  CHECK_EQ(devices.size(), shard_shapes.size());
  return std::unique_ptr<ConcreteSharding>(
      new ConcreteSharding(std::move(devices), memory_kind, std::move(shape),
                           std::move(shard_shapes)));
}

std::unique_ptr<ConcreteSharding> ConcreteSharding::Create(
    DeviceList devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes) {
  CHECK_EQ(devices.size(), shard_dynamic_shapes.size());
  return std::unique_ptr<ConcreteSharding>(new ConcreteSharding(
      std::move(devices), memory_kind, std::move(dynamic_shape),
      std::move(shard_dynamic_shapes)));
}

ConcreteSharding::ConcreteSharding(DeviceList devices, MemoryKind memory_kind,
                                   Shape shape, std::vector<Shape> shard_shapes)
    : llvm::RTTIExtends<ConcreteSharding, Sharding>(std::move(devices),
                                                    memory_kind),
      shape_(std::move(shape)),
      shard_shapes_(std::move(shard_shapes)) {}

ConcreteSharding::ConcreteSharding(
    DeviceList devices, MemoryKind memory_kind, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes)
    : llvm::RTTIExtends<ConcreteSharding, Sharding>(std::move(devices),
                                                    memory_kind),
      shape_(std::move(dynamic_shape)),
      shard_shapes_(std::move(shard_dynamic_shapes)) {}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
ConcreteSharding::Disassemble(const Shape& shape) const {
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
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(devices_.size());
  const std::vector<Shape>& shard_shapes =
      std::get<std::vector<Shape>>(shard_shapes_);
  for (int i = 0; i < devices_.size(); ++i) {
    result.push_back({shard_shapes[i],
                      SingleDeviceSharding::Create(devices_[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
ConcreteSharding::Disassemble(const DynamicShape& dynamic_shape) const {
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
  std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>> result;
  result.reserve(devices_.size());
  const std::vector<DynamicShape>& shard_dynamic_shapes =
      std::get<std::vector<DynamicShape>>(shard_shapes_);
  for (int i = 0; i < devices_.size(); ++i) {
    result.push_back({shard_dynamic_shapes[i],
                      SingleDeviceSharding::Create(devices_[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "ConcreteSharding does not have index domain information");
}

std::string ConcreteSharding::DebugString() const {
  DCHECK(this);
  return std::visit(
      [this](const auto& shape, const auto& shard_shapes) {
        return absl::StrFormat(
            "ConcreteSharding(devices: %s, shape: %s, shard_shapes: %s, "
            "memory_kind: %s)",
            absl::StrJoin(devices_, ",",
                          [](std::string* out, const Device* device) {
                            absl::StrAppend(out, device->ToString());
                          }),
            shape.DebugString(),
            absl::StrJoin(shard_shapes, ",",
                          [](std::string* out, const auto& shard_shape) {
                            absl::StrAppend(out, shard_shape.DebugString());
                          }),
            memory_kind_.DebugString());
      },
      shape_, shard_shapes_);
}

std::unique_ptr<ConcreteEvenSharding> ConcreteEvenSharding::Create(
    DeviceList devices, MemoryKind memory_kind, Shape shape,
    Shape shard_shape) {
  return std::unique_ptr<ConcreteEvenSharding>(
      new ConcreteEvenSharding(std::move(devices), memory_kind,
                               std::move(shape), std::move(shard_shape)));
}

ConcreteEvenSharding::ConcreteEvenSharding(DeviceList devices,
                                           MemoryKind memory_kind, Shape shape,
                                           Shape shard_shape)
    : llvm::RTTIExtends<ConcreteEvenSharding, Sharding>(std::move(devices),
                                                        memory_kind),
      shape_(std::move(shape)),
      shard_shape_(std::move(shard_shape)) {}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
ConcreteEvenSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  if (shape != shape_) {
    return InvalidArgument(
        "ConcreteEvenSharding can only disassemble shape %s, but was asked "
        "to disassemble shape %s",
        shape_.DebugString(), shape.DebugString());
  }
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(devices_.size());
  for (int i = 0; i < devices_.size(); ++i) {
    result.push_back({shard_shape_,
                      SingleDeviceSharding::Create(devices_[i], memory_kind_)});
  }
  return result;
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
ConcreteEvenSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "ConcreteEvenSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "ConcreteEvenSharding does not have index domain information");
}

std::string ConcreteEvenSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ConcreteEvenSharding(devices: %s, shape: %s, shard_shape: %s, "
      "memory_kind: %s)",
      absl::StrJoin(devices_, ",",
                    [](std::string* out, const Device* device) {
                      absl::StrAppend(out, device->ToString());
                    }),
      shape_.DebugString(), shard_shape_.DebugString(),
      memory_kind_.DebugString());
}

absl::StatusOr<std::unique_ptr<ShardingParamSharding>>
ShardingParamSharding::Create(ShardingParam sharding_param, DeviceList devices,
                              MemoryKind memory_kind) {
  int64_t device_count =
      absl::c_accumulate(sharding_param.minor_to_major().axis_sizes, 1,
                         std::multiplies<int64_t>());
  if (device_count != devices.size()) {
    return FailedPrecondition(
        "Device counts don't match. From ShardingParam %d vs from DeviceList "
        "%d",
        device_count, devices.size());
  }
  return std::unique_ptr<ShardingParamSharding>(new ShardingParamSharding(
      std::move(sharding_param), std::move(devices), memory_kind));
}

absl::StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
ShardingParamSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  if (shape.dims().size() != sharding_param_.dim_shards().size()) {
    return FailedPrecondition(
        "Ranks don't match. From Shape %d vs from ShardingParam %d",
        shape.dims().size(), sharding_param_.dim_shards().size());
  }

  TF_ASSIGN_OR_RETURN(Shape local_shape,
                      GetDisassembledShape(sharding_param_, shape));

  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  for (Device* device : devices_) {
    result.push_back(
        {local_shape, SingleDeviceSharding::Create(device, memory_kind_)});
  }

  return result;
}

absl::StatusOr<
    std::vector<std::pair<DynamicShape, std::shared_ptr<const Sharding>>>>
ShardingParamSharding::Disassemble(const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "ShardingParamSharding can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);

  // Calculate the origins of tiles, ignoring device assignments.
  TF_ASSIGN_OR_RETURN(Shape local_shape,
                      GetDisassembledShape(sharding_param_, shape));
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

  std::vector<IndexDomain> result;
  result.reserve(device_to_index.size());
  for (int i = 0; i < device_to_index.size(); ++i) {
    int index = device_to_index[i];
    DCHECK_NE(index, kInvalidIndex);
    result.push_back(IndexDomain(origins[index / replication], local_shape));
  }
  return result;
}

std::string ShardingParamSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ShardingParamSharding(%s, devices: %s, memory_kind: %s)",
      sharding_param_.DebugString(),
      absl::StrJoin(devices_, ",",
                    [](std::string* out, const Device* device) {
                      absl::StrAppend(out, device->ToString());
                    }),
      memory_kind_.DebugString());
}

}  // namespace ifrt
}  // namespace xla
