/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/sharding.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/index.h"
#include "tensorflow/compiler/xla/python/ifrt/index_domain.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

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
std::vector<Index> GetTileIndicies(absl::Span<const int64_t> dim_shards) {
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
StatusOr<Shape> GetDisassembledShape(const ShardingParam& sharding_param,
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
char ShardingParamSharding::ID = 0;

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << sharding.DebugString();
}

std::shared_ptr<const Sharding> SingleDeviceSharding::Create(Device* device) {
  return std::shared_ptr<const Sharding>(new SingleDeviceSharding(device));
}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
SingleDeviceSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument("Single-device sharding does not support disassembly");
}

StatusOr<std::vector<IndexDomain>> SingleDeviceSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  std::vector<IndexDomain> result;
  result.reserve(1);
  result.push_back(IndexDomain(shape));
  return result;
}

std::string SingleDeviceSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("SingleDeviceSharding(%s)",
                         devices_.front()->ToString());
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(DeviceList devices) {
  return std::shared_ptr<const Sharding>(new OpaqueSharding(
      std::move(devices),
      DisassembleFunc([](const OpaqueSharding& sharding,
                         const Shape& shape) -> StatusOr<std::vector<Shape>> {
        return FailedPrecondition(
            "Using an opaque sharding that disallows disassembly: "
            "sharding=%s; shape=%s",
            sharding.DebugString(), shape.DebugString());
      })));
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(
    DeviceList devices, DisassembleFunc disassemble_func) {
  return std::shared_ptr<const Sharding>(
      new OpaqueSharding(std::move(devices), std::move(disassemble_func)));
}

OpaqueSharding::DisassembleFunc OpaqueSharding::MakeDisassembleFuncFromShapes(
    std::vector<Shape> shapes) {
  // Capture shapes in a shared_ptr so that the disassemble function can be
  // copied cheaply.
  return DisassembleFunc(
      [shapes = std::make_shared<std::vector<Shape>>(std::move(shapes))](
          const OpaqueSharding&, const Shape&) -> StatusOr<std::vector<Shape>> {
        return *shapes;
      });
}

OpaqueSharding::OpaqueSharding(DeviceList devices,
                               DisassembleFunc disassemble_func)
    : llvm::RTTIExtends<OpaqueSharding, Sharding>(std::move(devices)),
      disassemble_func_(std::move(disassemble_func)) {}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
OpaqueSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto shapes, disassemble_func_(*this, shape));
  if (shapes.size() != devices_.size()) {
    return FailedPrecondition(
        "DisassembleFunc returned an incorrect number of shapes");
  }
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    result.push_back(
        {std::move(shapes[i]), SingleDeviceSharding::Create(devices_[i])});
  }
  return result;
}

StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have index domain information");
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "OpaqueSharding(%s)",
      absl::StrJoin(devices_, ",", [](std::string* out, const Device* device) {
        absl::StrAppend(out, device->ToString());
      }));
}

StatusOr<std::shared_ptr<const Sharding>> ShardingParamSharding::Create(
    ShardingParam sharding_param, DeviceList devices) {
  int64_t device_count =
      absl::c_accumulate(sharding_param.minor_to_major().axis_sizes, 1,
                         std::multiplies<int64_t>());
  if (device_count != devices.size()) {
    return FailedPrecondition(
        "Device counts don't match. From ShardingParam %d vs from DeviceList "
        "%d",
        device_count, devices.size());
  }
  return std::shared_ptr<const Sharding>(
      new ShardingParamSharding(std::move(sharding_param), std::move(devices)));
}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
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
    result.push_back({local_shape, SingleDeviceSharding::Create(device)});
  }

  return result;
}

StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);

  // Calculate the origins of tiles, ignoring device assignments.
  TF_ASSIGN_OR_RETURN(Shape local_shape,
                      GetDisassembledShape(sharding_param_, shape));
  std::vector<Index> tile_indices =
      GetTileIndicies(sharding_param_.dim_shards());
  std::vector<Index> origins;
  origins.reserve(tile_indices.size());
  for (const Index& tile_index : tile_indices) {
    origins.push_back(tile_index * local_shape.dims());
  }

  // Calculate the device assignments.
  // `origins[i]` should go to `device_list[i]`.
  static constexpr int64_t kInvalidIndex = -1;
  llvm::SmallVector<int64_t, 4> device_list;
  sharding_param_.minor_to_major().ToDeviceList(device_list);
  std::vector<int64_t> device_to_index(device_list.size(), kInvalidIndex);
  for (int i = 0; i < device_list.size(); ++i) {
    device_to_index[device_list[i]] = i;
  }

  // Replication is the minor axis in `device_list`.
  DCHECK_EQ(device_to_index.size() % origins.size(), 0);
  int replication = device_to_index.size() / origins.size();

  std::vector<IndexDomain> result;
  result.reserve(device_to_index.size());
  for (int i = 0; i < device_to_index.size(); ++i) {
    int64_t index = device_to_index[i];
    DCHECK_NE(index, kInvalidIndex);
    result.push_back(IndexDomain(origins[index / replication], local_shape));
  }
  return result;
}

std::string ShardingParamSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ShardingParamSharding(%s, devices: %s)", sharding_param_.DebugString(),
      absl::StrJoin(devices_, ",", [](std::string* out, const Device* device) {
        absl::StrAppend(out, device->ToString());
      }));
}

}  // namespace ifrt
}  // namespace xla
