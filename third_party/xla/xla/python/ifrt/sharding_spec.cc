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

#include "xla/python/ifrt/sharding_spec.h"

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
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

namespace {

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

char ShardingSpec::ID = 0;
char SingleDeviceShardingSpec::ID = 0;
char OpaqueShardingSpec::ID = 0;
char ConcreteShardingSpec::ID = 0;
char ConcreteEvenShardingSpec::ID = 0;
char ShardingParamShardingSpec::ID = 0;

ShardingSpec::ShardingSpec(int num_shards, bool is_fully_replicated)
    : num_shards_(num_shards), is_fully_replicated_(is_fully_replicated) {}

bool ShardingSpec::operator==(const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  return num_shards_ == other.num_shards_ && HasSamePartitioning(other);
}

absl::StatusOr<ShardingSpecRef> ShardingSpec::FromProto(
    const ShardingSpecProto& sharding_spec_proto) {
  return Deserialize<ShardingSpec>(
      sharding_spec_proto.serialized_sharding_spec(), /*options=*/nullptr);
}

absl::Status ShardingSpec::ToProto(ShardingSpecProto& sharding_spec_proto,
                                   SerDesVersion version) const {
  // `ShardingSpecProto` does not store its own version. It delegates the
  // details to SerDes of the `ShardingSpec` subclasses.
  auto options = std::make_unique<SerializeOptions>(version);
  return Serialize(*this, std::move(options),
                   *sharding_spec_proto.mutable_serialized_sharding_spec());
}

std::ostream& operator<<(std::ostream& os, const ShardingSpec& sharding_spec) {
  return os << sharding_spec.DebugString();
}

std::unique_ptr<SingleDeviceShardingSpec> SingleDeviceShardingSpec::Create() {
  return std::unique_ptr<SingleDeviceShardingSpec>(
      new SingleDeviceShardingSpec());
}

SingleDeviceShardingSpec::SingleDeviceShardingSpec()
    : llvm::RTTIExtends<SingleDeviceShardingSpec, ShardingSpec>(
          /*num_shards=*/1, /*is_fully_replicated=*/true) {}

absl::StatusOr<Shape> SingleDeviceShardingSpec::GetShardShape(
    const Shape& shape) const {
  return shape;
}

bool SingleDeviceShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  return llvm::isa<SingleDeviceShardingSpec>(&other);
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
SingleDeviceShardingSpec::Disassemble(const Shape& shape) const {
  return std::vector<std::pair<Shape, ShardingSpecRef>>{
      {shape, SingleDeviceShardingSpec::Create()}};
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
SingleDeviceShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return std::vector<std::pair<DynamicShape, ShardingSpecRef>>{
      {dynamic_shape, SingleDeviceShardingSpec::Create()}};
}

absl::StatusOr<std::vector<IndexDomain>> SingleDeviceShardingSpec::IndexDomains(
    const Shape& shape) const {
  return std::vector<IndexDomain>{IndexDomain(shape)};
}

std::string SingleDeviceShardingSpec::DebugString() const {
  return "SingleDeviceShardingSpec()";
}

void SingleDeviceShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_);
}

std::unique_ptr<OpaqueShardingSpec> OpaqueShardingSpec::Create(int num_shards) {
  return std::unique_ptr<OpaqueShardingSpec>(
      new OpaqueShardingSpec(num_shards));
}

OpaqueShardingSpec::OpaqueShardingSpec(int num_shards)
    : llvm::RTTIExtends<OpaqueShardingSpec, ShardingSpec>(
          num_shards, /*is_fully_replicated=*/false) {}

absl::StatusOr<Shape> OpaqueShardingSpec::GetShardShape(
    const Shape& shape) const {
  return InvalidArgument(
      "OpaqueShardingSpec does not have shard shape information");
}

bool OpaqueShardingSpec::HasSamePartitioning(const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  // If the objects are not the same, we cannot tell whether the two
  // OpaqueShardingSpecs are using the same logical partitioning.
  return this == &other;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
OpaqueShardingSpec::Disassemble(const Shape& shape) const {
  return InvalidArgument(
      "OpaqueShardingSpec does not have shard shape information");
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
OpaqueShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "OpaqueShardingSpec does not have shard shape information");
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueShardingSpec::IndexDomains(
    const Shape& shape) const {
  return InvalidArgument(
      "OpaqueShardingSpec does not have index domain information");
}

std::string OpaqueShardingSpec::DebugString() const {
  return absl::StrFormat("OpaqueShardingSpec(num_shards: %d)", num_shards_);
}

void OpaqueShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_);
}

std::unique_ptr<ConcreteShardingSpec> ConcreteShardingSpec::Create(
    Shape shape, std::vector<Shape> shard_shapes,
    std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains) {
  int num_shards = shard_shapes.size();
  return std::unique_ptr<ConcreteShardingSpec>(new ConcreteShardingSpec(
      num_shards, std::move(shape), std::move(shard_shapes),
      std::move(index_domains)));
}

std::unique_ptr<ConcreteShardingSpec> ConcreteShardingSpec::Create(
    DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes) {
  int num_shards = shard_dynamic_shapes.size();
  return std::unique_ptr<ConcreteShardingSpec>(new ConcreteShardingSpec(
      num_shards, std::move(dynamic_shape), std::move(shard_dynamic_shapes)));
}

ConcreteShardingSpec::ConcreteShardingSpec(
    int num_shards, Shape shape, std::vector<Shape> shard_shapes,
    std::optional<std::vector<xla::ifrt::IndexDomain>> index_domains)
    : llvm::RTTIExtends<ConcreteShardingSpec, ShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      shape_(std::move(shape)),
      shard_shapes_(std::move(shard_shapes)),
      index_domains_(std::move(index_domains)) {
  // If all per-shard shapes are the same, cache this shape for
  // `GetShardShape()`. Ideally, users should have used
  // `ConcreteEvenShardingSpec` for such a case, but there are existing use
  // cases that instantiate `ConcreteShardingSpec` from a list of per-shard
  // shapes without checking for identical per-shard shapes.
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

ConcreteShardingSpec::ConcreteShardingSpec(
    int num_shards, DynamicShape dynamic_shape,
    std::vector<DynamicShape> shard_dynamic_shapes)
    : llvm::RTTIExtends<ConcreteShardingSpec, ShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      shape_(std::move(dynamic_shape)),
      shard_shapes_(std::move(shard_dynamic_shapes)) {}

absl::StatusOr<Shape> ConcreteShardingSpec::GetShardShape(
    const Shape& shape) const {
  if (shard_shape_.has_value()) {
    return *shard_shape_;
  }
  return InvalidArgument(
      "ConcreteShardingSpec does not have a fixed shard shape");
}

bool ConcreteShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_sharding_spec =
      llvm::dyn_cast<ConcreteShardingSpec>(&other);
  if (!other_concrete_sharding_spec) {
    return false;
  }
  return shape_ == other_concrete_sharding_spec->shape_ &&
         shard_shapes_ == other_concrete_sharding_spec->shard_shapes_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
ConcreteShardingSpec::Disassemble(const Shape& shape) const {
  if (!has_static_shape()) {
    return InvalidArgument(
        "ConcreteShardingSpec holds dynamic shape, but was asked "
        "to disassemble static shape %s",
        shape.DebugString());
  }
  if (shape != std::get<Shape>(shape_)) {
    return InvalidArgument(
        "ConcreteShardingSpec can only disassemble shape %s, but was asked "
        "to disassemble shape %s",
        std::get<Shape>(shape_).DebugString(), shape.DebugString());
  }
  const std::vector<Shape>& shard_shapes =
      std::get<std::vector<Shape>>(shard_shapes_);
  std::vector<std::pair<Shape, ShardingSpecRef>> result;
  result.reserve(shard_shapes.size());
  for (const auto& shard_shape : shard_shapes) {
    result.push_back({shard_shape, SingleDeviceShardingSpec::Create()});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
ConcreteShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  if (!has_dynamic_shape()) {
    return InvalidArgument(
        "ConcreteShardingSpec holds static shape, but was asked "
        "to disassemble dynamic shape %s",
        dynamic_shape.DebugString());
  }
  if (dynamic_shape != std::get<DynamicShape>(shape_)) {
    return InvalidArgument(
        "ConcreteShardingSpec can only disassemble dynamic shape %s, but was "
        "asked to disassemble dynamic shape %s",
        std::get<DynamicShape>(shape_).DebugString(),
        dynamic_shape.DebugString());
  }
  const std::vector<DynamicShape>& shard_dynamic_shapes =
      std::get<std::vector<DynamicShape>>(shard_shapes_);
  std::vector<std::pair<DynamicShape, ShardingSpecRef>> result;
  result.reserve(shard_dynamic_shapes.size());
  for (const auto& shard_dynamic_shape : shard_dynamic_shapes) {
    result.push_back({shard_dynamic_shape, SingleDeviceShardingSpec::Create()});
  }
  return result;
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteShardingSpec::IndexDomains(
    const Shape& shape) const {
  if (!index_domains_.has_value()) {
    return InvalidArgument(
        "ConcreteShardingSpec does not have index domain information");
  }
  return *index_domains_;
}

std::string ConcreteShardingSpec::DebugString() const {
  return std::visit(
      [this](const auto& shape, const auto& shard_shapes) {
        return absl::StrFormat(
            "ConcreteShardingSpec(num_shards: %d, shape: %s, "
            "shard_shapes: [%s], index_domains: %s)",
            num_shards_, shape.DebugString(),
            absl::StrJoin(shard_shapes, ",",
                          [](std::string* out, const auto& shard_shape) {
                            absl::StrAppend(out, shard_shape.DebugString());
                          }),
            index_domains_.has_value()
                ? absl::StrCat("[", absl::StrJoin(*index_domains_, ","), "]")
                : "<nullopt>");
      },
      shape_, shard_shapes_);
}

void ConcreteShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_, shape_,
                           shard_shapes_);
}

std::unique_ptr<ConcreteEvenShardingSpec> ConcreteEvenShardingSpec::Create(
    int num_shards, Shape shape, Shape shard_shape, bool is_fully_replicated) {
  return std::unique_ptr<ConcreteEvenShardingSpec>(new ConcreteEvenShardingSpec(
      num_shards, std::move(shape), std::move(shard_shape),
      is_fully_replicated));
}

ConcreteEvenShardingSpec::ConcreteEvenShardingSpec(int num_shards, Shape shape,
                                                   Shape shard_shape,
                                                   bool is_fully_replicated)
    : llvm::RTTIExtends<ConcreteEvenShardingSpec, ShardingSpec>(
          num_shards, is_fully_replicated),
      shape_(std::move(shape)),
      shard_shape_(std::move(shard_shape)) {}

absl::StatusOr<Shape> ConcreteEvenShardingSpec::GetShardShape(
    const Shape& shape) const {
  if (shape != shape_) {
    return InvalidArgument(
        "ConcreteEvenShardingSpec has a shard shape for shape %s, but was "
        "asked to get a shard shape for shape %s",
        shape_.DebugString(), shape.DebugString());
  }
  return shard_shape_;
}

bool ConcreteEvenShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_even_sharding_spec =
      llvm::dyn_cast<ConcreteEvenShardingSpec>(&other);
  if (!other_concrete_even_sharding_spec) {
    return false;
  }
  return num_shards_ == other_concrete_even_sharding_spec->num_shards_ &&
         shape_ == other_concrete_even_sharding_spec->shape_ &&
         shard_shape_ == other_concrete_even_sharding_spec->shard_shape_ &&
         is_fully_replicated_ ==
             other_concrete_even_sharding_spec->is_fully_replicated_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
ConcreteEvenShardingSpec::Disassemble(const Shape& shape) const {
  if (shape != shape_) {
    return InvalidArgument(
        "ConcreteEvenShardingSpec can only disassemble shape %s, but was "
        "asked to disassemble shape %s",
        shape_.DebugString(), shape.DebugString());
  }
  std::vector<std::pair<Shape, ShardingSpecRef>> result;
  result.reserve(num_shards_);
  for (int i = 0; i < num_shards_; ++i) {
    result.push_back({shard_shape_, SingleDeviceShardingSpec::Create()});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
ConcreteEvenShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "ConcreteEvenShardingSpec can only disassemble static shape, but was "
      "asked to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenShardingSpec::IndexDomains(
    const Shape& shape) const {
  return InvalidArgument(
      "ConcreteEvenShardingSpec does not have index domain information");
}

std::string ConcreteEvenShardingSpec::DebugString() const {
  return absl::StrFormat(
      "ConcreteEvenShardingSpec(num_shards: %d, shape: %s, "
      "shard_shape: %s, is_fully_replicated: %s)",
      num_shards_, shape_.DebugString(), shard_shape_.DebugString(),
      is_fully_replicated_ ? "true" : "false");
}

void ConcreteEvenShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_, is_fully_replicated_,
                           shape_, shard_shape_);
}

std::unique_ptr<ShardingParamShardingSpec> ShardingParamShardingSpec::Create(
    ShardingParam sharding_param) {
  int num_shards = sharding_param.NumDevices();
  return std::unique_ptr<ShardingParamShardingSpec>(
      new ShardingParamShardingSpec(num_shards, std::move(sharding_param)));
}

ShardingParamShardingSpec::ShardingParamShardingSpec(
    int num_shards, ShardingParam sharding_param)
    : llvm::RTTIExtends<ShardingParamShardingSpec, ShardingSpec>(
          num_shards, ComputeIsFullyReplicated(sharding_param)),
      sharding_param_(std::move(sharding_param)) {}

absl::StatusOr<Shape> ShardingParamShardingSpec::GetShardShape(
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

bool ShardingParamShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_sharding_param_sharding_spec =
      llvm::dyn_cast<ShardingParamShardingSpec>(&other);
  if (!other_sharding_param_sharding_spec) {
    return false;
  }
  return sharding_param_ == other_sharding_param_sharding_spec->sharding_param_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
ShardingParamShardingSpec::Disassemble(const Shape& shape) const {
  TF_ASSIGN_OR_RETURN(Shape local_shape, GetShardShape(shape));
  std::vector<std::pair<Shape, ShardingSpecRef>> result;
  result.reserve(num_shards_);
  for (int i = 0; i < num_shards_; ++i) {
    result.push_back({local_shape, SingleDeviceShardingSpec::Create()});
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
ShardingParamShardingSpec::Disassemble(
    const DynamicShape& dynamic_shape) const {
  return InvalidArgument(
      "ShardingParamShardingSpec can only disassemble static shape, but was "
      "asked to disassemble dynamic shape %s",
      dynamic_shape.DebugString());
}

absl::StatusOr<std::vector<IndexDomain>>
ShardingParamShardingSpec::IndexDomains(const Shape& shape) const {
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

  DCHECK_EQ(device_to_index.size(), num_shards_);
  std::vector<IndexDomain> result;
  result.reserve(num_shards_);
  for (int i = 0; i < device_to_index.size(); ++i) {
    int index = device_to_index[i];
    DCHECK_NE(index, kInvalidIndex);
    result.push_back(IndexDomain(origins[index / replication], local_shape));
  }
  return result;
}

std::string ShardingParamShardingSpec::DebugString() const {
  return absl::StrFormat("ShardingParamShardingSpec(num_shards: %d, %s)",
                         num_shards_, sharding_param_.DebugString());
}

void ShardingParamShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_, is_fully_replicated_,
                           sharding_param_);
}

}  // namespace ifrt
}  // namespace xla
