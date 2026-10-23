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

#include <array>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_spec.pb.h"

namespace xla {
namespace ifrt {

char ShardingSpec::ID = 0;
char SingleDeviceShardingSpec::ID = 0;
char OpaqueShardingSpec::ID = 0;
char ConcreteShardingSpec::ID = 0;
char ConcreteEvenShardingSpec::ID = 0;

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
  return os << absl::StrCat(sharding_spec);
}

std::unique_ptr<SingleDeviceShardingSpec> SingleDeviceShardingSpec::Create() {
  return std::unique_ptr<SingleDeviceShardingSpec>(
      new SingleDeviceShardingSpec());
}

SingleDeviceShardingSpec::SingleDeviceShardingSpec()
    : RTTIExtends<SingleDeviceShardingSpec, ShardingSpec>(
          /*num_shards=*/1, /*is_fully_replicated=*/true) {}

absl::StatusOr<ShardingRef> SingleDeviceShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices->size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SingleDeviceShardingSpec requires 1 device, but received %d devices",
        devices->size()));
  }
  std::shared_ptr<const SingleDeviceShardingSpec> spec =
      std::static_pointer_cast<const SingleDeviceShardingSpec>(
          weak_from_this().lock());
  if (spec == nullptr) {
    spec = SingleDeviceShardingSpec::Create();
  }
  return std::unique_ptr<SingleDeviceSharding>(new SingleDeviceSharding(
      std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> SingleDeviceShardingSpec::GetShardShape(
    const Shape& shape) const {
  return shape;
}

bool SingleDeviceShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  return isa<SingleDeviceShardingSpec>(&other);
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

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
SingleDeviceShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  static constexpr std::array<int, 1> kShardIndices({0});
  return absl::InlinedVector<IndexDomainAndShardIndices, 1>{
      IndexDomainAndShardIndices{
          /*index_domain=*/IndexDomain(shape),
          /*shard_indices=*/absl::MakeConstSpan(kShardIndices),
      },
  };
}

absl::StatusOr<absl::Span<const int>>
SingleDeviceShardingSpec::ShardToUniqueIndexDomainIndex() const {
  static constexpr std::array<int, 1> kShardToUniqueIndexDomainIndex({0});
  return absl::MakeConstSpan(kShardToUniqueIndexDomainIndex);
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
    : RTTIExtends<OpaqueShardingSpec, ShardingSpec>(
          num_shards, /*is_fully_replicated=*/false) {}

absl::StatusOr<ShardingRef> OpaqueShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices->size() != num_shards()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OpaqueShardingSpec requires %d devices, but received %d devices",
        num_shards(), devices->size()));
  }
  std::shared_ptr<const OpaqueShardingSpec> spec =
      std::static_pointer_cast<const OpaqueShardingSpec>(
          weak_from_this().lock());
  if (spec == nullptr) {
    spec = OpaqueShardingSpec::Create(num_shards());
  }
  return std::unique_ptr<OpaqueSharding>(
      new OpaqueSharding(std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> OpaqueShardingSpec::GetShardShape(
    const Shape& shape) const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not have shard shape information");
}

bool OpaqueShardingSpec::HasSamePartitioning(const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  // If the objects are not the same, we cannot tell whether the two
  // OpaqueShardingSpecs are using the same logical partitioning.
  return false;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
OpaqueShardingSpec::Disassemble(const Shape& shape) const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not have shard shape information");
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
OpaqueShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not have shard shape information");
}

absl::StatusOr<std::vector<IndexDomain>> OpaqueShardingSpec::IndexDomains(
    const Shape& shape) const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not have index domain information");
}

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
OpaqueShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not support UniqueIndexDomains");
}

absl::StatusOr<absl::Span<const int>>
OpaqueShardingSpec::ShardToUniqueIndexDomainIndex() const {
  return absl::InvalidArgumentError(
      "OpaqueShardingSpec does not support ShardToUniqueIndexDomainIndex");
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
    : RTTIExtends<ConcreteShardingSpec, ShardingSpec>(
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
    : RTTIExtends<ConcreteShardingSpec, ShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      shape_(std::move(dynamic_shape)),
      shard_shapes_(std::move(shard_dynamic_shapes)) {}

ConcreteShardingSpec::ConcreteShardingSpec(const ConcreteShardingSpec& other)
    : RTTIExtends<ConcreteShardingSpec, ShardingSpec>(other),
      shape_(other.shape_),
      shard_shapes_(other.shard_shapes_),
      shard_shape_(other.shard_shape_),
      index_domains_(other.index_domains_) {}

absl::StatusOr<ShardingRef> ConcreteShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices->size() != num_shards()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec requires %d devices, but received %d devices",
        num_shards(), devices->size()));
  }
  std::shared_ptr<const ConcreteShardingSpec> spec =
      std::static_pointer_cast<const ConcreteShardingSpec>(
          weak_from_this().lock());
  if (spec == nullptr) {
    if (has_static_shape()) {
      spec = ConcreteShardingSpec::Create(shape(), shard_shapes(),
                                          index_domains());
    } else {
      spec =
          ConcreteShardingSpec::Create(dynamic_shape(), shard_dynamic_shapes());
    }
  }
  return std::unique_ptr<ConcreteSharding>(
      new ConcreteSharding(std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> ConcreteShardingSpec::GetShardShape(
    const Shape& shape) const {
  if (shard_shape_.has_value()) {
    return *shard_shape_;
  }
  return absl::InvalidArgumentError(
      "ConcreteShardingSpec does not have a fixed shard shape");
}

bool ConcreteShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_sharding_spec =
      dyn_cast<ConcreteShardingSpec>(&other);
  if (!other_concrete_sharding_spec) {
    return false;
  }
  return shape_ == other_concrete_sharding_spec->shape_ &&
         shard_shapes_ == other_concrete_sharding_spec->shard_shapes_ &&
         index_domains_ == other_concrete_sharding_spec->index_domains_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
ConcreteShardingSpec::Disassemble(const Shape& shape) const {
  if (!has_static_shape()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec holds dynamic shape, but was asked "
        "to disassemble static shape %v",
        shape));
  }
  if (shape != std::get<Shape>(shape_)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec can only disassemble shape %v, but was asked "
        "to disassemble shape %v",
        std::get<Shape>(shape_), shape));
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
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec holds static shape, but was asked "
        "to disassemble dynamic shape %v",
        dynamic_shape));
  }
  if (dynamic_shape != std::get<DynamicShape>(shape_)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec can only disassemble dynamic shape %v, but was "
        "asked to disassemble dynamic shape %v",
        std::get<DynamicShape>(shape_), dynamic_shape));
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
    return absl::InvalidArgumentError(
        "ConcreteShardingSpec does not have index domain information");
  }
  return *index_domains_;
}

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
ConcreteShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  if (!index_domains_.has_value()) {
    return absl::InvalidArgumentError(
        "ConcreteShardingSpec does not have index domain information");
  }
  if (has_static_shape() && this->shape() != shape) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteShardingSpec has index domains for shape %v, but was asked "
        "to get unique index domains for shape %v",
        this->shape(), shape));
  }
  absl::call_once(unique_shard_indices_once_, [this] {
    absl::flat_hash_map<IndexDomain, int> index_domain_to_unique_idx;
    std::vector<std::vector<int>> shard_indices;
    for (int i = 0; i < index_domains_->size(); ++i) {
      const IndexDomain& domain = (*index_domains_)[i];
      auto [it, inserted] =
          index_domain_to_unique_idx.try_emplace(domain, shard_indices.size());
      if (inserted) {
        shard_indices.emplace_back();
      }
      shard_indices[it->second].push_back(i);
    }
    cached_shard_indices_.reserve(index_domains_->size());
    cached_shard_indices_offsets_.reserve(shard_indices.size() + 1);
    for (const auto& indices : shard_indices) {
      cached_shard_indices_offsets_.push_back(cached_shard_indices_.size());
      cached_shard_indices_.insert(cached_shard_indices_.end(), indices.begin(),
                                   indices.end());
    }
    cached_shard_indices_offsets_.push_back(cached_shard_indices_.size());
  });

  const int num_unique = cached_shard_indices_offsets_.size() - 1;
  absl::InlinedVector<IndexDomainAndShardIndices, 1> unique_domains;
  unique_domains.reserve(num_unique);
  for (int i = 0; i < num_unique; ++i) {
    const int offset = cached_shard_indices_offsets_[i];
    const int count = cached_shard_indices_offsets_[i + 1] - offset;
    const int first_shard = cached_shard_indices_[offset];
    unique_domains.push_back(IndexDomainAndShardIndices{
        /*index_domain=*/(*index_domains_)[first_shard],
        /*shard_indices=*/
        absl::MakeConstSpan(cached_shard_indices_).subspan(offset, count),
    });
  }
  return unique_domains;
}

absl::StatusOr<absl::Span<const int>>
ConcreteShardingSpec::ShardToUniqueIndexDomainIndex() const {
  if (!index_domains_.has_value()) {
    return absl::InvalidArgumentError(
        "ConcreteShardingSpec does not have index domain information");
  }
  absl::call_once(shard_to_unique_index_domain_index_once_, [this] {
    absl::flat_hash_map<IndexDomain, int> domain_to_unique_idx;
    cached_shard_to_unique_index_domain_index_.reserve(index_domains_->size());
    for (int i = 0; i < index_domains_->size(); ++i) {
      const IndexDomain& domain = (*index_domains_)[i];
      auto it =
          domain_to_unique_idx.try_emplace(domain, domain_to_unique_idx.size())
              .first;
      cached_shard_to_unique_index_domain_index_.push_back(it->second);
    }
  });
  return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
}

std::string ConcreteShardingSpec::DebugString() const {
  return std::visit(
      [this](const auto& shape, const auto& shard_shapes) {
        return absl::StrFormat(
            "ConcreteShardingSpec(num_shards: %d, shape: %v, "
            "shard_shapes: [%s], index_domains: %s)",
            num_shards_, shape, absl::StrJoin(shard_shapes, ","),
            index_domains_.has_value()
                ? absl::StrCat("[", absl::StrJoin(*index_domains_, ","), "]")
                : "<nullopt>");
      },
      shape_, shard_shapes_);
}

void ConcreteShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_, shape_, shard_shapes_,
                           index_domains_);
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
    : RTTIExtends<ConcreteEvenShardingSpec, ShardingSpec>(num_shards,
                                                          is_fully_replicated),
      shape_(std::move(shape)),
      shard_shape_(std::move(shard_shape)) {}

ConcreteEvenShardingSpec::ConcreteEvenShardingSpec(
    const ConcreteEvenShardingSpec& other)
    : RTTIExtends<ConcreteEvenShardingSpec, ShardingSpec>(other),
      shape_(other.shape_),
      shard_shape_(other.shard_shape_) {}

absl::StatusOr<ShardingRef> ConcreteEvenShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices->size() != num_shards()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteEvenShardingSpec requires %d devices, but received %d devices",
        num_shards(), devices->size()));
  }
  std::shared_ptr<const ConcreteEvenShardingSpec> spec =
      std::static_pointer_cast<const ConcreteEvenShardingSpec>(
          weak_from_this().lock());
  if (spec == nullptr) {
    spec = ConcreteEvenShardingSpec::Create(num_shards(), shape(),
                                            shard_shape(), IsFullyReplicated());
  }
  return std::unique_ptr<ConcreteEvenSharding>(new ConcreteEvenSharding(
      std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> ConcreteEvenShardingSpec::GetShardShape(
    const Shape& shape) const {
  if (shape != shape_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteEvenShardingSpec has a shard shape for shape %v, but was "
        "asked to get a shard shape for shape %v",
        shape_, shape));
  }
  return shard_shape_;
}

bool ConcreteEvenShardingSpec::HasSamePartitioning(
    const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  const auto* other_concrete_even_sharding_spec =
      dyn_cast<ConcreteEvenShardingSpec>(&other);
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
    return absl::InvalidArgumentError(absl::StrFormat(
        "ConcreteEvenShardingSpec can only disassemble shape %v, but was "
        "asked to disassemble shape %v",
        shape_, shape));
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
  return absl::InvalidArgumentError(absl::StrFormat(
      "ConcreteEvenShardingSpec can only disassemble static shape, but was "
      "asked to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::StatusOr<std::vector<IndexDomain>> ConcreteEvenShardingSpec::IndexDomains(
    const Shape& shape) const {
  return absl::InvalidArgumentError(
      "ConcreteEvenShardingSpec does not have index domain information");
}

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
ConcreteEvenShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  if (!IsFullyReplicated() || this->shape() != shard_shape() ||
      this->shape() != shape) {
    return absl::InvalidArgumentError(
        "ConcreteEvenShardingSpec does not have index domain information");
  }
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

absl::StatusOr<absl::Span<const int>>
ConcreteEvenShardingSpec::ShardToUniqueIndexDomainIndex() const {
  if (!IsFullyReplicated() || this->shape() != shard_shape()) {
    return absl::InvalidArgumentError(
        "ConcreteEvenShardingSpec does not have index domain information");
  }
  absl::call_once(shard_to_unique_index_domain_index_once_, [this] {
    cached_shard_to_unique_index_domain_index_.assign(num_shards_, 0);
  });
  return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
}

std::string ConcreteEvenShardingSpec::DebugString() const {
  return absl::StrFormat(
      "ConcreteEvenShardingSpec(num_shards: %d, shape: %v, "
      "shard_shape: %v, is_fully_replicated: %s)",
      num_shards_, shape_, shard_shape_,
      is_fully_replicated_ ? "true" : "false");
}

void ConcreteEvenShardingSpec::Hash(absl::HashState state) const {
  absl::HashState::combine(std::move(state), num_shards_, is_fully_replicated_,
                           shape_, shard_shape_);
}

}  // namespace ifrt
}  // namespace xla
