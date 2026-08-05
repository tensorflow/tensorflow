/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/abstract_array_spec.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/abstract_array_spec.pb.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/sharding_spec.h"

namespace xla {
namespace ifrt {

AbstractArraySpec::Rep::Rep(
    DType dtype, Shape shape, ShardingSpecRef sharding_spec,
    MemoryKind memory_kind,
    absl_nullable std::shared_ptr<const xla::PjRtLayout> layout)
    : dtype(dtype),
      shape(std::move(shape)),
      sharding_spec(std::move(sharding_spec)),
      memory_kind(memory_kind),
      layout(std::move(layout)) {}

AbstractArraySpec::AbstractArraySpec(
    DType dtype, Shape shape, ShardingSpecRef sharding_spec,
    MemoryKind memory_kind,
    absl_nullable std::shared_ptr<const xla::PjRtLayout> layout)
    : rep_(std::make_shared<const Rep>(
          dtype, std::move(shape), std::move(sharding_spec),
          std::move(memory_kind), std::move(layout))) {}

absl::StatusOr<AbstractArraySpec> AbstractArraySpec::Create(
    DType dtype, Shape shape, ShardingSpecRef sharding_spec,
    MemoryKind memory_kind,
    absl_nullable std::shared_ptr<const xla::PjRtLayout> layout) {
  // `ShardingSpecRef` is non-null. This check ensures that any invalid nullptr
  // that has not been caught at compile time is caught at runtime.
  if (sharding_spec == nullptr) {
    return absl::InvalidArgumentError("sharding_spec must not be null");
  }
  return AbstractArraySpec(dtype, std::move(shape), std::move(sharding_spec),
                           std::move(memory_kind), std::move(layout));
}

bool AbstractArraySpec::operator==(const AbstractArraySpec& other) const {
  if (rep_ == other.rep_) {
    return true;
  }
  if (Hash() != other.Hash()) {
    return false;
  }
  auto are_layouts_equal =
      [](const absl_nullable std::shared_ptr<const xla::PjRtLayout>& lhs,
         const absl_nullable std::shared_ptr<const xla::PjRtLayout>& rhs) {
        if (lhs == nullptr || rhs == nullptr) {
          return lhs == nullptr && rhs == nullptr;
        }
        return lhs == rhs || *lhs == *rhs;
      };
  return dtype() == other.dtype() && shape() == other.shape() &&
         *sharding_spec() == *other.sharding_spec() &&
         memory_kind() == other.memory_kind() &&
         are_layouts_equal(layout(), other.layout());
}

uint64_t AbstractArraySpec::Hash() const {
  uint64_t hash = rep_->hash.load(std::memory_order_relaxed);
  if (hash == Rep::kUnsetHash) {
    if (rep_->layout != nullptr) {
      hash = absl::HashOf(rep_->dtype, rep_->shape, *rep_->sharding_spec,
                          rep_->memory_kind, *rep_->layout);
    } else {
      hash = absl::HashOf(rep_->dtype, rep_->shape, *rep_->sharding_spec,
                          rep_->memory_kind);
    }
    if (ABSL_PREDICT_FALSE(hash == Rep::kUnsetHash)) {
      ++hash;
    }
    rep_->hash.store(hash, std::memory_order_relaxed);
  }
  return hash;
}

absl::StatusOr<ArraySpec> AbstractArraySpec::ToArraySpec(
    DeviceListRef devices) const {
  ASSIGN_OR_RETURN(
      ShardingRef sharding,
      rep_->sharding_spec->ToSharding(std::move(devices), rep_->memory_kind));
  return ArraySpec{
      /*dtype=*/rep_->dtype,
      /*shape=*/rep_->shape,
      /*sharding=*/std::move(sharding),
      /*layout=*/rep_->layout,
  };
}

absl::StatusOr<AbstractArraySpec> AbstractArraySpec::FromProto(
    const AbstractArraySpecProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(5)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version_number,
                     " for AbstractArraySpec deserialization"));
  }

  ASSIGN_OR_RETURN(DType dtype, DType::FromProto(proto.dtype()));
  ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  ASSIGN_OR_RETURN(ShardingSpecRef sharding_spec,
                   ShardingSpec::FromProto(proto.sharding_spec()));
  MemoryKind memory_kind;
  if (!proto.memory_kind().empty()) {
    memory_kind = MemoryKind(proto.memory_kind());
  }
  absl_nullable std::shared_ptr<const xla::PjRtLayout> layout;
  if (proto.has_layout()) {
    ASSIGN_OR_RETURN(layout, xla::PjRtLayout::Deserialize(proto.layout()));
  }
  return AbstractArraySpec::Create(dtype, std::move(shape),
                                   std::move(sharding_spec), memory_kind,
                                   std::move(layout));
}

absl::Status AbstractArraySpec::ToProto(AbstractArraySpecProto& proto,
                                        SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(5)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for AbstractArraySpec serialization"));
  }

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(5).value());
  dtype().ToProto(*proto.mutable_dtype(), version);
  shape().ToProto(*proto.mutable_shape(), version);
  ASSIGN_OR_RETURN(*proto.mutable_sharding_spec(),
                   sharding_spec()->ToProto(version));
  if (memory_kind().memory_kind().has_value()) {
    // NOLINTNEXTLINE(*-readability-redundant-string-conversions)
    proto.set_memory_kind(std::string(*memory_kind().memory_kind()));
  }
  if (layout() != nullptr) {
    proto.set_layout(layout()->Serialize());
  }
  return absl::OkStatus();
}

std::string AbstractArraySpec::DebugString() const {
  return absl::StrCat(
      "AbstractArraySpec(dtype=", rep_->dtype, ", shape=", rep_->shape,
      ", sharding_spec=", absl::StrFormat("%v", *rep_->sharding_spec),
      ", memory_kind=", rep_->memory_kind, ", layout=",
      (rep_->layout != nullptr ? rep_->layout->ToString() : "<nullptr>"), ")");
}

}  // namespace ifrt
}  // namespace xla
