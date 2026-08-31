/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/array_spec.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/abstract_array_spec.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

ArraySpec::ArraySpec(
    DType dtype, Shape shape, ShardingRef sharding,
    absl_nullable std::shared_ptr<const xla::PjRtLayout> layout)
    : dtype_(dtype),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      layout_(std::move(layout)) {}

absl::StatusOr<ArraySpec> ArraySpec::FromProto(Client* client,
                                               const ArraySpecProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for ArraySpec deserialization"));
  }

  ABSL_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(proto.dtype()));
  ABSL_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
  ABSL_ASSIGN_OR_RETURN(auto sharding,
                   Sharding::FromProto(client, proto.sharding()));
  std::shared_ptr<const xla::PjRtLayout> layout;
  if (proto.has_layout()) {
    ABSL_ASSIGN_OR_RETURN(layout, xla::PjRtLayout::Deserialize(proto.layout()));
  }
  return ArraySpec(dtype, std::move(shape), std::move(sharding),
                   std::move(layout));
}

absl::Status ArraySpec::ToProto(ArraySpecProto& proto,
                                SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for ArraySpec serialization"));
  }

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());
  dtype_.ToProto(*proto.mutable_dtype(), version);
  shape_.ToProto(*proto.mutable_shape(), version);
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_sharding(), sharding_->ToProto(version));
  if (layout_ != nullptr) {
    proto.set_layout(layout_->Serialize());
  }
  return absl::OkStatus();
}

absl::StatusOr<AbstractArraySpec> ArraySpec::ToAbstractArraySpec() const {
  return AbstractArraySpec::Create(dtype_, shape_, sharding_->sharding_spec(),
                                   sharding_->memory_kind(), layout_);
}

}  // namespace ifrt
}  // namespace xla
