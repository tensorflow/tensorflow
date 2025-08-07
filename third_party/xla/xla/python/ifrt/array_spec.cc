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
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

absl::StatusOr<ArraySpec> ArraySpec::FromProto(Client* client,
                                               const ArraySpecProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for ArraySpec deserialization"));
  }

  TF_ASSIGN_OR_RETURN(auto dtype, DType::FromProto(proto.dtype()));
  TF_ASSIGN_OR_RETURN(auto shape, Shape::FromProto(proto.shape()));
  TF_ASSIGN_OR_RETURN(auto sharding,
                      Sharding::FromProto(client, proto.sharding()));
  std::shared_ptr<const xla::PjRtLayout> layout;
  if (proto.has_layout()) {
    TF_ASSIGN_OR_RETURN(layout, xla::PjRtLayout::Deserialize(proto.layout()));
  }
  return ArraySpec{
      /*dtype=*/dtype,
      /*shape=*/std::move(shape),
      /*sharding=*/std::move(sharding),
      /*layout=*/std::move(layout),
  };
}

absl::StatusOr<ArraySpecProto> ArraySpec::ToProto(SerDesVersion version) const {
  if (version.version_number() < SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unsupported ", version.version_number(),
                     " for ArraySpec serialization"));
  }

  ArraySpecProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());
  *proto.mutable_dtype() = dtype.ToProto(version);
  *proto.mutable_shape() = shape.ToProto(version);
  TF_ASSIGN_OR_RETURN(*proto.mutable_sharding(), sharding->ToProto(version));
  if (layout != nullptr) {
    proto.set_layout(layout->Serialize());
  }
  return proto;
}

std::string ArraySpec::DebugString() const {
  return absl::StrCat(
      "ArraySpec(dtype=", dtype.DebugString(), ",shape=", shape.DebugString(),
      ",sharding=", sharding->DebugString(),
      ",layout=", (layout != nullptr ? layout->ToString() : "<nullptr>"), ")");
}

}  // namespace ifrt
}  // namespace xla
