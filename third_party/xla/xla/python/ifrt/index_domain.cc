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

#include "xla/python/ifrt/index_domain.h"

#include <ostream>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/python/ifrt/index_domain.pb.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

absl::StatusOr<IndexDomain> IndexDomain::FromProto(
    const IndexDomainProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for IndexDomain deserialization"));
  }

  ASSIGN_OR_RETURN(Index origin_val, Index::FromProto(proto.origin()));
  ASSIGN_OR_RETURN(Shape shape_val, Shape::FromProto(proto.shape()));

  return IndexDomain(std::move(origin_val), std::move(shape_val));
}

void IndexDomain::ToProto(IndexDomainProto& proto,
                          SerDesVersion version) const {
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for IndexDomain serialization";

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  origin_.ToProto(*proto.mutable_origin(), version);
  shape_.ToProto(*proto.mutable_shape(), version);
}

std::ostream& operator<<(std::ostream& os, const IndexDomain& index_domain) {
  return os << absl::StrCat(index_domain);
}

}  // namespace ifrt
}  // namespace xla
