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

#include "xla/python/ifrt/index.h"

#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/index.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

absl::StatusOr<Index> Index::FromProto(const IndexProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for Index deserialization"));
  }

  Index::Elements elements;
  elements.reserve(proto.elements_size());
  for (int64_t element : proto.elements()) {
    if (element < 0) {
      return InvalidArgument(
          "Index expects non-negative element values, but got %d", element);
    }
    elements.push_back(element);
  }
  return Index(std::move(elements));
}

void Index::ToProto(IndexProto& proto, SerDesVersion version) const {
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for Index serialization";

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_elements()->Reserve(elements().size());
  for (int64_t element : elements()) {
    proto.mutable_elements()->AddAlreadyReserved(element);
  }
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
  return os << absl::StrCat(index);
}

}  // namespace ifrt
}  // namespace xla
