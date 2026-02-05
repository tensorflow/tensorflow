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

#include "xla/pjrt/pjrt_device_dimensions.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/proto/pjrt_device_dimensions.pb.h"

namespace xla {

PjRtDeviceDimensionsProto PjRtDeviceDimensions::ToProto() const {
  PjRtDeviceDimensionsProto proto;
  for (int32_t dim : dimensions_) {
    proto.add_dimensions(dim);
  }
  return proto;
}

std::string PjRtDeviceDimensions::ToString(absl::string_view sep) const {
  return absl::StrJoin(dimensions_, sep);
}

absl::StatusOr<PjRtDeviceDimensions> PjRtDeviceDimensions::FromString(
    absl::string_view text) {
  if (text.empty()) {
    return PjRtDeviceDimensions({});
  }
  std::vector<std::string> bounds_str = absl::StrSplit(text, ',');

  absl::InlinedVector<int32_t, 3> dims;
  for (auto const& b : bounds_str) {
    int32_t bound;
    if (!absl::SimpleAtoi(b, &bound)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Number parsing error for pjrt device dimensions %s "
                          "while parsing %s.",
                          text, b));
    }
    dims.push_back(bound);
  }

  return PjRtDeviceDimensions(dims);
}

bool AbslParseFlag(absl::string_view text, PjRtDeviceDimensions* bounds,
                   std::string* err) {
  const auto status_or_dimensions = PjRtDeviceDimensions::FromString(text);
  if (!status_or_dimensions.ok()) {
    *err = status_or_dimensions.status().ToString();
    return false;
  }
  *bounds = status_or_dimensions.value();
  return true;
}

std::string AbslUnparseFlag(PjRtDeviceDimensions bounds) {
  return bounds.ToString();
}

}  // namespace xla
