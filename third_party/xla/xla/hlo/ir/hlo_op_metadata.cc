/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/ir/hlo_op_metadata.h"

#include <string>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/xla_data.pb.h"

namespace xla {

std::string OpMetadataToString(const OpMetadata& metadata, bool only_op_name) {
  std::vector<std::string> result;
  if (only_op_name) {
    if (!metadata.op_name().empty()) {
      return absl::StrCat("op_name=\"", absl::CEscape(metadata.op_name()),
                          "\"");
    }
    return "";
  }
  if (!metadata.op_type().empty()) {
    result.push_back(
        absl::StrCat("op_type=\"", absl::CEscape(metadata.op_type()), "\""));
  }
  if (!metadata.op_name().empty()) {
    result.push_back(
        absl::StrCat("op_name=\"", absl::CEscape(metadata.op_name()), "\""));
  }
  if (!metadata.source_file().empty()) {
    result.push_back(absl::StrCat("source_file=\"",
                                  absl::CEscape(metadata.source_file()), "\""));
  }
  if (metadata.source_line() != 0) {
    result.push_back(absl::StrCat("source_line=", metadata.source_line()));
  }
  if (!metadata.profile_type().empty()) {
    result.push_back(absl::StrCat(
        "profile_type={", absl::StrJoin(metadata.profile_type(), ","), "}"));
  }
  if (!metadata.deduplicated_name().empty()) {
    result.push_back(absl::StrCat("deduplicated_name=\"",
                                  absl::CEscape(metadata.deduplicated_name()),
                                  "\""));
  }
  if (!metadata.scheduling_name().empty()) {
    result.push_back(
        absl::StrCat("scheduling_name=\"", metadata.scheduling_name(), "\""));
  }
  return absl::StrJoin(result, " ");
}

}  // namespace xla
