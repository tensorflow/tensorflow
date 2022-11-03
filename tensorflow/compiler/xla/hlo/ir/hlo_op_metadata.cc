/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/hlo/ir/hlo_op_metadata.h"

#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {

std::string OpMetadataToString(const OpMetadata& metadata) {
  std::vector<std::string> result;
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
  return absl::StrJoin(result, " ");
}

}  // namespace xla
