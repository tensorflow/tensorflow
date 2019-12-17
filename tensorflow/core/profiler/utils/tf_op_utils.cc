/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/tf_op_utils.h"

#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr absl::string_view kIterator = "Iterator";
constexpr absl::string_view kSeparator = "::";

}  // namespace

const absl::string_view kUnknownOp = "";  // op types are non-empty strings
const absl::string_view kDatasetOp = "Dataset";

TfOp ParseTfOpFullname(absl::string_view tf_op_fullname) {
  // TF Op names have the format "name:type" where:
  // - name is a NodeDef.name and must match:
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_./]*"};
  // - if type starts with underscore it is internal to TensorFlow.
  // - type is an OpDef.name, must be CamelCase and match:
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};

  // JAX op types have only lowercase letters and underscores.
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_]*"};

  TfOp tf_op = {tf_op_fullname, kUnknownOp};
  std::vector<absl::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));
  if (parts.size() != 2) {
  } else if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.type = kDatasetOp;
  } else if (RE2::FullMatch(parts[1], *kTfOpTypeRegEx) &&
             RE2::FullMatch(parts[0], *kTfOpNameRegEx)) {  // TensorFlow
    tf_op = {parts[0], parts[1]};
  } else if (absl::StrContains(parts[0], " = ") &&
             RE2::FullMatch(parts[1], *kJaxOpTypeRegEx)) {  // JAX
    tf_op = {parts[0], parts[1]};
  }
  return tf_op;
}

std::string TfOpEventName(absl::string_view tf_op_fullname) {
  std::string event_name;
  TfOp op = ParseTfOpFullname(tf_op_fullname);
  if (op.type == kUnknownOp) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name =
        std::string(absl::StripTrailingAsciiWhitespace(tf_op_fullname));
  } else if (op.type == kDatasetOp) {
    std::vector<absl::string_view> op_parts =
        absl::StrSplit(tf_op_fullname, kSeparator);
    event_name = absl::StrCat(kIterator, kSeparator, op_parts.back());
  } else {
    event_name = std::string(op.type);
  }
  return event_name;
}

}  // namespace profiler
}  // namespace tensorflow
