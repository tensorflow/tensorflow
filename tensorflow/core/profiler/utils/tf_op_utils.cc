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

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kIterator = "Iterator";
const absl::string_view kSeparator = "::";

}  // namespace

const absl::string_view kUnknownOp = "";  // op types are non-empty strings
const absl::string_view kDatasetOp = "Dataset";
const absl::string_view kMemcpyHToDOp = "MemcpyHToD";
const absl::string_view kMemcpyDToHOp = "MemcpyDToH";

TfOp ParseTfOpFullname(absl::string_view tf_op_fullname) {
  // TF Op names have the format "name:type" where:
  // - name is a NodeDef.name and must match:
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_./]*"};
  // - if type starts with underscore it is internal to TensorFlow.
  // - type is an OpDef.name, must be CamelCase and match:
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};

  // JAX op types have only lowercase letters and underscores.
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_]*"};

  TfOp tf_op = {tf_op_fullname, kUnknownOp, /*is_tf_op=*/false};
  std::vector<absl::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));
  if (parts.size() != 2) {
    // GPU-related Ops that need to be tracked.
    if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
      tf_op.type = kMemcpyHToDOp;
    } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
      tf_op.type = kMemcpyDToHOp;
    }
    // TODO(ckluk): Include the corresponding Ops on TPU.
  } else if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.type = kDatasetOp;
  } else if (RE2::FullMatch(parts[1], *kTfOpTypeRegEx) &&
             RE2::FullMatch(parts[0], *kTfOpNameRegEx)) {  // TensorFlow
    tf_op = {parts[0], parts[1], /*is_tf_op=*/true};
  } else if (RE2::FullMatch(parts[1], *kJaxOpTypeRegEx)) {  // JAX
    tf_op = {parts[0], parts[1], /*is_tf_op=*/false};
  }
  return tf_op;
}

std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  std::vector<absl::string_view> name_scopes = absl::StrSplit(tf_op.name, '/');
  // The last element is an op name not TF name scope.
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}

std::string TfOpEventName(const TfOp& tf_op) {
  std::string event_name;
  if (tf_op.type == kUnknownOp) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name = std::string(absl::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.type == kDatasetOp) {
    std::vector<absl::string_view> op_parts =
        absl::StrSplit(tf_op.name, kSeparator);
    event_name = absl::StrCat(kIterator, kSeparator, op_parts.back());
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}

std::string TfOpEventName(absl::string_view tf_op_fullname) {
  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}

}  // namespace profiler
}  // namespace tensorflow
