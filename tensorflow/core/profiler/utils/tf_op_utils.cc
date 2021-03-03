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

#include <string>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/regexp.h"

namespace tensorflow {
namespace profiler {
namespace {

const absl::string_view kIterator = "Iterator";
const absl::string_view kSeparator = "::";
constexpr char kNameScopeSeparator = '/';

}  // namespace

const absl::string_view kUnknownOp = "";  // op types are non-empty strings
const absl::string_view kDatasetOp = "Dataset";
const absl::string_view kMemcpyHToDOp = "MemcpyHToD";
const absl::string_view kMemcpyDToHOp = "MemcpyDToH";

bool IsTfOpName(absl::string_view op_name) {
  // TODO(b/177602927): Confirm the naming convention with the TF team.
  static const LazyRE2 kTfOpNameRegEx = {"[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*"};
  return RE2::FullMatch(op_name, *kTfOpNameRegEx);
}

bool IsTfOpType(absl::string_view op_type) {
  static const LazyRE2 kTfOpTypeRegEx = {"[A-Z_][a-zA-Z0-9_]*"};
  return RE2::FullMatch(op_type, *kTfOpTypeRegEx);
}

bool IsJaxOpType(absl::string_view op_type) {
  static const LazyRE2 kJaxOpTypeRegEx = {"[a-z_][a-z0-9_]*"};
  return RE2::FullMatch(op_type, *kJaxOpTypeRegEx);
}

bool IsJaxOpNameAndType(absl::string_view op_name, absl::string_view op_type) {
  if (op_name.empty() || !IsJaxOpType(op_type)) return false;
  std::vector<absl::string_view> split_result =
      absl::StrSplit(op_name, kNameScopeSeparator);
  return absl::StrContains(split_result.back(), op_type);
}

TfOp ParseTfOpFullname(absl::string_view tf_op_fullname) {
  // TF Op names have the format "name:type".
  TfOp tf_op = {Category::kUnknown, tf_op_fullname, kUnknownOp};
  std::vector<absl::string_view> parts =
      absl::StrSplit(tf_op_fullname, absl::MaxSplits(':', 1));
  if (parts.size() != 2) {
    // GPU-related Ops that need to be tracked.
    if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYHToD")) {
      tf_op.category = Category::kMemcpyHToD;
      tf_op.type = kMemcpyHToDOp;
    } else if (absl::StartsWithIgnoreCase(tf_op_fullname, "MEMCPYDToH")) {
      tf_op.category = Category::kMemcpyDToH;
      tf_op.type = kMemcpyDToHOp;
    }
    // TODO(ckluk): Include the corresponding Ops on TPU.
  } else if (parts[0] == kIterator) {
    // Dataset Op names (e.g., Iterator::Batch::Map::TFRecord) do not follow the
    // format of TF Op names. But we still want to capture them for
    // input-pipeline analysis.
    tf_op.category = Category::kTfData;
    tf_op.type = kDatasetOp;
  } else if (IsTfOpType(parts[1]) && IsTfOpName(parts[0])) {
    tf_op = {Category::kTensorFlow, parts[0], parts[1]};
  } else if (IsJaxOpType(parts[1])) {
    tf_op = {Category::kJax, parts[0], parts[1]};
  } else if (parts[1].empty()) {
    tf_op.name = parts[0];  // remove trailing ':'
  }
  return tf_op;
}

std::vector<absl::string_view> ParseTfNameScopes(const TfOp& tf_op) {
  std::vector<absl::string_view> name_scopes =
      absl::StrSplit(tf_op.name, kNameScopeSeparator);
  // The last element is an op name not TF name scope.
  if (!name_scopes.empty()) name_scopes.pop_back();
  return name_scopes;
}

std::string TfOpEventName(const TfOp& tf_op) {
  std::string event_name;
  if (tf_op.category == Category::kUnknown) {
    // Some TraceMe names contain trailing whitespace, remove it.
    event_name = std::string(absl::StripTrailingAsciiWhitespace(tf_op.name));
  } else if (tf_op.category == Category::kTfData) {
    event_name = DatasetOpEventName(tf_op.name);
  } else {
    event_name = std::string(tf_op.type);
  }
  return event_name;
}

std::string TfOpEventName(absl::string_view tf_op_fullname) {
  return TfOpEventName(ParseTfOpFullname(tf_op_fullname));
}

std::string DatasetOpEventName(absl::string_view full_name) {
  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return absl::StrCat(kIterator, kSeparator, split_result.back());
}

std::string IteratorName(absl::string_view full_name) {
  std::vector<absl::string_view> split_result =
      absl::StrSplit(full_name, kSeparator);
  return std::string(split_result.back());
}

std::vector<absl::string_view> ParseTensorShapes(
    absl::string_view tensor_shapes) {
  absl::ConsumePrefix(&tensor_shapes, "(");
  absl::ConsumeSuffix(&tensor_shapes, ")");
  return absl::StrSplit(tensor_shapes, ';');
}

}  // namespace profiler
}  // namespace tensorflow
