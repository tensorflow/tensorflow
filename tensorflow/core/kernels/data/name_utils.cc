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
#include "tensorflow/core/kernels/data/name_utils.h"

#include "absl/strings/match.h"
#include "re2/re2.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data {
namespace name_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kDefaultDatasetDebugStringPrefix[] = "";

constexpr char kDataset[] = "Dataset";
constexpr char kOp[] = "Op";
constexpr char kVersion[] = "V";
constexpr char kVersionNumRegex[] = "V(\\d+)$";

string OpName(const string& dataset_type) {
  // For the dataset ops with different versions of kernels (e.g.
  // ParallelInterleaveDatasetOp), the version number needs to be added to the
  // end of op name.
  int version_num;
  if (RE2::PartialMatch(dataset_type, kVersionNumRegex, &version_num)) {
    string op_name = dataset_type;
    RE2::Replace(&op_name, kVersionNumRegex,
                 strings::StrCat(kDataset, kVersion, version_num));
    return op_name;
  } else {
    return strings::StrCat(dataset_type, kDataset);
  }
}

string ArgsToString(std::initializer_list<StringPiece> args) {
  return strings::StrCat("(", absl::StrJoin(args, ", "), ")");
}

string DatasetDebugString(const string& dataset_type,
                          const string& dataset_name_prefix,
                          std::initializer_list<StringPiece> args) {
  if (args.size() == 0) {
    return strings::StrCat(OpName(dataset_type), kOp, kDelimiter,
                           dataset_name_prefix, kDataset);
  }

  string debug_str;
  strings::StrAppend(&debug_str, OpName(dataset_type), kOp, ArgsToString(args));
  strings::StrAppend(&debug_str, kDelimiter, dataset_name_prefix, kDataset);
  return debug_str;
}

string IteratorPrefix(const string& dataset_type, const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, dataset_type);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
