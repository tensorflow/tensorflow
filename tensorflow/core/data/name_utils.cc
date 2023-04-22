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
#include "tensorflow/core/data/name_utils.h"

#include "absl/strings/str_join.h"

namespace tensorflow {
namespace data {
namespace name_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kDefaultDatasetDebugStringPrefix[] = "";

constexpr char kDataset[] = "Dataset";
constexpr char kOp[] = "Op";
constexpr char kVersion[] = "V";

string OpName(const string& dataset_type) {
  return OpName(dataset_type, OpNameParams());
}

string OpName(const string& dataset_type, const OpNameParams& params) {
  if (params.op_version == 1) {
    return strings::StrCat(dataset_type, kDataset);
  }
  return strings::StrCat(dataset_type, kDataset, kVersion, params.op_version);
}

string ArgsToString(const std::vector<string>& args) {
  if (args.empty()) {
    return "";
  }
  return strings::StrCat("(", absl::StrJoin(args, ", "), ")");
}

string DatasetDebugString(const string& dataset_type) {
  return DatasetDebugString(dataset_type, DatasetDebugStringParams());
}

string DatasetDebugString(const string& dataset_type,
                          const DatasetDebugStringParams& params) {
  OpNameParams op_name_params;
  op_name_params.op_version = params.op_version;
  string op_name = OpName(dataset_type, op_name_params);
  return strings::StrCat(op_name, kOp, ArgsToString(params.args), kDelimiter,
                         params.dataset_prefix, kDataset);
}

string IteratorPrefix(const string& dataset_type, const string& prefix) {
  return IteratorPrefix(dataset_type, prefix, IteratorPrefixParams());
}

string IteratorPrefix(const string& dataset_type, const string& prefix,
                      const IteratorPrefixParams& params) {
  if (params.op_version == 1) {
    return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                           dataset_type);
  }
  return strings::StrCat(prefix, kDelimiter, params.dataset_prefix,
                         dataset_type, kVersion, params.op_version);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
