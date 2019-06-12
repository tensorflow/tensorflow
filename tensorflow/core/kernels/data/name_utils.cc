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

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace data {
namespace name_utils {

ABSL_CONST_INIT const char kDelimiter[] = "::";
ABSL_CONST_INIT const char kDefaultDatasetDebugStringPrefix[] = "";

constexpr char kDefaultDatasetName[] = "Dataset";

string OpName(const string& dataset_type) {
  return strings::StrCat(dataset_type, kDefaultDatasetName);
}

string DatasetDebugString(const string& dataset_type,
                          const string& dataset_name_prefix,
                          std::initializer_list<StringPiece> args) {
  if (args.size() == 0) {
    return strings::StrCat(OpName(dataset_type), "Op", kDelimiter,
                           dataset_name_prefix, kDefaultDatasetName);
  }

  string debug_str;
  strings::StrAppend(&debug_str, OpName(dataset_type), "Op(",
                     absl::StrJoin(args, ", "), ")");
  strings::StrAppend(&debug_str, kDelimiter, dataset_name_prefix,
                     kDefaultDatasetName);
  return debug_str;
}

string IteratorPrefix(const string& dataset_type, const string& prefix) {
  return strings::StrCat(prefix, kDelimiter, dataset_type);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
