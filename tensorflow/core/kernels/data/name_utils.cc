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

string OpName(const string& dataset_type) {
  if (dataset_type == "Map") {
    return "MapDataset";
  } else if (dataset_type == "Prefetch") {
    return "PrefetchDataset";
  } else if (dataset_type == "Range") {
    return "RangeDataset";
  }
  LOG(FATAL) << "Unknown dataset type " << dataset_type << std::endl;
  return "Unknown dataset type";
}

namespace internal {

string DatasetDebugString(const string& dataset_type,
                          std::initializer_list<StringPiece> args) {
  if (args.size() == 0) {
    return strings::StrCat(dataset_type, "Op::Dataset");
  }

  string debug_str;
  strings::StrAppend(&debug_str, dataset_type, "Op(");
  auto iter = args.begin();
  while (iter != args.end() - 1) {
    strings::StrAppend(&debug_str, *iter, ", ");
    ++iter;
  }
  strings::StrAppend(&debug_str, *iter, ")::Dataset");
  return debug_str;
}

}  // namespace internal

string IteratorPrefix(const string& dataset_type, const string& prefix) {
  return strings::StrCat(prefix, "::", dataset_type);
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
