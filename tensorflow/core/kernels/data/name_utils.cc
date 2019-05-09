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

namespace tensorflow {
namespace data {
namespace name_utils {

string ToString(const DatasetType& dataset_type) {
  switch (dataset_type) {
    case RANGE_DATASET:
      return "range";
    case MAP_DATASET:
      return "map";
    default:
      LOG(FATAL) << "Unknown dataset type " << dataset_type << std::endl;
  }
  return "Unknown dataset type";
}

string OpName(const DatasetType& dataset_type) {
  switch (dataset_type) {
    case RANGE_DATASET:
      return "RangeDataset";
    case MAP_DATASET:
      return "MapDataset";
    default:
      LOG(FATAL) << "Unknown dataset type " << dataset_type << std::endl;
  }
  return "Unknown dataset type";
}

string DatasetDebugString(const DatasetType& dataset_type,
                          std::initializer_list<StringPiece> args) {
  if (args.size() == 0) {
    return StrCat(OpName(dataset_type), "Op::Dataset");
  }

  string debug_str;
  StrAppend(&debug_str, OpName(dataset_type), "Op(");
  auto iter = args.begin();
  while (iter != args.end() - 1) {
    StrAppend(&debug_str, *iter, ", ");
    ++iter;
  }
  StrAppend(&debug_str, *iter, ")::Dataset");
  return debug_str;
}

string IteratorPrefix(const DatasetType& dataset_type, const string& prefix) {
  return strings::StrCat(prefix, "::", ToString(dataset_type));
}

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow
