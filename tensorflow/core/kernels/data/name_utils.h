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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_NAME_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_NAME_UTILS_H_

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace name_utils {
using namespace tensorflow::strings;

enum DatasetType {
  RANGE_DATASET = 0,
  MAP_DATASET = 1,
};

// Returns the base name of the dataset.
string ToString(const DatasetType& dataset_type);

// Returns the dataset op name.
string OpName(const DatasetType& dataset_type);

string DatasetDebugString(const DatasetType& dataset_type,
                          std::initializer_list<StringPiece> args);

// Returns a human-readable debug string for this dataset in the format of
// "FooDatasetOp(arg1, arg2, ...)::Dataset"
template <typename... Args>
string DatasetDebugString(const DatasetType& dataset_type,
                          const Args&... args) {
  return DatasetDebugString(dataset_type,
                            {static_cast<const AlphaNum&>(args).Piece()...});
}

// Returns a string that identifies the sequence of iterators leading up to
// the iterator of this dataset.
string IteratorPrefix(const DatasetType& dataset_type, const string& prefix);

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_NAME_UTILS_H_
