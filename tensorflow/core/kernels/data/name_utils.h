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

#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace name_utils {

extern const char kDelimiter[];
extern const char kDefaultDatasetDebugStringPrefix[];

// Merge the given args in the format of "(arg1, arg2, ..., argn)".
//
// e.g. ArgsToString({"1", "2", "3"}) -> "(1, 2, 3)".
string ArgsToString(std::initializer_list<StringPiece> args);

// Returns the dataset op name.
//
// e.g. OpName("Map") -> "MapDataset".
string OpName(const string& dataset_type);

// Returns a human-readable debug string for this dataset in the format of
// "FooDatasetOp(arg1, arg2, ...)::Dataset".
//
// e.g. DatasetDebugString("Map", "", {}) -> "MapDatasetOp::Dataset";
// DatasetDebugString("Range", "", {"0", "10", "3"}) ->
// "RangeDatasetOp(0, 10, 3)::Dataset";
// DatasetDebugString("Shuffle", "FixedSeed", {"10", "1", "2"}) ->
// "ShuffleDatasetOp(10, 1, 2)::FixedSeedDataset";
string DatasetDebugString(const string& dataset_type,
                          const string& dataset_name_prefix,
                          std::initializer_list<StringPiece> args);

// Returns a human-readable debug string for this dataset in the format of
// "FooDatasetOp(arg1, arg2, ...)::Dataset".
//
// e.g. DatasetDebugString("Map") -> "MapDatasetOp::Dataset";
// DatasetDebugString("Range", 0, 10, 3) -> "RangeDatasetOp(0, 10, 3)::Dataset".
template <typename... Args>
string DatasetDebugString(const string& dataset_type, const Args&... args) {
  return DatasetDebugString(
      dataset_type, kDefaultDatasetDebugStringPrefix,
      {static_cast<const strings::AlphaNum&>(args).Piece()...});
}

// Returns a string that identifies the sequence of iterators leading up to
// the iterator of this dataset.
//
// e.g. IteratorPrefix("Map", "Iterator::range") -> "Iterator::Range::Map".
string IteratorPrefix(const string& dataset_type, const string& prefix);

}  // namespace name_utils
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_NAME_UTILS_H_
