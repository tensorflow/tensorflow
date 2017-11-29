// Copyright 2017 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "tensorflow/contrib/boosted_trees/lib/utils/optional_value.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {

// A matrix that given feature column id and feature value id will return
// either a value or an optional. First index indicates feature column, second
// index - the index of the value within this column - for single valued, it
// will be 0.
// Allows double-subscript access [][].
template <class T>
class SparseMatrix {
  typedef std::vector<std::tuple<int32, int32, T>> SparseMap;

  class Proxy {
   public:
    Proxy(const int32 feature_column_idx, const SparseMap& values)
        : feature_column_idx_(feature_column_idx), values_(values) {}

    OptionalValue<T> operator[](int feature_idx) const {
      auto value_iter = std::find_if(
          values_.begin(), values_.end(),
          [this, &feature_idx](const std::tuple<int32, int32, T>& element) {
            return std::get<0>(element) == feature_column_idx_ &&
                   std::get<1>(element) == feature_idx;
          });

      if (value_iter == values_.end()) {
        return OptionalValue<T>();
      }
      // There is this feature column and feature id.
      return OptionalValue<T>(std::get<2>(*value_iter));
    }

   private:
    int32 feature_column_idx_;
    const SparseMap& values_;
  };

 public:
  void addElement(const int32 feature_column_idx, const int32 feature_idx,
                  const T value) {
    values_.emplace_back(feature_column_idx, feature_idx, value);
  }

  void clear() { values_.clear(); }

  Proxy operator[](int feature_column_idx) const {
    return Proxy(feature_column_idx, values_);
  }

 private:
  SparseMap values_;
};

// Holds data for one example and enables lookup by feature column.
struct Example {
  // Default constructor creates an empty example.
  Example() : example_idx(-1) {}

  // Example index.
  int64 example_idx;

  // Dense and sparse float features indexed by feature column.
  // TODO(salehay): figure out a design to support multivalent float features.
  std::vector<float> dense_float_features;
  // Sparse float features are allowed to be multivalent and thus can be
  // represented as a sparse matrix.
  SparseMatrix<float> sparse_float_features;

  // Sparse integer features indexed by feature column.
  // Note that all integer features are assumed to be categorical, i.e. will
  // never be compared by order. Also these features can be multivalent.
  std::vector<std::unordered_set<int64>> sparse_int_features;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
