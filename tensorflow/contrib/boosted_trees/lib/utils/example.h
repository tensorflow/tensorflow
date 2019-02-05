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

#ifndef TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
#define TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_

#include <algorithm>
#include <unordered_set>
#include <vector>
#include "tensorflow/contrib/boosted_trees/lib/utils/optional_value.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {
namespace boosted_trees {
namespace utils {
// Represents sparse vector that have a value for some feature indices within
// the feature column.
// Allows subscript access [].
template <class T>
class SparseMultidimensionalValues {
 public:
  void Add(const int32 feature_idx, const T value) {
    values_.emplace_back(feature_idx, value);
  }

  void Clear() { values_.clear(); }

  void Reserve(const int32 size) { values_.reserve(size); }

  OptionalValue<T> operator[](int feature_idx) const {
    auto value_iter =
        std::find_if(values_.begin(), values_.end(),
                     [&feature_idx](const std::pair<int32, T>& element) {
                       return element.first == feature_idx;
                     });

    if (value_iter == values_.end()) {
      return OptionalValue<T>();
    }
    return OptionalValue<T>(value_iter->second);
  }

 private:
  std::vector<std::pair<int32, T>> values_;
};

// Represents storage for a sparse float feature column. Can store values either
// for one dimensional or a multivalent (multidimensional) sparse column.
// Allows subscript operator access [feature_id].
template <class T>
class SparseFloatFeatureColumn {
 public:
  void Reserve(const int32 size) {
    if (!single_dimensional_) {
      multidimensional_values.Reserve(size);
    }
  }

  void SetDimension(const int32 dimension) {
    single_dimensional_ = dimension <= 1;
  }

  void Add(const int32 feature_idx, const float value) {
    if (single_dimensional_) {
      DCHECK_EQ(0, feature_idx);
      single_value_ = value;
    } else {
      multidimensional_values.Add(feature_idx, value);
    }
    initialized_ = true;
  }

  void Clear() {
    single_dimensional_ = false;
    initialized_ = false;
    multidimensional_values.Clear();
  }

  OptionalValue<T> operator[](int feature_idx) const {
    if (!initialized_) {
      return OptionalValue<T>();
    }
    if (single_dimensional_) {
      return OptionalValue<T>(single_value_);
    } else {
      return multidimensional_values[feature_idx];
    }
  }

 private:
  bool single_dimensional_;
  bool initialized_;
  T single_value_;
  SparseMultidimensionalValues<T> multidimensional_values;
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

  // Sparse float features columns (can be either single or multivalent
  // (multidimensional).
  std::vector<SparseFloatFeatureColumn<float>> sparse_float_features;

  // Sparse integer features indexed by feature column.
  // Note that all integer features are assumed to be categorical, i.e. will
  // never be compared by order. Also these features can be multivalent.
  // By default we allocate a InlinedVector of length 1 though since that is
  // the most common case.
  std::vector<gtl::InlinedVector<int64, 1>> sparse_int_features;
};

}  // namespace utils
}  // namespace boosted_trees
}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_BOOSTED_TREES_LIB_UTILS_EXAMPLE_H_
