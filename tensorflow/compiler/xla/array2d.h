/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY2D_H_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <random>
#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Simple 2D array structure.
//
// The data layout in major-to-minor order is: n1, n2.
template <typename T>
class Array2D {
 public:
  // Creates an empty array.
  Array2D() : n1_(0), n2_(0) {}

  // Creates an array of dimensions n1 x n2, uninitialized values.
  Array2D(const int64 n1, const int64 n2) : n1_(n1), n2_(n2) {
    values_.resize(n1 * n2);
  }

  // Creates an array of dimensions n1 x n2, initialized to value.
  Array2D(const int64 n1, const int64 n2, const T value) : Array2D(n1, n2) {
    Fill(value);
  }

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension; the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array2D(std::initializer_list<std::initializer_list<T>> values)
      : Array2D(values.size(), values.begin()->size()) {
    int64 n1 = 0;
    for (auto n1_it = values.begin(); n1_it != values.end(); ++n1_it, ++n1) {
      int64 n2 = 0;
      for (auto n2_it = n1_it->begin(); n2_it != n1_it->end(); ++n2_it, ++n2) {
        (*this)(n1, n2) = *n2_it;
      }
    }
  }

  T& operator()(const int64 n1, const int64 n2) {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    return values_[n1 * n2_ + n2];
  }

  const T& operator()(const int64 n1, const int64 n2) const {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    return values_[n1 * n2_ + n2];
  }

  // Access to the array's dimensions. height() and width() provide the
  // canonical interpretation of the array n1 x n2 having n1 rows of n2 columns
  // each (height is number of rows; width is number of columns).
  int64 n1() const { return n1_; }
  int64 n2() const { return n2_; }
  int64 height() const { return n1_; }
  int64 width() const { return n2_; }
  int64 num_elements() const { return values_.size(); }

  // Low-level accessor for stuff like memcmp, handle with care. Returns pointer
  // to the underlying storage of the array (similarly to std::vector::data()).
  T* data() const { return const_cast<Array2D*>(this)->values_.data(); }

  // Fills the array with the given value.
  void Fill(const T& value) {
    std::fill(values_.begin(), values_.end(), value);
  }

  // Applies f to all cells in this array, in row-major order.
  void Each(std::function<void(int64, int64, T*)> f) {
    for (int64 i0 = 0; i0 < n1(); ++i0) {
      for (int64 i1 = 0; i1 < n2(); ++i1) {
        f(i0, i1, &(*this)(i0, i1));
      }
    }
  }

  // Fills the array with a pattern of values of the form:
  //
  //    (rowno << log2ceil(width) | colno) + start_value
  //
  // This makes it easy to see distinct row/column values in the array.
  void FillUnique(T start_value = 0) {
    for (int64 i0 = 0; i0 < n1(); ++i0) {
      for (int64 i1 = 0; i1 < n2(); ++i1) {
        (*this)(i0, i1) =
            ((i0 << tensorflow::Log2Ceiling64(n2())) | i1) + start_value;
      }
    }
  }

  // Fills the array with random normal variables of deviation value.
  void FillRandom(const T& value, const double mean = 0.0,
                  const int seed = 12345) {
    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean,
                                                  static_cast<double>(value));
    for (auto& v : values_) {
      v = static_cast<T>(distribution(g));
    }
  }

  // Returns a readable string representation of the array.
  string ToString() const {
    std::vector<string> pieces = {"["};
    for (int64 row = 0; row < height(); ++row) {
      pieces.push_back("[");
      for (int64 col = 0; col < width(); ++col) {
        pieces.push_back(tensorflow::strings::StrCat((*this)(row, col)));
        pieces.push_back(", ");
      }
      pieces.pop_back();
      pieces.push_back("]");
      pieces.push_back(",\n ");
    }
    pieces.pop_back();
    pieces.push_back("]");
    return tensorflow::str_util::Join(pieces, "");
  }

 private:
  int64 n1_;
  int64 n2_;
  std::vector<T> values_;
};

// Returns a linspace-populated Array2D in the range [from, to] (inclusive)
// with dimensions n1 x n2.
std::unique_ptr<Array2D<float>> MakeLinspaceArray2D(float from, float to,
                                                    int64 n1, int64 n2);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
