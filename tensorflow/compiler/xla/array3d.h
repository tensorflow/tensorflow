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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY3D_H_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Simple 3D array structure.
//
// The data layout in major-to-minor order is: n1, n2, n3.
template <typename T>
class Array3D {
 public:
  // Creates an array of dimensions n1 x n2 x n3, uninitialized values.
  Array3D(const int64 n1, const int64 n2, const int64 n3)
      : n1_(n1), n2_(n2), n3_(n3) {
    values_.resize(n1 * n2 * n3);
  }

  // Creates an array of dimensions n1 x n2 x n3, initialized to value.
  Array3D(const int64 n1, const int64 n2, const int64 n3, const T value)
      : Array3D(n1, n2, n3) {
    Fill(value);
  }

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  //
  // For example {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
  //              {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
  //              {{17, 18}, {19, 20}, {21, 22}, {23, 24}}}
  // results in an array with n1=3, n2=4, n3=2.
  Array3D(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
              values)
      : Array3D(values.size(), values.begin()->size(),
                values.begin()->begin()->size()) {
    int64 n1 = 0;
    for (auto n1_it = values.begin(); n1_it != values.end(); ++n1_it, ++n1) {
      int64 n2 = 0;
      for (auto n2_it = n1_it->begin(); n2_it != n1_it->end(); ++n2_it, ++n2) {
        int64 n3 = 0;
        for (auto n3_it = n2_it->begin(); n3_it != n2_it->end();
             ++n3_it, ++n3) {
          (*this)(n1, n2, n3) = *n3_it;
        }
      }
    }
  }

  T& operator()(const int64 n1, const int64 n2, const int64 n3) {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    CHECK_LT(n3, n3_);
    return values_[n1 * n2_ * n3_ + n2 * n3_ + n3];
  }

  const T& operator()(const int64 n1, const int64 n2, const int64 n3) const {
    CHECK_LT(n1, n1_);
    CHECK_LT(n2, n2_);
    CHECK_LT(n3, n3_);
    return values_[n1 * n2_ * n3_ + n2 * n3_ + n3];
  }

  // Access to the array's dimensions.
  int64 n1() const { return n1_; }
  int64 n2() const { return n2_; }
  int64 n3() const { return n3_; }
  int64 num_elements() const { return values_.size(); }

  // Fills the array with the given value.
  void Fill(const T& value) {
    std::fill(values_.begin(), values_.end(), value);
  }

  // Fills the array with sequentially increasing values.
  void FillIota(const T& value) {
    std::iota(values_.begin(), values_.end(), value);
  }

  // Fills the array with random normal values with a mean of 0 and standard
  // deviation of value.
  void FillRandom(const T& value, const double mean = 0.0,
                  const int seed = 12345) {
    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean,
                                                  static_cast<double>(value));
    for (auto& v : values_) {
      v = static_cast<T>(distribution(g));
    }
  }

 private:
  int64 n1_;
  int64 n2_;
  int64 n3_;
  std::vector<T> values_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
