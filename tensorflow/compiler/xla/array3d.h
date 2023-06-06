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
#include <memory>
#include <numeric>
#include <random>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

// Simple 3D array structure.
template <typename T>
class Array3D : public Array<T> {
 public:
  Array3D() : Array<T>(std::vector<int64_t>{0, 0, 0}) {}

  // Creates an array of dimensions n1 x n2 x n3, uninitialized values.
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}) {}

  // Creates an array of dimensions n1 x n2 x n3, initialized to value.
  Array3D(const int64_t n1, const int64_t n2, const int64_t n3, const T value)
      : Array<T>(std::vector<int64_t>{n1, n2, n3}, value) {}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension, and so on.
  //
  // For example {{{1, 2}, {3, 4}, {5, 6}, {7, 8}},
  //              {{9, 10}, {11, 12}, {13, 14}, {15, 16}},
  //              {{17, 18}, {19, 20}, {21, 22}, {23, 24}}}
  // results in an array with n1=3, n2=4, n3=2.
  Array3D(std::initializer_list<std::initializer_list<std::initializer_list<T>>>
              values)
      : Array<T>(values) {}

  // Creates an array of a floating-point type (half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array3D(
      std::initializer_list<std::initializer_list<std::initializer_list<T2>>>
          values)
      : Array<T>(values) {}

  int64_t n1() const { return this->dim(0); }
  int64_t n2() const { return this->dim(1); }
  int64_t n3() const { return this->dim(2); }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY3D_H_
