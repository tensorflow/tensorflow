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

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

template <typename T>
class Array2D : public Array<T> {
 public:
  Array2D() : Array<T>(std::vector<int64>{0, 0}) {}

  Array2D(const int64 n1, const int64 n2)
      : Array<T>(std::vector<int64>{n1, n2}) {}

  Array2D(const int64 n1, const int64 n2, const T value)
      : Array<T>({n1, n2}, value) {}

  // Creates an array from the given nested initializer list. The outer
  // initializer list is the first dimension; the inner is the second dimension.
  // For example, {{1, 2, 3}, {4, 5, 6}} results in an array with n1=2 and n2=3.
  Array2D(std::initializer_list<std::initializer_list<T>> values)
      : Array<T>(values) {}

  Array2D(const Array2D<T>& other) : Array<T>(other) {}

  int64 n1() const { return this->dim(0); }
  int64 n2() const { return this->dim(1); }

  int64 height() const { return this->dim(0); }
  int64 width() const { return this->dim(1); }

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

  // Applies f to all cells in this array, in row-major order.
  void Each(std::function<void(int64, int64, T*)> f) {
    for (int64 i0 = 0; i0 < n1(); ++i0) {
      for (int64 i1 = 0; i1 < n2(); ++i1) {
        f(i0, i1, &(*this)(i0, i1));
      }
    }
  }
};

// Returns a linspace-populated Array2D in the range [from, to] (inclusive)
// with dimensions n1 x n2.
std::unique_ptr<Array2D<float>> MakeLinspaceArray2D(float from, float to,
                                                    int64 n1, int64 n2);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY2D_H_
