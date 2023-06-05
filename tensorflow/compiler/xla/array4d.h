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

#ifndef TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
#define TENSORFLOW_COMPILER_XLA_ARRAY4D_H_

#include <algorithm>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

// Simple 4D array structure, similar in form to Array2D, for use primarily in
// testing and describing to XLA APIs values in the 4D array structures used
// in convolutions.
//
// The data layout is, in order from major to minor:
//
//    First dimension: plane, batch, n1
//   Second dimension: depth, feature, z, n2
//    Third dimension: height, y, n3
//   Fourth dimension: width, x, n4
//
// These dimensions are referred to by various names, so that is why
// more than one name is given above. See operator() for the exact
// calculation of 1d indices from 4d indices.
template <typename T>
class Array4D : public Array<T> {
 public:
  Array4D() : Array<T>(std::vector<int64_t>{0, 0, 0, 0}) {}

  // Creates a 4D array, uninitialized values.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}) {}

  // Creates a 4D array, initialized to value.
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width, T value)
      : Array<T>(std::vector<int64_t>{planes, depth, height, width}, value) {}

  // Creates a 4D array, filled with values.
  //
  // We need to set a default type for Container so that code like
  // Array4D(1, 1, 1, 1, {1}) will work. The template cannot infer the
  // initializer_list type in that case without this default.
  template <typename Container = std::initializer_list<T>>
  Array4D(int64_t planes, int64_t depth, int64_t height, int64_t width,
          const Container& values)
      : Array4D(planes, depth, height, width) {
    this->SetValues(values);
  }

  // Construct an Array4D with the given nested initializer list.
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>
              values)
      : Array<T>(values) {}

  // Creates an array of a floating-point type (half, bfloat16, float,
  // or double) from the given nested initializer list of float values.
  template <typename T2, array_impl::overload_for_float<T, T2> = true>
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T2>>>>
              values)
      : Array<T>(values) {}

  // Numerically-named aliases for the various dimensions. This matches the
  // dimension names used in array3d.
  int64_t n4() const { return this->dim(3); }
  int64_t n3() const { return this->dim(2); }
  int64_t n2() const { return this->dim(1); }
  int64_t n1() const { return this->dim(0); }

  int64_t width() const { return this->dim(3); }
  int64_t height() const { return this->dim(2); }
  int64_t depth() const { return this->dim(1); }
  int64_t planes() const { return this->dim(0); }

  // Fills all of the {p,z} with the array provided, which specifies {y,x}.
  void FillWithYX(const Array2D<T>& value) {
    CHECK_EQ(value.height(), height());
    CHECK_EQ(value.width(), width());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(height, width);
          }
        }
      }
    }
  }

  // Fills all of the {p,x} with the array provided, which specifies {z,y}.
  void FillWithZY(const Array2D<T>& value) {
    CHECK_EQ(value.height(), depth());
    CHECK_EQ(value.width(), height());
    for (int64_t plane = 0; plane < planes(); ++plane) {
      for (int64_t depth = 0; depth < this->depth(); ++depth) {
        for (int64_t height = 0; height < this->height(); ++height) {
          for (int64_t width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(depth, height);
          }
        }
      }
    }
  }

  // Fills all of the {x,y} with the array provided, which specifies {p,z}.
  void FillWithPZ(const Array2D<T>& value) {
    CHECK_EQ(value.height(), planes());
    CHECK_EQ(value.width(), depth());
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            (*this)(plane, depth, height, width) = value(plane, depth);
          }
        }
      }
    }
  }

  // Fills each of the minor-dim matrices with a number designating which minor
  // dim matrix is enclosed by the shape.
  void FillWithMinorDimNum() {
    LOG(INFO) << "width: " << this->width();
    LOG(INFO) << "height: " << this->height();
    LOG(INFO) << "depth: " << this->depth();
    LOG(INFO) << "planes: " << this->planes();
    for (int64_t height = 0; height < this->height(); ++height) {
      for (int64_t width = 0; width < this->width(); ++width) {
        for (int64_t plane = 0; plane < planes(); ++plane) {
          for (int64_t depth = 0; depth < this->depth(); ++depth) {
            float this_val = plane * this->depth() + depth;
            (*this)(plane, depth, height, width) = this_val;
          }
        }
      }
    }
  }
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
