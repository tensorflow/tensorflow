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

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

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
class Array4D {
 public:
  // Creates a 4D array, uninitialized values.
  Array4D(int64 planes, int64 depth, int64 height, int64 width)
      : planes_(planes),
        depth_(depth),
        height_(height),
        width_(width),
        values_(new T[planes * depth * height * width]) {
    Fill(T());
  }

  // Creates a 4D array, initialized to value.
  Array4D(int64 planes, int64 depth, int64 height, int64 width, T value)
      : Array4D(planes, depth, height, width) {
    Fill(value);
  }

  // Creates a 4D array, filled with values.
  //
  // We need to set a default type for Container so that code like
  // Array4D(1, 1, 1, 1, {1}) will work. The template cannot infer the
  // initializer_list type in that case without this default.
  template <typename Container = std::initializer_list<T>>
  Array4D(int64 planes, int64 depth, int64 height, int64 width,
          const Container& values)
      : Array4D(planes, depth, height, width) {
    SetValues(values);
  }

  // Construct an Array4D with the given nested initializer list.
  Array4D(std::initializer_list<std::initializer_list<
              std::initializer_list<std::initializer_list<T>>>>
              values)
      : Array4D(values.size(), values.begin()->size(),
                values.begin()->begin()->size(),
                values.begin()->begin()->begin()->size()) {
    int64 plane = 0;
    for (const auto values_in_plane : values) {
      DCHECK_EQ(values_in_plane.size(), depth_);
      int64 depth = 0;
      for (const auto values_in_depth : values_in_plane) {
        DCHECK_EQ(values_in_depth.size(), height_);
        int64 height = 0;
        for (const auto values_in_height : values_in_depth) {
          DCHECK_EQ(values_in_height.size(), width_);
          int64 width = 0;
          for (const auto element_value : values_in_height) {
            (*this)(plane, depth, height, width) = element_value;
            ++width;
          }
          ++height;
        }
        ++depth;
      }
      ++plane;
    }
  }

  Array4D(const Array4D<T>& other)
      : Array4D(other.planes(), other.depth(), other.height(), other.width()) {
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
  }

  Array4D<T>& operator=(const Array4D<T>& other) {
    planes_ = other.planes();
    depth_ = other.depth();
    height_ = other.height();
    width_ = other.width();
    values_.reset(new T[num_elements()]);
    std::copy(&other.values_[0], &other.values_[0] + num_elements(),
              &values_[0]);
    return *this;
  }

  T& operator()(int64 plane, int64 depth, int64 height, int64 width) {
    CHECK_LT(plane, planes_);
    CHECK_LT(depth, depth_);
    CHECK_LT(height, height_);
    CHECK_LT(width, width_);
    return values_[plane * (depth_ * height_ * width_) +
                   depth * (height_ * width_) + height * (width_) + width];
  }
  const T& operator()(int64 plane, int64 depth, int64 height,
                      int64 width) const {
    return const_cast<Array4D*>(this)->operator()(plane, depth, height, width);
  }

  int64 width() const { return width_; }
  int64 height() const { return height_; }
  int64 depth() const { return depth_; }
  int64 planes() const { return planes_; }

  // Numerically-named aliases for the various dimensions. This matches the
  // dimension names used in array3d.
  int64 n4() const { return width_; }
  int64 n3() const { return height_; }
  int64 n2() const { return depth_; }
  int64 n1() const { return planes_; }
  int64 num_elements() const { return width_ * height_ * depth_ * planes_; }

  // Sets all the values in the array to values.
  template <typename Container = std::initializer_list<T>>
  void SetValues(const Container& container) {
    CHECK_EQ(std::distance(std::begin(container), std::end(container)),
             num_elements());
    std::copy(std::begin(container), std::end(container), &values_[0]);
  }

  // Fills the array with the given value.
  void Fill(const T& value) {
    std::fill(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with iota.
  void FillIota(const T& value) {
    std::iota(&values_[0], &values_[0] + num_elements(), value);
  }

  // Fills the array with random variable with a deviation of value and a mean
  // of mean.
  void FillRandom(const T& value, const double mean = 0.0,
                  const int seed = 12345) {
    std::mt19937 g(seed);
    std::normal_distribution<double> distribution(mean,
                                                  static_cast<double>(value));
    for (int64 i = 0; i < num_elements(); ++i) {
      values_[i] = static_cast<T>(distribution(g));
    }
  }

  // Fills values with the sequence i*multiplier for i=0,1,...
  void FillWithMultiples(float multiplier) {
    for (int64 i = 0; i < num_elements(); ++i) {
      values_[i] = i * multiplier;
    }
  }

  // Invokes a callback with the (indices, value_ptr) for each cell in the 4D
  // array.
  void Each(std::function<void(tensorflow::gtl::ArraySlice<int64>, T*)> f) {
    for (int64 plane = 0; plane < planes(); ++plane) {
      for (int64 depth = 0; depth < this->depth(); ++depth) {
        for (int64 height = 0; height < this->height(); ++height) {
          for (int64 width = 0; width < this->width(); ++width) {
            auto& value = (*this)(plane, depth, height, width);
            f({plane, depth, height, width}, &value);
          }
        }
      }
    }
  }

  // Fills all of the {p,z} with the array provided, which specifies {y,x}.
  void FillWithYX(const Array2D<T>& value) {
    CHECK_EQ(value.height(), height());
    CHECK_EQ(value.width(), width());
    for (int64 plane = 0; plane < planes(); ++plane) {
      for (int64 depth = 0; depth < this->depth(); ++depth) {
        for (int64 height = 0; height < this->height(); ++height) {
          for (int64 width = 0; width < this->width(); ++width) {
            (*this)(plane, depth, height, width) = value(height, width);
          }
        }
      }
    }
  }

  // Fills all of the {x,y} with the array provided, which specifies {p,z}.
  void FillWithPZ(const Array2D<T>& value) {
    CHECK_EQ(value.height(), planes());
    CHECK_EQ(value.width(), depth());
    for (int64 height = 0; height < this->height(); ++height) {
      for (int64 width = 0; width < this->width(); ++width) {
        for (int64 plane = 0; plane < planes(); ++plane) {
          for (int64 depth = 0; depth < this->depth(); ++depth) {
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
    for (int64 height = 0; height < this->height(); ++height) {
      for (int64 width = 0; width < this->width(); ++width) {
        for (int64 plane = 0; plane < planes(); ++plane) {
          for (int64 depth = 0; depth < this->depth(); ++depth) {
            float this_val = plane * this->depth() + depth;
            (*this)(plane, depth, height, width) = this_val;
          }
        }
      }
    }
  }

  // Returns a string representation of the 4D array suitable for debugging.
  string ToString() const {
    std::vector<string> pieces = {
        tensorflow::strings::Printf("p=%lld,z=%lld,y=%lld,x=%lld {\n", planes(),
                                    depth(), height(), width())};
    for (int64 plane = 0; plane < planes_; ++plane) {
      pieces.push_back("  {\n");
      for (int64 depth = 0; depth < depth_; ++depth) {
        pieces.push_back("    {\n");
        for (int64 height = 0; height < height_; ++height) {
          pieces.push_back("      {");
          for (int64 width = 0; width < width_; ++width) {
            pieces.push_back(tensorflow::strings::StrCat(
                (*this)(plane, depth, height, width), ", "));
          }
          pieces.push_back("},\n");
        }
        pieces.push_back("    },\n");
      }
      pieces.push_back("  },\n");
    }
    pieces.push_back("}");
    return tensorflow::str_util::Join(pieces, "");
  }

 private:
  int64 planes_;
  int64 depth_;
  int64 height_;
  int64 width_;
  std::unique_ptr<T[]> values_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_ARRAY4D_H_
