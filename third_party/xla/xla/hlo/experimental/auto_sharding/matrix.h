/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_MATRIX_H_
#define XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_MATRIX_H_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace spmd {
// A simple matrix class to store and manipulate the cost matrices on edges.
// It can create a view for matrix transpose without copying the memory.
// TODO (zhuohan): Inherit from Array2D and add Transpose and operator+ (See
// tensorflow/compiler/xla/array2d.h;l=39)
template <typename T>
class Matrix {
 public:
  Matrix() : n_(0), m_(0), transpose_(false), data_(nullptr) {}

  Matrix(size_t n, size_t m) {
    this->n_ = n;
    this->m_ = m;
    transpose_ = false;
    data_ = std::make_shared<std::vector<T>>(n * m, T());
  }

  Matrix(size_t n, size_t m, bool transpose,
         std::shared_ptr<std::vector<T>> data) {
    this->n_ = n;
    this->m_ = m;
    this->transpose_ = transpose;
    this->data_ = data;
  }

  Matrix Transpose() { return Matrix(m_, n_, !transpose_, data_); }

  T operator()(size_t i, size_t j) const {
    size_t idx;
    if (transpose_) {
      idx = j * n_ + i;
    } else {
      idx = i * m_ + j;
    }
    CHECK(data_ != nullptr) << n_ << " , " << m_;
    CHECK(idx < n_ * m_) << idx << " , " << n_ << " , " << m_;
    return (*data_)[idx];
  }

  T& operator()(size_t i, size_t j) {
    size_t idx;
    if (transpose_) {
      idx = j * n_ + i;
    } else {
      idx = i * m_ + j;
    }
    CHECK(data_ != nullptr) << n_ << " , " << m_;
    CHECK(idx < n_ * m_) << idx << " , " << n_ << " , " << m_;
    return (*data_)[idx];
  }

  Matrix<T> operator+(const Matrix<T>& other) {
    CHECK_EQ(n_, other.n_);
    CHECK_EQ(m_, other.m_);
    Matrix ret = Matrix(n_, m_);
    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < m_; ++j) {
        ret(i, j) = operator()(i, j) + other(i, j);
      }
    }
    return ret;
  }

  std::string ToString() const {
    std::string str;

    for (size_t i = 0; i < n_; ++i) {
      for (size_t j = 0; j < m_; ++j) {
        absl::StrAppend(&str, operator()(i, j).ToString(), " ");
      }
      absl::StrAppend(&str, "\n");
    }

    return str;
  }

  size_t n_;
  size_t m_;
  bool transpose_;
  std::shared_ptr<std::vector<T>> data_;
};
}  // namespace spmd
}  // namespace xla
#endif  // XLA_HLO_EXPERIMENTAL_AUTO_SHARDING_MATRIX_H_
