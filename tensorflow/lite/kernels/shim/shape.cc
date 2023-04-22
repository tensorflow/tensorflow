/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/shape.h"

#include <string>

namespace tflite {
namespace shim {

bool Shape::operator==(const Shape& rhs) const {
  if (!has_value() || !rhs.has_value()) return false;
  if (value_.size() != rhs.value_.size()) return false;
  for (int i = 0; i < value_.size(); ++i)
    if (value_[i] != rhs.value_[i] || value_[i] == kUnknownDim ||
        rhs.value_[i] == kUnknownDim)
      return false;
  return true;
}

bool Shape::operator!=(const Shape& rhs) const { return !(*this == rhs); }

bool Shape::Compatible(const Shape& rhs) const {
  if (!has_value() || !rhs.has_value()) return true;
  if (value_.size() != rhs.value_.size()) return false;
  for (int i = 0; i < value_.size(); ++i) {
    const auto lhs_i = value_[i];
    const auto rhs_i = rhs.value_[i];
    if (lhs_i != rhs_i && lhs_i != kUnknownDim && rhs_i != kUnknownDim)
      return false;
  }
  return true;
}

std::string Shape::ToString() const {
  std::string ret;
  if (has_value()) {
    ret += "[";
    if (!value_.empty()) ret += " ";
    for (const auto dim : value_) {
      if (dim != kUnknownDim) {
        ret += std::to_string(dim);
      } else {
        ret += "?";
      }
      ret += " ";
    }
    ret += "]";
  } else {
    ret += "?";
  }
  return ret;
}

bool Shape::FullyDefined() const {
  if (!has_value_) return false;
  for (const auto dim : value_)
    if (dim == kUnknownDim) return false;
  return true;
}

int Shape::AddDims(const int dim1, const int dim2) {
  if (dim1 == kUnknownDim || dim2 == kUnknownDim) return kUnknownDim;
  return dim1 + dim2;
}

int Shape::Dim(const int idx) const {
  if (has_value_) return value_[idx];
  return kUnknownDim;
}

}  // namespace shim
}  // namespace tflite
