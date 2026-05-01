/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
// This file is copied in MLIR to avoid a dependency on TFLite.
// LINT.IfChange

#include "tensorflow/lite/kernels/internal/runtime_shape.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tflite {

namespace {

bool CheckedMul(size_t a, size_t b, size_t& product) {
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
    return false;
  }
  product = a * b;
  return true;
}

bool CheckedMul(size_t a, int32_t b, size_t& product) {
  if (b < 0) return false;
  return CheckedMul(a, static_cast<size_t>(b), product);
}

bool CheckedCast(size_t in, int& out) {
  if (in > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
  out = static_cast<int>(in);
  return true;
}

}  // namespace

RuntimeShape::~RuntimeShape() {
#ifndef TF_LITE_STATIC_MEMORY
  if (size_ > kMaxSmallSize) {
    delete[] dims_pointer_;
  }
#endif  // TF_LITE_STATIC_MEMORY
}

int32_t RuntimeShape::Dims(int i) const {
  TFLITE_DCHECK_GE(i, 0);
  TFLITE_DCHECK_LT(i, size_);
#ifndef TF_LITE_STATIC_MEMORY
  return size_ > kMaxSmallSize ? dims_pointer_[i] : dims_[i];
#else
  return dims_[i];
#endif  // TF_LITE_STATIC_MEMORY
}

void RuntimeShape::ReplaceWith(int dimensions_count, const int32_t* dims_data) {
#ifndef TF_LITE_STATIC_MEMORY
  Resize(dimensions_count);
  int32_t* dst_dims = DimsData();
#else
  TFLITE_DCHECK_LE(dimensions_count, kMaxSmallSize);
  size_ = dimensions_count;
  int32_t* dst_dims = DimsData();
#endif  // TF_LITE_STATIC_MEMORY
  std::memcpy(dst_dims, dims_data, dimensions_count * sizeof(int32_t));
}

int RuntimeShape::FlatSize() const {
  int buffer_size = 1;
  const int* dims_data = reinterpret_cast<const int*>(DimsData());
  for (int i = 0; i < size_; i++) {
    buffer_size *= dims_data[i];
  }
  return buffer_size;
}

bool RuntimeShape::CheckedNumElementsInRange(int start, int end,
                                             size_t& out) const {
  if (start < 0 || end < start || end > size_) return false;
  size_t checked_out = 1;
  const int32_t* dims_data = DimsData();
  for (int i = start; i < end; ++i) {
    if (!CheckedMul(checked_out, dims_data[i], checked_out)) return false;
  }
  out = checked_out;
  return true;
}

bool RuntimeShape::CheckedNumElementsInRange(int start, int end,
                                             int& out) const {
  size_t checked_out = 0;
  return CheckedNumElementsInRange(start, end, checked_out) &&
         CheckedCast(checked_out, out);
}

bool RuntimeShape::CheckedSizeToDimension(int end, size_t& out) const {
  return CheckedNumElementsInRange(0, end, out);
}

bool RuntimeShape::CheckedSizeToDimension(int end, int& out) const {
  return CheckedNumElementsInRange(0, end, out);
}

bool RuntimeShape::CheckedSizeFromDimension(int start, size_t& out) const {
  return CheckedNumElementsInRange(start, size_, out);
}

bool RuntimeShape::CheckedSizeFromDimension(int start, int& out) const {
  return CheckedNumElementsInRange(start, size_, out);
}

bool RuntimeShape::CheckedFlatSize(size_t& flat_size) const {
  return CheckedNumElementsInRange(0, size_, flat_size);
}

bool RuntimeShape::CheckedFlatSizeSkipDim(int skip_dim,
                                          size_t& flat_size) const {
  if (skip_dim < 0 || skip_dim >= size_) return false;
  size_t prefix = 0;
  if (!CheckedNumElementsInRange(0, skip_dim, prefix)) return false;
  size_t suffix = 0;
  if (!CheckedNumElementsInRange(skip_dim + 1, size_, suffix)) return false;
  return CheckedMul(prefix, suffix, flat_size);
}

}  // namespace tflite

// LINT.ThenChange(//tensorflow/compiler/mlir/lite/kernels/internal/runtime_shape.cc)
