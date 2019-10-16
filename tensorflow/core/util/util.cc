/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/util.h"

#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

StringPiece NodeNamePrefix(const StringPiece& op_name) {
  StringPiece sp(op_name);
  auto p = sp.find('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

StringPiece NodeNameFullPrefix(const StringPiece& op_name) {
  StringPiece sp(op_name);
  auto p = sp.rfind('/');
  if (p == StringPiece::npos || p == 0) {
    return "";
  } else {
    return StringPiece(sp.data(), p);
  }
}

MovingAverage::MovingAverage(int window)
    : window_(window),
      sum_(0.0),
      data_(new double[window_]),
      head_(0),
      count_(0) {
  CHECK_GE(window, 1);
}

MovingAverage::~MovingAverage() { delete[] data_; }

void MovingAverage::Clear() {
  count_ = 0;
  head_ = 0;
  sum_ = 0;
}

double MovingAverage::GetAverage() const {
  if (count_ == 0) {
    return 0;
  } else {
    return static_cast<double>(sum_) / count_;
  }
}

void MovingAverage::AddValue(double v) {
  if (count_ < window_) {
    // This is the warmup phase. We don't have a full window's worth of data.
    head_ = count_;
    data_[count_++] = v;
  } else {
    if (window_ == ++head_) {
      head_ = 0;
    }
    // Toss the oldest element
    sum_ -= data_[head_];
    // Add the newest element
    data_[head_] = v;
  }
  sum_ += v;
}

static char hex_char[] = "0123456789abcdef";

string PrintMemory(const char* ptr, size_t n) {
  string ret;
  ret.resize(n * 3);
  for (int i = 0; i < n; ++i) {
    ret[i * 3] = ' ';
    ret[i * 3 + 1] = hex_char[ptr[i] >> 4];
    ret[i * 3 + 2] = hex_char[ptr[i] & 0xf];
  }
  return ret;
}

string SliceDebugString(const TensorShape& shape, const int64 flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64 left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

#ifdef INTEL_MKL
bool DisableMKL() {
  enum MklStatus { MKL_DEFAULT = 0, MKL_ON = 1, MKL_OFF = 2 };
  static MklStatus status = MKL_DEFAULT;
  if (status == MKL_DEFAULT) {
    char* tf_disable_mkl = getenv("TF_DISABLE_MKL");
    if ((tf_disable_mkl != NULL) && (std::stoi(tf_disable_mkl) == 1)) {
      VLOG(2) << "TF-MKL: Disabling MKL";
      status = MKL_OFF;
    } else {
      status = MKL_ON;
    }
  }
  return status == MKL_OFF ? true : false;
}
#endif  // INTEL_MKL
}  // namespace tensorflow
