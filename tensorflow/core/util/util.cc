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

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

absl::string_view NodeNamePrefix(const absl::string_view op_name) {
  absl::string_view sp(op_name);
  auto p = sp.find('/');
  if (p == absl::string_view::npos || p == 0) {
    return "";
  } else {
    return absl::string_view(sp.data(), p);
  }
}

absl::string_view NodeNameFullPrefix(const absl::string_view op_name) {
  absl::string_view sp(op_name);
  auto p = sp.rfind('/');
  if (p == absl::string_view::npos || p == 0) {
    return "";
  } else {
    return absl::string_view(sp.data(), p);
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

string SliceDebugString(const TensorShape& shape, const int64_t flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return absl::StrCat("[", flat, "]");

  // Compute strides
  absl::InlinedVector<int64_t, 32UL> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64_t left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    absl::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  absl::StrAppend(&result, "]");
  return result;
}

// TODO(penporn): Remove this function from util.cc
bool IsMKLEnabled() { return IsMklEnabled(); }

void DataTypeUnsupportedWarning(const DataType& dt) {
  static absl::once_flag cpu_dt_warn_once_flag;
  absl::call_once(cpu_dt_warn_once_flag, [dt] {
    LOG(ERROR) << "oneDNN supports " << DataType_Name(dt) << " only on "
               << "platforms with AVX-512. Falling back to the default "
               << "Eigen-based implementation if present.";
  });
}

bool IsDataTypeSupportedByOneDNNOnThisCPU(const DataType& dt) {
  bool result = false;
#ifdef INTEL_MKL
  using port::TestCPUFeature;
  if (dt == DT_FLOAT) {
    result = true;
  } else if (dt == DT_BFLOAT16) {
    result = (TestCPUFeature(port::CPUFeature::AVX512F) ||
              TestCPUFeature(port::CPUFeature::AVX_NE_CONVERT));
    if (result) VLOG(2) << "CPU supports " << DataType_Name(dt);
  } else if (DataTypeIsQuantized(dt)) {
    result = (TestCPUFeature(port::CPUFeature::AVX512F) ||
              TestCPUFeature(port::CPUFeature::AVX_VNNI_INT8));
    if (result) VLOG(2) << "CPU supports " << DataType_Name(dt);
  } else if (dt == DT_HALF) {
    // Float16 is not supported in oneDNN v2.x
#ifdef ENABLE_ONEDNN_V3
    // Some CPUs that don't support AVX-512 use AVX-NE-CONVERT to cast to and
    // from FP32
    result = ((TestCPUFeature(port::CPUFeature::AVX512BW) &&
               (TestCPUFeature(port::CPUFeature::AVX512_FP16) ||
                TestCPUFeature(port::CPUFeature::AMX_FP16))) ||
              TestCPUFeature(port::CPUFeature::AVX_NE_CONVERT));
    if (result) VLOG(2) << "CPU supports " << DataType_Name(dt);
#endif  // ENABLE_ONEDNN_V3
  } else {
    LOG(WARNING) << "Not handling type " << DataType_Name(dt);
  }
#endif  // INTEL_MKL
  return result;
}

bool IsAMXDataTypeSupportedByOneDNNOnThisCPU(const DataType& dt) {
  bool result = false;
#ifdef INTEL_MKL
  using port::TestCPUFeature;
  if (dt == DT_BFLOAT16) {
    result = TestCPUFeature(port::CPUFeature::AMX_BF16);
    if (result) VLOG(2) << "CPU supports AMX " << DataType_Name(dt);
  } else if (dt == DT_HALF) {
    // Float16 is not supported in oneDNN v2.x
#ifdef ENABLE_ONEDNN_V3
    result = TestCPUFeature(port::CPUFeature::AMX_FP16);
    if (result) VLOG(2) << "CPU supports AMX " << DataType_Name(dt);
#endif  // ENABLE_ONEDNN_V3
  } else if (DataTypeIsQuantized(dt)) {
    result = TestCPUFeature(port::CPUFeature::AMX_INT8);
    if (result) VLOG(2) << "CPU supports AMX " << DataType_Name(dt);
  } else {
    LOG(WARNING) << "Not handling type " << DataType_Name(dt);
  }
#endif  // INTEL_MKL
  return result;
}

// Check if oneDNN supports AVX-NE-CONVERT on CPU
bool IsAVXConvertSupportedByOneDNNOnThisCPU() {
  bool result = false;
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)
  using port::TestCPUFeature;
  result = TestCPUFeature(port::CPUFeature::AVX_NE_CONVERT);
#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
  return result;
}

}  // namespace tensorflow
