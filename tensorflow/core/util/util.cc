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

#include <string>
#include <vector>

#include "absl/base/call_once.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"

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

string SliceDebugString(const TensorShape& shape, const int64_t flat) {
  // Special case rank 0 and 1
  const int dims = shape.dims();
  if (dims == 0) return "";
  if (dims == 1) return strings::StrCat("[", flat, "]");

  // Compute strides
  gtl::InlinedVector<int64_t, 32> strides(dims);
  strides.back() = 1;
  for (int i = dims - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape.dim_size(i + 1);
  }

  // Unflatten index
  int64_t left = flat;
  string result;
  for (int i = 0; i < dims; i++) {
    strings::StrAppend(&result, i ? "," : "[", left / strides[i]);
    left %= strides[i];
  }
  strings::StrAppend(&result, "]");
  return result;
}

bool IsMKLEnabled() {
#ifndef INTEL_MKL
  return false;
#endif  // !INTEL_MKL
  static absl::once_flag once;
#ifdef ENABLE_MKL
  // Keeping TF_DISABLE_MKL env variable for legacy reasons.
  static bool oneDNN_disabled = false;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_DISABLE_MKL", false, &oneDNN_disabled));
    if (oneDNN_disabled) VLOG(2) << "TF-MKL: Disabling oneDNN";
  });
  return (!oneDNN_disabled);
#else
  static bool oneDNN_enabled = false;
  absl::call_once(once, [&] {
    TF_CHECK_OK(
        ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_OPTS", false, &oneDNN_enabled));
    if (oneDNN_enabled) {
      // Warn that this is not tested with GPU if there are GPUs available.
      std::vector<std::string> devices;
      Status s = DeviceFactory::ListAllPhysicalDevices(&devices);
      std::string gpu_message = "";
      for (const auto& device : devices) {
        if (device.find("GPU") != std::string::npos) {
          gpu_message =
              "We do NOT recommend turning them on with GPUs in the system. ";
          break;
        }
      }
      LOG(INFO) << "Experimental oneDNN custom operations are on. "
                << gpu_message
                << "To turn them off, set the environment variable "
                   "`TF_ENABLE_ONEDNN_OPTS=0`.";
    }
  });
  return oneDNN_enabled;
#endif  // ENABLE_MKL
}

}  // namespace tensorflow
