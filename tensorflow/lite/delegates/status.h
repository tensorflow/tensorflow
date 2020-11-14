/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_STATUS_H_
#define TENSORFLOW_LITE_DELEGATES_STATUS_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"

// This file defines data structures to represent detailed TFLite delegate
// status, e.g. NNAPI delegate application failure because of a driver issue
// etc. Such status is ONLY to be used for internal APIs.
// Note, we simply use TfLiteStatus to represent high-level status while
// delegate-specific status codes are defined with DelegateStatus.
// WARNING: This is an experimental feature that is subject to change.
namespace tflite {
namespace delegates {

// Defines the source of the code where it is generated from. We list all TFLite
// delegates that're officially implemented and available as of April, 2020
// (i.e. w/ 'TFLITE_' prefix to imply this).
enum class DelegateStatusSource {
  NONE = 0,
  TFLITE_GPU = 1,
  TFLITE_NNAPI = 2,
  TFLITE_HEXAGON = 3,
  TFLITE_XNNPACK = 4,
  TFLITE_COREML = 5,
  MAX_NUM_SOURCES = std::numeric_limits<int32_t>::max(),
};

// Defines the detailed status that combines a DelegateStatusSource and a
// status int32_t code.
class DelegateStatus {
 public:
  DelegateStatus() : DelegateStatus(DelegateStatusSource::NONE, 0) {}
  explicit DelegateStatus(int32_t code)
      : DelegateStatus(DelegateStatusSource::NONE, code) {}
  explicit DelegateStatus(int64_t full_status)
      : DelegateStatus(
            static_cast<DelegateStatusSource>(
                full_status >> 32 &
                static_cast<int32_t>(DelegateStatusSource::MAX_NUM_SOURCES)),
            static_cast<int32_t>(full_status &
                                 std::numeric_limits<int32_t>::max())) {}
  DelegateStatus(DelegateStatusSource source, int32_t code)
      : source_(static_cast<int32_t>(source)), code_(code) {}

  // Return the detailed full status encoded as a int64_t value.
  int64_t full_status() const {
    return static_cast<int64_t>(source_) << 32 | code_;
  }

  DelegateStatusSource source() const {
    return static_cast<DelegateStatusSource>(source_);
  }

  int32_t code() const { return code_; }

 private:
  // value of a DelegateStatusSource, like DelegateStatusSource::TFLITE_GPU
  int32_t source_;
  // value of a status code, like kTfLiteOk.
  int32_t code_;
};

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_STATUS_H_
