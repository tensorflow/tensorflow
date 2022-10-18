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
#ifndef TENSORFLOW_LITE_DELEGATES_TELEMETRY_H_
#define TENSORFLOW_LITE_DELEGATES_TELEMETRY_H_

#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"

// This file implements utilities for delegate telemetry. These enable
// representation and reporting of hardware-specific configurations, status
// codes, etc.
// These APIs are for internal use *only*, and should be modified with care to
// avoid incompatibilities between delegates & runtime.
// WARNING: This is an experimental feature that is subject to change.
namespace tflite {
namespace delegates {

// Used to identify specific events for tflite::Profiler.
constexpr char kDelegateSettingsTag[] = "delegate_settings";
constexpr char kDelegateStatusTag[] = "delegate_status";

// Defines the delegate or hardware-specific 'namespace' that a status code
// belongs to. For example, GPU delegate errors might be belong to TFLITE_GPU,
// while OpenCL-specific ones might be TFLITE_GPU_CL.
enum class DelegateStatusSource {
  NONE = 0,
  TFLITE_GPU = 1,
  TFLITE_NNAPI = 2,
  TFLITE_HEXAGON = 3,
  TFLITE_XNNPACK = 4,
  TFLITE_COREML = 5,
  MAX_NUM_SOURCES = std::numeric_limits<int32_t>::max(),
};

// DelegateStatus defines a namespaced status with a combination of
// DelegateStatusSource & the corresponding fine-grained 32-bit code. Used to
// convert to/from a 64-bit representation as follows:
//
// delegates::DelegateStatus status(
//      delegates::DelegateStatusSource::TFLITE_NNAPI,
//      ANEURALNETWORKS_OP_FAILED);
// int64_t code = status.full_status();
//
// auto parsed_status = delegates::DelegateStatus(code);
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

// Used by delegates to report their configuration/settings to TFLite.
// Calling this method adds a new GENERAL_RUNTIME_INSTRUMENTATION_EVENT to
// the runtime Profiler.
TfLiteStatus ReportDelegateSettings(TfLiteContext* context,
                                    TfLiteDelegate* delegate,
                                    const TFLiteSettings& settings);

// Used by delegates to report their status to the TFLite runtime.
// Calling this method adds a new GENERAL_RUNTIME_INSTRUMENTATION_EVENT to
// the runtime Profiler.
TfLiteStatus ReportDelegateStatus(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  const DelegateStatus& status);

}  // namespace delegates
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_TELEMETRY_H_
