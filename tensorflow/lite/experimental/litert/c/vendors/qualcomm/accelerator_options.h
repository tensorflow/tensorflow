// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

extern "C" {

enum LiteRtQnnLogLevel {
  /// Disable delegate and QNN backend logging messages.
  kLogOff = 0,
  kLogLevelError = 1,
  kLogLevelWarn = 2,
  kLogLevelInfo = 3,
  kLogLevelVerbose = 4,
  kLogLevelDebug = 5,
};

enum TfLiteQnnDelegateHtpPerformanceMode {
  kHtpDefault = 0,
  kHtpSustainedHighPerformance = 1,
  kHtpBurst = 2,
  kHtpHighPerformance = 3,
  kHtpPowerSaver = 4,
  kHtpLowPowerSaver = 5,
  kHtpHighPowerSaver = 6,
  kHtpLowBalanced = 7,
  kHtpBalanced = 8,
  kHtpExtremePowerSaver = 9,
};

LiteRtStatus LiteRtCreateQualcommAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options);

LiteRtStatus LiteRtSetQualcommAcceleratorLogLevel(
    LiteRtAcceleratorCompilationOptions options, LiteRtQnnLogLevel level);

LiteRtStatus LiteRtSetQualcommAcceleratorHtpPerformanceMode(
    LiteRtAcceleratorCompilationOptions options,
    TfLiteQnnDelegateHtpPerformanceMode modek);

}  // extern "C"

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_VENDORS_QUALCOMM_ACCELERATOR_OPTIONS_H_
