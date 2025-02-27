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

#include "tensorflow/lite/experimental/litert/c/vendors/qualcomm/accelerator_options.h"

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"

namespace {

struct QnnAccleratorCompilationOptions {
  static constexpr const absl::string_view kIdentifier = "qnn-accelerator";
  static constexpr const LiteRtApiVersion kVersion = {0, 1, 0};

  // This NEEDS to be the first non-static field of the structure.
  LiteRtAcceleratorCompilationOptionsHeader link;

  // The options, in whatever form/hierarchy.
  LiteRtQnnLogLevel log_level;
  TfLiteQnnDelegateHtpPerformanceMode htp_performance_mode;

  // Everything below is just helpers.

  // Allocates and sets the basic structure for the accelerator options.
  static LiteRtStatus Create(LiteRtAcceleratorCompilationOptions* options) {
    if (!options) {
      return kLiteRtStatusErrorInvalidArgument;
    }
    *options = reinterpret_cast<LiteRtAcceleratorCompilationOptions>(
        new QnnAccleratorCompilationOptions());
    LiteRtSetAcceleratorCompilationOptionsDestructor(*options, Destroy);
    LiteRtSetAcceleratorCompilationOptionsIdentifier(*options,
                                                     kIdentifier.data());
    LiteRtSetAcceleratorCompilationOptionsVersion(*options, kVersion);
    return kLiteRtStatusOk;
  }

  static litert::Expected<QnnAccleratorCompilationOptions*>
  ConvertFromErasedOptions(LiteRtAcceleratorCompilationOptions options) {
    LITERT_RETURN_IF_ERROR(
        options, litert::Unexpected(kLiteRtStatusErrorInvalidArgument));
    const char* identifier = nullptr;
    LITERT_RETURN_IF_ERROR(
        LiteRtGetAcceleratorCompilationOptionsIdentifier(options, &identifier));
    LITERT_RETURN_IF_ERROR(
        identifier == kIdentifier,
        litert::Unexpected(kLiteRtStatusErrorInvalidArgument));
    return reinterpret_cast<QnnAccleratorCompilationOptions*>(options);
  }

 private:
  // Destroys the options.
  static void Destroy(LiteRtAcceleratorCompilationOptions options) {
    delete reinterpret_cast<QnnAccleratorCompilationOptions*>(options);
  }
};

}  // namespace

// C API implementation, ABI stable.

extern "C" {

LiteRtStatus LiteRtCreateQualcommAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options) {
  return QnnAccleratorCompilationOptions::Create(options);
}

LiteRtStatus LiteRtSetQualcommAcceleratorLogLevel(
    LiteRtAcceleratorCompilationOptions options, LiteRtQnnLogLevel level) {
  LITERT_ASSIGN_OR_RETURN(
      auto* qnn_opts,
      QnnAccleratorCompilationOptions::ConvertFromErasedOptions(options));
  LITERT_RETURN_IF_ERROR(
      LiteRtCompareApiVersion(options->version, {1, 0, 0}) >= 0,
      kLiteRtStatusErrorWrongVersion);
  qnn_opts->log_level = level;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtSetQualcommAcceleratorHtpPerformanceMode(
    LiteRtAcceleratorCompilationOptions options,
    TfLiteQnnDelegateHtpPerformanceMode mode) {
  LITERT_ASSIGN_OR_RETURN(
      auto* qnn_opts,
      QnnAccleratorCompilationOptions::ConvertFromErasedOptions(options));
  LITERT_RETURN_IF_ERROR(
      LiteRtCompareApiVersion(options->version, {1, 0, 0}) >= 0,
      kLiteRtStatusErrorWrongVersion);
  qnn_opts->htp_performance_mode = mode;
  return kLiteRtStatusOk;
}

}  // extern "C"
