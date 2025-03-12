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

#include "tensorflow/lite/experimental/litert/c/litert_compilation_options.h"

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/runtime/compilation_options.h"

#define LRT_CHECK_NON_NULL(handle)                          \
  if (!(handle)) {                                          \
    LITERT_LOG(LITERT_ERROR, #handle " must not be null."); \
    return kLiteRtStatusErrorInvalidArgument;               \
  }

#define LRT_REQUIRE_VERSION(MAJOR, MINOR, PATCH)                               \
  if (LiteRtCompareApiVersion(options->version, {(MAJOR), (MINOR), (PATCH)}) < \
      0) {                                                                     \
    LITERT_LOG(LITERT_ERROR,                                                   \
               "The version of this option object is too old. Expected at "    \
               "least %d.%d.%d, got %d.%d.%d",                                 \
               (MAJOR), (MINOR), (PATCH), options->version.major,              \
               options->version.minor, options->version.patch);                \
    return kLiteRtStatusErrorWrongVersion;                                     \
  }

extern "C" {

LiteRtStatus LiteRtCreateCompilationOptions(LiteRtCompilationOptions* options) {
  LRT_CHECK_NON_NULL(options);
  *options = new LiteRtCompilationOptionsT;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilationOptions(LiteRtCompilationOptions options) {
  LiteRtDestroyAcceleratorCompilationOptions(
      options->accelerator_compilation_options);
  delete options;
}

LiteRtStatus LiteRtSetCompilationOptionsHardwareAccelerators(
    LiteRtCompilationOptions options,
    LiteRtHwAcceleratorSet hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  if ((hardware_accelerators &
       (kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
        kLiteRtHwAcceleratorNpu)) != hardware_accelerators) {
    LITERT_LOG(LITERT_ERROR,
               "Invalid bitfield value for hardware accelerator set: %d.",
               hardware_accelerators);
    return kLiteRtStatusErrorInvalidArgument;
  }
  options->hardware_accelerators = hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilationOptionsHardwareAccelerators(
    LiteRtCompilationOptions options,
    LiteRtHwAcceleratorSet* hardware_accelerators) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(hardware_accelerators);
  LRT_REQUIRE_VERSION(0, 0, 0);
  *hardware_accelerators = options->hardware_accelerators;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAddAcceleratorCompilationOptions(
    LiteRtCompilationOptions options,
    LiteRtAcceleratorCompilationOptions accelerator_compilation_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(accelerator_compilation_options);

  LiteRtAppendAcceleratorCompilationOptions(
      &options->accelerator_compilation_options,
      accelerator_compilation_options);

  return kLiteRtStatusOk;
}

// Retrieves the head of the accelerator compilation option list.
//
// Note: The following elements may be retrieved with
// `LiteRtGetNextAcceleratorCompilationOptions`.
LiteRtStatus LiteRtGetAcceleratorCompilationOptions(
    LiteRtCompilationOptions options,
    LiteRtAcceleratorCompilationOptions* accelerator_compilation_options) {
  LRT_CHECK_NON_NULL(options);
  LRT_CHECK_NON_NULL(accelerator_compilation_options);
  *accelerator_compilation_options = options->accelerator_compilation_options;
  return kLiteRtStatusOk;
}

}  // extern "C"
