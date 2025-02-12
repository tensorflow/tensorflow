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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"

extern "C" {

LiteRtStatus LiteRtGetNextAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options) {
  if (!options || !*options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *options = (*options)->next;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtAppendAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options,
    LiteRtAcceleratorCompilationOptions appended_options) {
  if (!options) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (options->next) {
    options = options->next;
  }
  options->next = appended_options;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorCompilationOptionsIdentifier(
    LiteRtAcceleratorCompilationOptions options, const char** identifier) {
  if (!options || !identifier) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *identifier = options->identifier;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtDestroyAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options) {
  if (!options || !options->ReleaseData) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  while (options) {
    LiteRtAcceleratorCompilationOptions next = options->next;
    options->ReleaseData(options);
    options = next;
  }
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorCompilationOptionsVersion(
    LiteRtAcceleratorCompilationOptions options, LiteRtApiVersion* version) {
  if (!options || !version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *version = options->version;
  return kLiteRtStatusOk;
}

}  // extern "C"
