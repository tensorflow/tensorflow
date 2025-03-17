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

#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/environment_options.h"

extern "C" {

LiteRtStatus LiteRtGetEnvironmentOptionsValue(LiteRtEnvironmentOptions options,
                                              LiteRtEnvOptionTag tag,
                                              LiteRtAny* value) {
  LITERT_RETURN_IF_ERROR(
      options, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument))
      << "`options` handle is null.";
  LITERT_RETURN_IF_ERROR(
      value, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument))
      << "`value` handle is null.";
  LITERT_ASSIGN_OR_RETURN(*value, options->GetOption(tag));
  return kLiteRtStatusOk;
}

}  // extern "C"
