// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/c/litert_environment.h"

#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"

LiteRtStatus LiteRtEnvironmentCreate(int num_options,
                                     const LiteRtEnvOption* options) {
  if (auto status = litert::internal::Environment::CreateWithOptions(
          absl::MakeSpan(options, num_options));
      !status) {
    return status.Error().Status();
  }
  return kLiteRtStatusOk;
}

void LiteRtEnvironmentDestroy() { litert::internal::Environment::Destroy(); }
