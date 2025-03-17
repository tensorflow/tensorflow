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
#include "tensorflow/lite/experimental/litert/c/litert_environment_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerators/auto_registration.h"
#include "tensorflow/lite/experimental/litert/runtime/gpu_environment.h"

#ifdef __cplusplus
extern "C" {
#endif

LiteRtStatus LiteRtEnvironmentCreate(int num_options,
                                     const LiteRtEnvOption* options,
                                     LiteRtEnvironment* environment) {
  LITERT_RETURN_IF_ERROR(environment != nullptr,
                         kLiteRtStatusErrorInvalidArgument);
  LITERT_ASSIGN_OR_RETURN(auto env, LiteRtEnvironmentT::CreateWithOptions(
                                        absl::MakeSpan(options, num_options)));
  litert::TriggerAcceleratorAutomaticRegistration(*env);
  *environment = env.release();
  return kLiteRtStatusOk;
}

void LiteRtDestroyEnvironment(LiteRtEnvironment environment) {
  if (environment != nullptr) {
    delete environment;
  }
}

LiteRtStatus LiteRtGetEnvironmentOptions(LiteRtEnvironment environment,
                                         LiteRtEnvironmentOptions* options) {
  LITERT_RETURN_IF_ERROR(
      environment, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                       << "Environment pointer is null.");
  LITERT_RETURN_IF_ERROR(
      options, litert::ErrorStatusBuilder(kLiteRtStatusErrorInvalidArgument)
                   << "Options pointer is null.");
  *options = &environment->GetOptions();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGpuGlobalEnvironmentCreate(int num_options,
                                              const LiteRtEnvOption* options) {
  LITERT_ASSIGN_OR_RETURN(auto env, LiteRtEnvironmentT::CreateWithOptions(
                                        absl::MakeSpan(options, num_options)));
  litert::internal::GpuEnvironmentSingleton::Create(env.release());
  return kLiteRtStatusOk;
}

#ifdef __cplusplus
}  // extern "C"
#endif
