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

#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/core/accelerator.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gets the number of accelerators registered to LiteRT.
LiteRtStatus LiteRtGetNumAccelerators(LiteRtEnvironment environment,
                                      LiteRtParamIndex* num_accelerators) {
  if (!num_accelerators) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!environment) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  *num_accelerators = environment->GetAcceleratorRegistry().size();
  return kLiteRtStatusOk;
}

// Gets the accelerator at given index that is registered to LiteRT.
LiteRtStatus LiteRtGetAccelerator(LiteRtEnvironment environment,
                                  LiteRtParamIndex index,

                                  LiteRtAccelerator* accelerator) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!environment) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::Expected<LiteRtAccelerator> registered_accelerator =
      environment->GetAcceleratorRegistry().Get(index);
  if (!registered_accelerator.HasValue()) {
    return registered_accelerator.Error().Status();
  }
  *accelerator = registered_accelerator.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorName(LiteRtAccelerator accelerator,
                                      char const** name) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!accelerator->GetName) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetName(accelerator, name);
}

LiteRtStatus LiteRtGetAcceleratorId(LiteRtAccelerator accelerator,
                                    LiteRtAcceleratorId* id) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!id) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!accelerator->env) {
    return kLiteRtStatusErrorInvalidArgument;
  }

  litert::Expected<LiteRtParamIndex> index =
      accelerator->env->GetAcceleratorRegistry().FindAcceleratorIndex(
          accelerator);
  if (!index.HasValue()) {
    return index.Error().Status();
  }
  *id = index.Value();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetAcceleratorVersion(LiteRtAccelerator accelerator,
                                         LiteRtApiVersion* version) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!accelerator->GetVersion) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetVersion(accelerator, version);
}

LiteRtStatus LiteRtGetAcceleratorHardwareSupport(
    LiteRtAccelerator accelerator, LiteRtHwAcceleratorSet* supported_hardware) {
  if (!accelerator) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!accelerator->GetHardwareSupport) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetHardwareSupport(accelerator, supported_hardware);
}

#ifdef __cplusplus
}  // extern "C"
#endif
