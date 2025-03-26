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
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/accelerator.h"

#ifdef __cplusplus
extern "C" {
#endif

// Gets the number of accelerators registered to LiteRT.
LiteRtStatus LiteRtGetNumAccelerators(LiteRtEnvironment environment,
                                      LiteRtParamIndex* num_accelerators) {
  if (!environment || !num_accelerators) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_accelerators = environment->GetAcceleratorRegistry().size();
  return kLiteRtStatusOk;
}

// Gets the accelerator at given index that is registered to LiteRT.
LiteRtStatus LiteRtGetAccelerator(LiteRtEnvironment environment,
                                  LiteRtParamIndex index,

                                  LiteRtAccelerator* accelerator) {
  if (!environment || !accelerator) {
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
  if (!accelerator || !accelerator->GetName || !name) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetName(accelerator, name);
}

LiteRtStatus LiteRtGetAcceleratorId(LiteRtAccelerator accelerator,
                                    LiteRtAcceleratorId* id) {
  if (!accelerator || !accelerator->env || !id) {
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
  if (!accelerator || !accelerator->GetVersion || !version) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetVersion(accelerator, version);
}

LiteRtStatus LiteRtGetAcceleratorHardwareSupport(
    LiteRtAccelerator accelerator, LiteRtHwAcceleratorSet* supported_hardware) {
  if (!accelerator || !accelerator->GetHardwareSupport || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  return accelerator->GetHardwareSupport(accelerator, supported_hardware);
}

LiteRtStatus LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
    LiteRtAccelerator accelerator, bool* does_jit_compilation) {
  if (!accelerator || !does_jit_compilation) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  if (!accelerator->IsTfLiteDelegateResponsibleForJitCompilation) {
    *does_jit_compilation = false;
    return kLiteRtStatusOk;
  }
  return accelerator->IsTfLiteDelegateResponsibleForJitCompilation(
      accelerator, does_jit_compilation);
}

#ifdef __cplusplus
}  // extern "C"
#endif
