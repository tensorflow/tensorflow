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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"

#ifdef __cplusplus
extern "C" {
#endif

LITERT_DEFINE_HANDLE(LiteRtAccelerator);

typedef size_t LiteRtAcceleratorId;

// Gets the number of accelerators registered to LiteRT.
LiteRtStatus LiteRtGetNumAccelerators(LiteRtEnvironment environment,
                                      LiteRtParamIndex* num_accelerators);

// Gets the accelerator at given index that is registered to LiteRT.
LiteRtStatus LiteRtGetAccelerator(LiteRtEnvironment environment,
                                  LiteRtParamIndex index,
                                  LiteRtAccelerator* accelerator);

// Fetches the name of the accelerator.
//
// Note: client code does not need to manage the `name` lifetime.
LiteRtStatus LiteRtGetAcceleratorName(LiteRtAccelerator accelerator,
                                      char const** name);

// Fetches the accelerator identifier.
//
// The identifier is a runtime unique number, provided by the registrar to the
// accelerator upon registration.
LiteRtStatus LiteRtGetAcceleratorId(LiteRtAccelerator accelerator,
                                    LiteRtAcceleratorId* id);

// Fetches the version of the accelerator implementation.
//
// Note: This is NOT the LiteRT version. It's the accelerator specific software
// implementation version.
LiteRtStatus LiteRtGetAcceleratorVersion(LiteRtAccelerator accelerator,
                                         LiteRtApiVersion* version);

// Fetches the accelerator hardware.
//
// `supported_hardware` is a bitfield of `LiteRtHwAccelerators` values.
LiteRtStatus LiteRtGetAcceleratorHardwareSupport(
    LiteRtAccelerator accelerator, LiteRtHwAcceleratorSet* supported_hardware);

// Returns whether the accelerator TFLite delegate does some JIT compilation.
LiteRtStatus LiteRtIsAcceleratorDelegateResponsibleForJitCompilation(
    LiteRtAccelerator accelerator, bool* does_jit_compilation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_H_
