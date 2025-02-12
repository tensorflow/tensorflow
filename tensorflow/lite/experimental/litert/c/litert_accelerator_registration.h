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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_REGISTRATION_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_REGISTRATION_H_

#include "tensorflow/lite/experimental/litert/c/litert_accelerator.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_compiled_model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Creates an empty accelerator handle.
LiteRtStatus LiteRtCreateAccelerator(LiteRtAccelerator* accelerator);

// Destroys an accelerator handle.
//
// Warning: This SHOULD NOT BE CALLED after a call to
// `LiteRtRegisterAccelerator`.
LiteRtStatus LiteRtDestroyAccelerator(LiteRtAccelerator accelerator);

// Sets the registration data AND clean-up function, then registers the
// accelerator with the LiteRT environment.
//
// - `data` and `ReleaseData` may be null.
//
// Note: After this function returns successfully, `data` is managed by the
// LiteRT environment. `ReleaseData` is called to release its memory.
//
// Warning: In case of failure, `accelerator` is released and `data` is released
// using `ReleaseData`.
LiteRtStatus LiteRtRegisterAccelerator(LiteRtEnvironment environment,
                                       LiteRtAccelerator accelerator,
                                       void* data, void (*ReleaseData)(void*));

// Sets the function used to retrieve the accelerator name.
LiteRtStatus LiteRtSetAcceleratorGetName(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetName)(LiteRtAccelerator accelerator, const char** name));

// Sets the function used to retrieve the accelerator implementation version.
//
// Note: This is NOT the LiteRT version. It's the accelerator specific software
// implementation version.
LiteRtStatus LiteRtSetAcceleratorGetVersion(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetVersion)(LiteRtAccelerator accelerator,
                               LiteRtApiVersion* version));

// Sets the function used to retrieve the accelerator hardware support.
LiteRtStatus LiteRtSetAcceleratorGetHardwareSupport(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*GetHardwareSupport)(
        LiteRtAccelerator accelerator,
        LiteRtHwAcceleratorSet* supported_hardware));

// Set the function used to apply the accelerator to the compiled model.
//
// This is called by LiteRtCreateCompiledModel.
LiteRtStatus LiteRtSetApplyToModel(
    LiteRtAccelerator accelerator,
    LiteRtStatus (*ApplyToModel)(LiteRtAccelerator accelerator,
                                 LiteRtCompiledModel compiled_model));

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_REGISTRATION_H_
