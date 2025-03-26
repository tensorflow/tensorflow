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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILATION_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILATION_OPTIONS_H_

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_compilation_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The compilation options for the LiteRtCompiledModel.
LITERT_DEFINE_HANDLE(LiteRtCompilationOptions);

// Creates a compilation option object.
LiteRtStatus LiteRtCreateCompilationOptions(LiteRtCompilationOptions* options);

// Destroys a compilation option object.
void LiteRtDestroyCompilationOptions(LiteRtCompilationOptions options);

// Sets the requested hardware accelerators to apply during model compilation.
LiteRtStatus LiteRtSetCompilationOptionsHardwareAccelerators(
    LiteRtCompilationOptions options,
    LiteRtHwAcceleratorSet hardware_accelerators);

// Gets the hardware accelerators to apply during model compilation.
LiteRtStatus LiteRtGetCompilationOptionsHardwareAccelerators(
    LiteRtCompilationOptions options,
    LiteRtHwAcceleratorSet* hardware_accelerators);

// Adds compilation options for a specific accelerator to the accelerator
// compilation option list.
//
// Note: Multiple accelerator options may be added to the options object.
//
// Note: `accelerator_compilation_options`'s ownership is transferred to
// `options`.
LiteRtStatus LiteRtAddAcceleratorCompilationOptions(
    LiteRtCompilationOptions options,
    LiteRtAcceleratorCompilationOptions accelerator_compilation_options);

// Retrieves the head of the accelerator compilation option list.
//
// Note: The following elements may be retrieved with
// `LiteRtGetNextAcceleratorCompilationOptions`.
LiteRtStatus LiteRtGetAcceleratorCompilationOptions(
    LiteRtCompilationOptions options,
    LiteRtAcceleratorCompilationOptions* accelerator_compilation_options);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILATION_OPTIONS_H_
