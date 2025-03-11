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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#ifdef __cplusplus
extern "C" {
#endif

struct LiteRtAcceleratorCompilationOptionsHeader;

typedef struct LiteRtAcceleratorCompilationOptionsHeader*
    LiteRtAcceleratorCompilationOptions;

// Gets the version of the accelerator option structure.
LiteRtStatus LiteRtGetAcceleratorCompilationOptionsVersion(
    LiteRtAcceleratorCompilationOptions options, int* version);

// Gets the accelerator option structure identifier.
LiteRtStatus LiteRtGetAcceleratorCompilationOptionsIdentifier(
    LiteRtAcceleratorCompilationOptions options, const char** identifier);

// Sets the identifier for an acceleration compilation option object.
//
// NOTE: The identifier's lifetime is managed by the caller.
LiteRtStatus LiteRtSetAcceleratorCompilationOptionsIdentifier(
    struct LiteRtAcceleratorCompilationOptionsHeader* options,
    const char* identifier);

// Gets the next link in the option list.
//
// Sets `accelerator_compilation_options` to NULL if none are left.
LiteRtStatus LiteRtGetNextAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* accelerator_compilation_options);

// Sets the options' destructor.
//
// We need this for option objects that may own some of their data. The most
// common use case here being helper functions that build a path from other
// program inputs. If the options structure doesn't own the data, then the user
// must ensure that the string outlives the compiled model, which may be tricky.
// This lets the user define a function that will be called to clean up the
// data.
LiteRtStatus LiteRtSetAcceleratorCompilationOptionsDestructor(
    struct LiteRtAcceleratorCompilationOptionsHeader* options,
    void (*destructor)(struct LiteRtAcceleratorCompilationOptionsHeader*));

// Appends a new compilation option object to the list.
//
// This goes through the links in the option list and appends the given link.
//
// `options` must be non-null, `*options` may however be null, in which case
// this call is equivalent to `*options = appended_options`.
LiteRtStatus LiteRtAppendAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions* options,
    LiteRtAcceleratorCompilationOptions appended_options);

// Releases an accelerator option structure list.
//
// Equivalent to calling the destructor passed to the function above.
//
// Warning: This should not be called manually after the option structure has
// been added to the compilation options.
LiteRtStatus LiteRtDestroyAcceleratorCompilationOptions(
    LiteRtAcceleratorCompilationOptions options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ACCELERATOR_COMPILATION_OPTIONS_H_
