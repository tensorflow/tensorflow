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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILATION_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILATION_OPTIONS_H_

#include "tensorflow/lite/experimental/litert/c/litert_accelerator_options.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

struct LiteRtCompilationOptionsT {
  // This should be updated every time a field is added/edited.
  //
  // - Renaming a field: increment patch;
  // - Adding or deprecating a field: set patch to 0, increment minor.
  // - Breaking layout compatibility: set patch and minor to 0, increment major.
  //
  // Note: Changing a default value does not impact the version.
  LiteRtApiVersion version = {.major = 0, .minor = 0, .patch = 1};
  LiteRtHwAcceleratorSet hardware_accelerators = kLiteRtHwAcceleratorNone;
  LiteRtAcceleratorCompilationOptions accelerator_compilation_options = nullptr;
};

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_RUNTIME_COMPILATION_OPTIONS_H_
