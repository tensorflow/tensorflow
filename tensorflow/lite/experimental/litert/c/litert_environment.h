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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ENVIRONMENT_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ENVIRONMENT_H_

#include "tensorflow/lite/experimental/litert/c/litert_any.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum {
  kLiteRtEnvOptionTagCompilerPluginLibraryPath = 0,
  kLiteRtEnvOptionTagDispatchLibraryPath = 1,
} LiteRtEnvOptionTag;

typedef struct {
  LiteRtEnvOptionTag tag;
  LiteRtAny value;
} LiteRtEnvOption;

// Create a singleton LiteRT environment with options. Returns an error if the
// instance already exists, in which case the specified options have no
// effect. If not created explicitly with options, the environment instance will
// be created (with no options) when needed.
LiteRtStatus LiteRtEnvironmentCreate(int num_options,
                                     const LiteRtEnvOption* options);

// Destroy the LiteRT environment instance.
void LiteRtEnvironmentDestroy();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_ENVIRONMENT_H_
