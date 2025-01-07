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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_OPTIONS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_OPTIONS_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The compilation options for the LiteRtCompiledModel.
// WARNING: This is an experimental and subject to change.
// TODO: b/379317134 - Add GPU support.
typedef enum LiteRtComplicationOptions : int {
  kHwAccelDefault = 0,
  kHwAccelCpu = 1 << 0,
  kHwAccelNpu = 1 << 1,
} LiteRtComplicationOptions;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_C_LITERT_COMPILED_MODEL_OPTIONS_H_
