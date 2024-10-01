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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Declares canonical opaque type.
#define LITE_RT_DEFINE_HANDLE(name) typedef struct name##T* name
// Declares an array of references to opaque type. `name` must be
// previously declared opaque type.
#define LITE_RT_DEFINE_HANDLE_ARRAY(name) typedef name* name##Array

typedef enum {
  kLrtStatusOk = 0,

  // Generic errors.
  kLrtStatusErrorInvalidArgument = 1,
  kLrtStatusErrorMemoryAllocationFailure = 2,
  kLrtStatusErrorRuntimeFailure = 3,
  kLrtStatusErrorMissingInputTensor = 4,
  kLrtStatusErrorUnsupported = 5,
  kLrtStatusErrorNotFound = 6,

  // File and loading related errors.
  kLrtStatusBadFileOp = 500,
  kLrtStatusFlatbufferFailedVerify = 501,
  kLrtStatusDynamicLoadErr = 502,

  // IR related errors.
  kLrtStatusParamIndexOOB = 1000,
  kLrtStatusBadTensorType = 1001,
  kLrtStatusGraphInvariantError = 1002,
} LrtStatus;

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_COMMON_H_
