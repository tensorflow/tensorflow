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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_

#include <cstddef>

#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_tensor_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

LITE_RT_DEFINE_HANDLE(LrtTensorBufferRequirements);

LrtStatus LrtCreateTensorBufferRequirements(
    int num_supported_tensor_buffer_types,
    const LrtTensorBufferType* supported_tensor_buffer_types,
    size_t buffer_size, LrtTensorBufferRequirements* requirements);

LrtStatus LrtGetTensorBufferRequirementsNumSupportedTensorBufferTypes(
    LrtTensorBufferRequirements requirements, int* num_types);

LrtStatus LrtGetTensorBufferRequirementsSupportedTensorBufferType(
    LrtTensorBufferRequirements requirements, int type_index,
    LrtTensorBufferType* type);

LrtStatus LrtGetTensorBufferRequirementsBufferSize(
    LrtTensorBufferRequirements requirements, size_t* buffer_size);

void LrtDestroyTensorBufferRequirements(
    LrtTensorBufferRequirements requirements);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_C_LITE_RT_TENSOR_BUFFER_REQUIREMENTS_H_
