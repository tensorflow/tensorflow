/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_FP16_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_FP16_COMMON_H_

// Experimental half precision floating point type compatible with IEEE 754-2008
// binary16 format.

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

#if __GNUC__ && ((__clang__ && (__aarch64__ || __arm__)) || \
                 (!__cplusplus && __ARM_FP16_FORMAT_IEEE))
#define TFL_HAS_IEEE_FP16 1
#endif
#if __GNUC__ && \
    (__clang__ || __ARM_FP16_FORMAT_IEEE || __ARM_FP16_FORMAT_ALTERNATIVE)
#define TFL_HAS_ARM_FP16 1
#endif

namespace tflite {

#if TFL_HAS_IEEE_FP16
typedef _Float16 tfl_float16_t;
#elif TFL_HAS_ARM_FP16
typedef __fp16 tfl_float16_t;
#else
// TODO(b/138252484): implement tfl_float16_t using third_party/FP16
#error "This header requires FP16 support."
#endif

// Check tfl_float16_t is 'compatible' with the placeholder type.
static_assert(sizeof(tfl_float16_t) == sizeof(TfLiteFloat16),
              "Size of real and placeholder FP16 types don't match.");
static_assert(alignof(tfl_float16_t) == alignof(TfLiteFloat16),
              "Alignment of real and placeholder FP16 types don't match.");

// Specialization of typeToTfLiteType with tfl_float16_t.
// Template is declared in interpreter.h
template <>
constexpr TfLiteType typeToTfLiteType<tfl_float16_t>() {
  return kTfLiteFloat16;
}

// Specialization of GetTensorData with tfl_float16_t.
// Template is declared in kernels/internal/tensor_ctypes.h
template <>
inline tfl_float16_t* GetTensorData(TfLiteTensor* tensor) {
  return tensor != nullptr ? reinterpret_cast<tfl_float16_t*>(tensor->data.f16)
                           : nullptr;
}

template <>
inline const tfl_float16_t* GetTensorData(const TfLiteTensor* tensor) {
  return tensor != nullptr
             ? reinterpret_cast<const tfl_float16_t*>(tensor->data.f16)
             : nullptr;
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_KERNELS_FP16_COMMON_H_
