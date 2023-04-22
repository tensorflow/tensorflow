/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file declares types used by the pure C inference API defined in c_api.h,
// some of which are also used in the C++ and C kernel and interpreter APIs.

#ifndef TENSORFLOW_LITE_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_C_C_API_TYPES_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Define TFL_CAPI_EXPORT macro to export a function properly with a shared
// library.
#ifdef SWIG
#define TFL_CAPI_EXPORT
#elif defined(TFL_STATIC_LIBRARY_BUILD)
#define TFL_CAPI_EXPORT
#else  // not definded TFL_STATIC_LIBRARY_BUILD
#if defined(_WIN32)
#ifdef TFL_COMPILE_LIBRARY
#define TFL_CAPI_EXPORT __declspec(dllexport)
#else
#define TFL_CAPI_EXPORT __declspec(dllimport)
#endif  // TFL_COMPILE_LIBRARY
#else
#define TFL_CAPI_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32
#endif  // SWIG

typedef enum TfLiteStatus {
  kTfLiteOk = 0,

  // Generally referring to an error in the runtime (i.e. interpreter)
  kTfLiteError = 1,

  // Generally referring to an error from a TfLiteDelegate itself.
  kTfLiteDelegateError = 2,

  // Generally referring to an error in applying a delegate due to
  // incompatibility between runtime and delegate, e.g., this error is returned
  // when trying to apply a TfLite delegate onto a model graph that's already
  // immutable.
  kTfLiteApplicationError = 3,

  // Generally referring to serialized delegate data not being found.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataNotFound = 4,

  // Generally referring to data-writing issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataWriteError = 5,

  // Generally referring to data-reading issues in delegate serialization.
  // See tflite::delegates::Serialization.
  kTfLiteDelegateDataReadError = 6,
} TfLiteStatus;

// Types supported by tensor
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
  kTfLiteComplex128 = 12,
  kTfLiteUInt64 = 13,
  kTfLiteResource = 14,
  kTfLiteVariant = 15,
  kTfLiteUInt32 = 16,
} TfLiteType;

// Legacy. Will be deprecated in favor of TfLiteAffineQuantization.
// If per-layer quantization is specified this field will still be populated in
// addition to TfLiteAffineQuantization.
// Parameters for asymmetric quantization. Quantized values can be converted
// back to float using:
//     real_value = scale * (quantized_value - zero_point)
typedef struct TfLiteQuantizationParams {
  float scale;
  int32_t zero_point;
} TfLiteQuantizationParams;

#ifdef __cplusplus
}  // extern C
#endif
#endif  // TENSORFLOW_LITE_C_C_API_TYPES_H_
