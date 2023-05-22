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

/// WARNING: Users of TensorFlow Lite should not include this file directly,
/// but should instead include
/// "third_party/tensorflow/lite/c/c_api_types.h".
/// Only the TensorFlow Lite implementation itself should include this
/// file directly.
// IWYU pragma: private, include "third_party/tensorflow/lite/c/c_api_types.h"

#ifndef TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_
#define TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_

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

// Note that new error status values may be added in future in order to
// indicate more fine-grained internal states, therefore, applications should
// not rely on status values being members of the enum.
typedef enum TfLiteStatus {
  kTfLiteOk = 0,

  // Generally referring to an error in the runtime (i.e. interpreter)
  kTfLiteError = 1,

  // Generally referring to an error from a TfLiteDelegate itself.
  kTfLiteDelegateError = 2,

  // Generally referring to an error in applying a delegate due to
  // incompatibility between runtime and delegate, e.g., this error is returned
  // when trying to apply a TF Lite delegate onto a model graph that's already
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

  // Generally referring to issues when the TF Lite model has ops that cannot be
  // resolved at runtime. This could happen when the specific op is not
  // registered or built with the TF Lite framework.
  kTfLiteUnresolvedOps = 7,

  // Generally referring to invocation cancelled by the user.
  // See `interpreter::Cancel`.
  // TODO(b/194915839): Implement `interpreter::Cancel`.
  // TODO(b/250636993): Cancellation triggered by `SetCancellationFunction`
  // should also return this status code.
  kTfLiteCancelled = 8,
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
  kTfLiteUInt16 = 17,
  kTfLiteInt4 = 18,
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

// --------------------------------------------------------------------------
// Opaque types used by c_api.h, c_api_opaque.h and common.h.

// TfLiteOpaqueContext is an opaque version of TfLiteContext;
typedef struct TfLiteOpaqueContext TfLiteOpaqueContext;

// TfLiteOpaqueNode is an opaque version of TfLiteNode;
typedef struct TfLiteOpaqueNode TfLiteOpaqueNode;

// TfLiteOpaqueTensor is an opaque version of TfLiteTensor;
typedef struct TfLiteOpaqueTensor TfLiteOpaqueTensor;

// TfLiteDelegate: allows delegation of nodes to alternative backends.
// Forward declaration of concrete type declared in common.h.
typedef struct TfLiteDelegate TfLiteDelegate;

// TfLiteOpaqueDelegateStruct: unconditionally opaque version of
// TfLiteDelegate; allows delegation of nodes to alternative backends.
//
// This is an abstract type that is intended to have the same
// role as TfLiteDelegate, but without exposing the implementation
// details of how delegates are implemented.
// WARNING: This is an experimental type and subject to change.
typedef struct TfLiteOpaqueDelegateStruct TfLiteOpaqueDelegateStruct;

// TfLiteOpaqueDelegate: conditionally opaque version of
// TfLiteDelegate; allows delegation of nodes to alternative backends.
// For TF Lite in Play Services, this is an opaque type,
// but for regular TF Lite, this is just a typedef for TfLiteDelegate.
// WARNING: This is an experimental type and subject to change.
#if TFLITE_WITH_STABLE_ABI || TFLITE_USE_OPAQUE_DELEGATE
typedef TfLiteOpaqueDelegateStruct TfLiteOpaqueDelegate;
#else
typedef TfLiteDelegate TfLiteOpaqueDelegate;
#endif

#ifdef __cplusplus
}  // extern C
#endif
#endif  // TENSORFLOW_LITE_CORE_C_C_API_TYPES_H_
