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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Enable XNNPACK acceleration for signed quantized 8-bit inference.
// This includes operators with channel-wise quantized weights.
#define TFLITE_XNNPACK_DELEGATE_FLAG_QS8 0x00000001
// Enable XNNPACK acceleration for unsigned quantized 8-bit inference.
#define TFLITE_XNNPACK_DELEGATE_FLAG_QU8 0x00000002
// Force FP16 inference for FP32 operators.
#define TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16 0x00000004

struct TfLiteXNNPackDelegateWeightsCache;

typedef struct {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;
  // Bitfield with any combination of the following binary options:
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QS8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QU8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
  uint32_t flags;
  // Cache for packed weights, can be shared between multiple instances of
  // delegates.
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
} TfLiteXNNPackDelegateOptions;

// Returns a structure with the default XNNPack delegate options.
TFL_CAPI_EXPORT TfLiteXNNPackDelegateOptions
TfLiteXNNPackDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, default values are used (see
// implementation of TfLiteXNNPackDelegateOptionsDefault in the .cc file for
// details).
TFL_CAPI_EXPORT TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options);

// Returns the pthreadpool_t object used for parallelization in XNNPACK.
// Can return NULL if the XNNPack delegate is single-threaded.
//
// WARNING: This API is experimental and subject to change.
TFL_CAPI_EXPORT void* TfLiteXNNPackDelegateGetThreadPool(
    TfLiteDelegate* delegate);

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
TFL_CAPI_EXPORT void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);

// Creates a new weights cache that can be shared with multiple delegate
// instances.
TFL_CAPI_EXPORT struct TfLiteXNNPackDelegateWeightsCache*
TfLiteXNNPackDelegateWeightsCacheCreate();
// Destroys a weights cache created with
// `TfLiteXNNPackDelegateWeightsCacheCreate` call.
TFL_CAPI_EXPORT void TfLiteXNNPackWeightsCacheDelete(
    struct TfLiteXNNPackDelegateWeightsCache* cache);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
