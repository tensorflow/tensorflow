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

#include "tensorflow/lite/core/c/common.h"

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
// Enable XNNPACK acceleration for FULLY_CONNECTED operator with dynamic
// weights.
#define TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED 0x00000008

struct TfLiteXNNPackDelegateWeightsCache;

typedef struct {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;
  // Bitfield with any combination of the following binary options:
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QS8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QU8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
  // - TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED
  uint32_t flags;
  // Cache for packed weights, can be shared between multiple instances of
  // delegates.
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
  // Whether READ_VARIABLE, ASSIGN_VARIABLE, and VARIABLE_HANDLE operations
  // should be handled by XNNPACK.
  bool handle_variable_ops;
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

// Performs the same task as TfLiteXNNPackDelegateCreate, with one exception.
// If the context passed contains a non-null xnnpack_threadpool field,
// we will use it as the threadpool for the delegate created.
TfLiteDelegate* TfLiteXNNPackDelegateCreateWithThreadpool(
    const TfLiteXNNPackDelegateOptions* options, TfLiteContext* context);

// Returns the pthreadpool_t object used for parallelization in XNNPACK.
// Can return NULL if the XNNPack delegate is single-threaded.
//
// WARNING: This API is experimental and subject to change.
TFL_CAPI_EXPORT void* TfLiteXNNPackDelegateGetThreadPool(
    TfLiteDelegate* delegate);

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
TFL_CAPI_EXPORT void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);

// Creates a new weights cache that can be shared with multiple delegate
// instances. Prefer TfLiteXNNPackDelegateWeightsCacheCreateWithSize which can
// reduce memory bandwidth.
TFL_CAPI_EXPORT struct TfLiteXNNPackDelegateWeightsCache*
TfLiteXNNPackDelegateWeightsCacheCreate();
// Creates a new weights cache with a specified initial size that can be shared
// with multiple delegate instances. The weights cache can hold up to size bytes
// without growing.
TFL_CAPI_EXPORT struct TfLiteXNNPackDelegateWeightsCache*
TfLiteXNNPackDelegateWeightsCacheCreateWithSize(size_t size);
// Soft-finalize a weights cache. Extra space will be left in the weights cache
// to allow for cache "insertion" only if it is a cache hit. This has memory
// overhead compared to TfLiteXNNPackDelegateWeightsCacheFinalizeHard. Use this
// if the number of interpreter instances using XNNPACK delegate is not fixed
// (e.g. created based on workload in a server daemon).
// Returns true on success, false on error.
TFL_CAPI_EXPORT bool TfLiteXNNPackDelegateWeightsCacheFinalizeSoft(
    struct TfLiteXNNPackDelegateWeightsCache* cache);
// Hard-finalize a weights cache, cache is effectively frozen and no more cache
// operations are allowed. Memory is resized to smallest possible. Use this if
// the number of interpreter instances using XNNPACK delegate can be fixed and
// all creation of instances can happen up front. This has the lowest memory
// usage.
// Returns true on success, false on error.
TFL_CAPI_EXPORT bool TfLiteXNNPackDelegateWeightsCacheFinalizeHard(
    struct TfLiteXNNPackDelegateWeightsCache* cache);
// Destroys a weights cache created with
// `TfLiteXNNPackDelegateWeightsCacheCreate` call.
TFL_CAPI_EXPORT void TfLiteXNNPackDelegateWeightsCacheDelete(
    struct TfLiteXNNPackDelegateWeightsCache* cache);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
