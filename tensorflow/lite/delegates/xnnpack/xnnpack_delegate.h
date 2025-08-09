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

#include <stddef.h>
#include <stdint.h>

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
// Enable XNNPACK acceleration for VAR_HANDLE, READ_VARIABLE, and
// ASSIGN_VARIABLE operators.
#define TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS 0x00000010
// Enable transient indirection buffer to reduce memory usage in selected
// operators. Indirection buffer initialization will take place on every
// inference run, instead of only once during initialization of the operators.
#define TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER 0x00000020
// Enable the latest XNNPACK operators and features in the delegate which have
// not yet been enabled by default.
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS 0x00000040
// Enable XNNPack subgraph reshaping. This means that models with dynamic
// tensors are supported and that inputs may be efficiently resized.
#define TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING 0x00000080
// This flag indicates that XNNPACK should attempt to produce numerically
// consistent results from a specific build of XNNPACK. This causes XNNPACK
// to avoid using faster codepaths that are numerically inconsistent with any
// other codepath that could be used in the same compiled delegate.
#define TFLITE_XNNPACK_DELEGATE_FLAG_SLOW_CONSISTENT_ARITHMETIC 0x00000200
// Disable XNNPack subgraph reshaping. This means that models with dynamic
// tensors are not supported.
#define TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_SUBGRAPH_RESHAPING 0x00000400

struct TfLiteXNNPackDelegateWeightsCache;

typedef struct {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;
  // Flags to pass to `xnn_create_runtime`
  uint32_t runtime_flags;
  // Bitfield with any combination of the following binary options:
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QS8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_QU8
  // - TFLITE_XNNPACK_DELEGATE_FLAG_FORCE_FP16
  // - TFLITE_XNNPACK_DELEGATE_FLAG_DYNAMIC_FULLY_CONNECTED
  // - TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS
  // - TFLITE_XNNPACK_DELEGATE_FLAG_TRANSIENT_INDIRECTION_BUFFER
  // - TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS
  // - TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_SUBGRAPH_RESHAPING
  // - TFLITE_XNNPACK_DELEGATE_FLAG_DISABLE_SUBGRAPH_RESHAPING
  // - TFLITE_XNNPACK_DELEGATE_FLAG_SLOW_CONSISTENT_ARITHMETIC
  uint32_t flags;
  // Cache for packed weights, can be shared between multiple instances of
  // delegates.
  struct TfLiteXNNPackDelegateWeightsCache* weights_cache;
  // Deprecated. Use the flags bitfield with the
  // TFLITE_XNNPACK_DELEGATE_FLAG_VARIABLE_OPERATORS mask.
  bool handle_variable_ops;
  // Path to the weight cache to load.
  //
  // Note: To keep backwards compatibility with the previous caching mechanism,
  // the weight cache will only be loaded from this if `weights_cache` is
  // undefined.
  const char* weight_cache_file_path;
  // Explicit file descriptor for the weight cache.
  //
  // Warning: This will override opening the file from `weight_cache_file_path`.
  //
  // Warning: Because value initialization of a C structure will initialize this
  // field to 0, we cannot accept a file descriptor with that value to remain
  // compatible with existing code. Hopefully this won't cause issues as the
  // file descriptor 0 is usually a special one.
  //
  // Warning: Ownership of the file descriptor is taken by the XNNPack delegate
  // weight cache. `dup` it if you want to keep it open for longer.
  //
  // Note: To keep backwards compatibility with the previous caching mechanism,
  // the weight cache will only be loaded from this if `weights_cache` is
  // undefined.
  int weight_cache_file_descriptor;
  // Points to an existing instance of a weight cache provider.
  //
  // Warning: Ownership of the cache provider is **NOT** taken by the XNNPack
  // delegate.
  //
  // Warning: This will override opening the file from `weight_cache_file_path`
  // and `weight_cache_file_descriptor`.
  //
  // Note: To keep backwards compatibility with the previous caching mechanism,
  // the weight cache will only be loaded from this if `weights_cache` is
  // undefined.
  void* weight_cache_provider;
} TfLiteXNNPackDelegateOptions;

// Returns true on systems that support running the in-memory weight cache
// provider.
TFL_CAPI_EXPORT bool TfLiteXNNPackDelegateCanUseInMemoryWeightCacheProvider();

// Returns a file path that will activate the in-memory weight cache that
// enables weight deduplication.
TFL_CAPI_EXPORT const char* TfLiteXNNPackDelegateInMemoryFilePath();

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

// Returns the options in the delegate.
// Returns NULL if the delegate is NULL.
//
// WARNING: This API is experimental and subject to change.
TFL_CAPI_EXPORT const TfLiteXNNPackDelegateOptions*
TfLiteXNNPackDelegateGetOptions(TfLiteDelegate* delegate);

// Returns the flags used for an XNNPack delegate.
// See documentation for TfLiteXNNPackDelegateOptions.flags.
//
// WARNING: This API is experimental and subject to change.
TFL_CAPI_EXPORT int TfLiteXNNPackDelegateGetFlags(TfLiteDelegate* delegate);

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
