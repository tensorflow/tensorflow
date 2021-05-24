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

typedef struct {
  // Number of threads to use in the thread pool.
  // 0 or negative value means no thread pool used.
  int32_t num_threads;

  // Whether to enable the support of INT8-weights unpacking in a similar way as
  // FP16-weights unpacking is supported, which requires an additional
  // Dequantize op for the quantized weights.
  // By default, this is disabled. However, when compiling the XNNPACK delegate,
  // defining macro ENABLE_TFLITE_XNNPACK_DEQUANTIZED_INT8_WEIGHTS will enable
  // this feature.
  bool enable_int8_weights_unpacking;
} TfLiteXNNPackDelegateOptions;

// Returns a structure with the default XNNPack delegate options.
TfLiteXNNPackDelegateOptions TfLiteXNNPackDelegateOptionsDefault();

// Creates a new delegate instance that need to be destroyed with
// `TfLiteXNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
TfLiteDelegate* TfLiteXNNPackDelegateCreate(
    const TfLiteXNNPackDelegateOptions* options);

// Returns the pthreadpool_t object used for parallelization in XNNPACK.
// Can return NULL if the XNNPack delegate is single-threaded.
//
// WARNING: This API is experimental and subject to change.
void* TfLiteXNNPackDelegateGetThreadPool(TfLiteDelegate* delegate);

// Destroys a delegate created with `TfLiteXNNPackDelegateCreate` call.
void TfLiteXNNPackDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_XNNPACK_DELEGATE_H_
