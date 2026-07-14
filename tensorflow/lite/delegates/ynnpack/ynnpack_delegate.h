/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_YNNPACK_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_YNNPACK_DELEGATE_H_

#include <memory>

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Number of threads to use.
  int num_threads;
  // If true, the YNNPACK subgraph will be statically shaped. This can sometimes
  // improve performance, at the cost of making resizing the graph much more
  // expensive.
  bool static_shape;
  // If true, enable YNN_FLAG_FAST_MATH.
  bool fast_math;
  // If true, enable YNN_FLAG_CONSISTENT_ARITHMETIC.
  bool consistent_arithmetic;
  // If true, enable YNN_FLAG_NO_EXCESS_PRECISION.
  bool no_excess_precision;
} TfLiteYNNPackDelegateOptions;

// Returns a structure with the default delegate options.
TfLiteYNNPackDelegateOptions TfLiteYNNPackDelegateOptionsDefault();

// Creates a new delegate instance that needs to be destroyed with
// `TfLiteYNNPackDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the default values are used.
TfLiteDelegate* TfLiteYNNPackDelegateCreate(
    const TfLiteYNNPackDelegateOptions* options);

// Destroys a delegate created with `TfLiteYNNPackDelegateCreate` call.
void TfLiteYNNPackDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

// A convenient wrapper that returns C++ std::unique_ptr for automatic memory
// management.
inline std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
TfLiteYNNPackDelegateCreateUnique(const TfLiteYNNPackDelegateOptions* options) {
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteYNNPackDelegateCreate(options), TfLiteYNNPackDelegateDelete);
}

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_YNNPACK_DELEGATE_H_
