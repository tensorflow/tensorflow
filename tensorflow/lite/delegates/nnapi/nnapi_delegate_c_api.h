/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_C_API_H_
#define TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_C_API_H_

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Use TfLiteNnapiDelegateOptionsDefault() for Default options.
// WARNING: This is an experimental API and subject to change.
struct TFL_CAPI_EXPORT TfLiteNnapiDelegateOptions {
  // Preferred Power/perf trade-off. For more details please see
  // ANeuralNetworksCompilation_setPreference documentation in :
  // https://developer.android.com/ndk/reference/group/neural-networks.html
  enum ExecutionPreference {
    kUndefined = -1,
    kLowPower = 0,
    kFastSingleAnswer = 1,
    kSustainedSpeed = 2,
  };

  // Preferred Power/perf trade-off. Default to kUndefined.
  ExecutionPreference execution_preference;

  // Selected NNAPI accelerator with nul-terminated name.
  // Default to nullptr, which implies the NNAPI default behavior: NNAPI
  // runtime is allowed to use all available accelerators. If the selected
  // accelerator cannot be found, NNAPI will not be used.
  // It is the caller's responsibility to ensure the string is valid for the
  // duration of the Options object lifetime.
  const char* accelerator_name;

  // The nul-terminated cache dir for NNAPI model.
  // Default to nullptr, which implies the NNAPI will not try caching the
  // compilation.
  const char* cache_dir;

  // The unique nul-terminated token string for NNAPI model.
  // Default to nullptr, which implies the NNAPI will not try caching the
  // compilation. It is the caller's responsibility to ensure there is no
  // clash of the tokens.
  // NOTE: when using compilation caching, it is not recommended to use the
  // same delegate instance for multiple models.
  const char* model_token;

  // Whether to disallow NNAPI CPU usage. Default to 1 (true). Only effective on
  // Android 10 and above. The NNAPI CPU typically performs less well than
  // built-in TfLite kernels, but allowing CPU allows partial acceleration of
  // models. If this is set to true, NNAPI is only used if the whole model is
  // accelerated.
  int disallow_nnapi_cpu;

  // Whether to allow fp32 computation to be run in fp16. Default to 0 (false).
  int allow_fp16;

  // Specifies the max number of partitions to delegate. A value <= 0 means
  // no limit. Default to 3.
  // If the delegation of the full set of supported nodes would generate a
  // number of partition greater than this parameter, only
  // <max_number_delegated_partitions> of them will be actually accelerated.
  // The selection is currently done sorting partitions in decreasing order
  // of number of nodes and selecting them until the limit is reached.
  int max_number_delegated_partitions;

  // The pointer to NNAPI support lib implementation. Default to nullptr.
  // If specified, NNAPI delegate will use the support lib instead of NNAPI in
  // Android OS.
  void* nnapi_support_library_handle;
};

// Returns a delegate that uses NNAPI for ops execution.
// Must outlive the interpreter.
// WARNING: This is an experimental API and subject to change.
TfLiteDelegate* TFL_CAPI_EXPORT
TfLiteNnapiDelegateCreate(const TfLiteNnapiDelegateOptions* options);

// Returns TfLiteNnapiDelegateOptions populated with default values.
// WARNING: This is an experimental API and subject to change.
TFL_CAPI_EXPORT TfLiteNnapiDelegateOptions TfLiteNnapiDelegateOptionsDefault();

// Does any needed cleanup and deletes 'delegate'.
// WARNING: This is an experimental API and subject to change.
void TFL_CAPI_EXPORT TfLiteNnapiDelegateDelete(TfLiteDelegate* delegate);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_C_API_H_
