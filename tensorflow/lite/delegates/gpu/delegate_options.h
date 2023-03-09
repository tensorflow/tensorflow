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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_OPTIONS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_OPTIONS_H_

#include <stdint.h>

#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Encapsulated compilation/runtime tradeoffs.
enum TfLiteGpuInferenceUsage {
  // Delegate will be used only once, therefore, bootstrap/init time should
  // be taken into account.
  TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER = 0,

  // Prefer maximizing the throughput. Same delegate will be used repeatedly on
  // multiple inputs.
  TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED = 1,

  // Balance init latency and throughput. This option will result in slightly
  // higher init latency than FAST_SINGLE_ANSWER but should have inference
  // latency closer to SUSTAINED_SPEED.
  TFLITE_GPU_INFERENCE_PREFERENCE_BALANCED = 2,
};

enum TfLiteGpuInferencePriority {
  // AUTO priority is needed when a single priority is the most important
  // factor. For example,
  // priority1 = MIN_LATENCY would result in the configuration that achieves
  // maximum performance.
  TFLITE_GPU_INFERENCE_PRIORITY_AUTO = 0,
  TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION = 1,
  TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY = 2,
  TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE = 3,
};

// Used to toggle experimental flags used in the delegate. Note that this is a
// bitmask, so the values should be 1, 2, 4, 8, ...etc.
enum TfLiteGpuExperimentalFlags {
  TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE = 0,
  // Enables inference on quantized models with the delegate.
  // NOTE: This is enabled in TfLiteGpuDelegateOptionsV2Default.
  TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT = 1 << 0,
  // Enforces execution with the provided backend.
  TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY = 1 << 1,
  TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY = 1 << 2,
  // Enable serialization of GPU kernels & model data. Speeds up initialization
  // at the cost of space on disk.
  // Delegate performs serialization the first time it is applied with a new
  // model or inference params. Later initializations are fast.
  // ModifyGraphWithDelegate will fail if data cannot be serialized.
  //
  // NOTE: User also needs to set serialization_dir & model_token in
  // TfLiteGpuDelegateOptionsV2.
  // Currently works only if CL backend is used.
  TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION = 1 << 3,
};

// IMPORTANT: Always use TfLiteGpuDelegateOptionsV2Default() method to create
// new instance of TfLiteGpuDelegateOptionsV2, otherwise every new added option
// may break inference.
typedef struct {
  // When set to zero, computations are carried out in maximal possible
  // precision. Otherwise, the GPU may quantify tensors, downcast values,
  // process in FP16 to increase performance. For most models precision loss is
  // warranted.
  // [OBSOLETE]: to be removed
  int32_t is_precision_loss_allowed;

  // Preference is defined in TfLiteGpuInferenceUsage.
  int32_t inference_preference;

  // Ordered priorities provide better control over desired semantics,
  // where priority(n) is more important than priority(n+1), therefore,
  // each time inference engine needs to make a decision, it uses
  // ordered priorities to do so.
  // For example:
  //   MAX_PRECISION at priority1 would not allow to decrease precision,
  //   but moving it to priority2 or priority3 would result in F16 calculation.
  //
  // Priority is defined in TfLiteGpuInferencePriority.
  // AUTO priority can only be used when higher priorities are fully specified.
  // For example:
  //   VALID:   priority1 = MIN_LATENCY, priority2 = AUTO, priority3 = AUTO
  //   VALID:   priority1 = MIN_LATENCY, priority2 = MAX_PRECISION,
  //            priority3 = AUTO
  //   INVALID: priority1 = AUTO, priority2 = MIN_LATENCY, priority3 = AUTO
  //   INVALID: priority1 = MIN_LATENCY, priority2 = AUTO,
  //            priority3 = MAX_PRECISION
  // Invalid priorities will result in error.
  int32_t inference_priority1;
  int32_t inference_priority2;
  int32_t inference_priority3;

  // Bitmask flags. See the comments in TfLiteGpuExperimentalFlags.
  int64_t experimental_flags;

  // A graph could have multiple partitions that can be delegated to the GPU.
  // This limits the maximum number of partitions to be delegated. By default,
  // it's set to 1 in TfLiteGpuDelegateOptionsV2Default().
  int32_t max_delegated_partitions;

  // The nul-terminated directory to use for serialization.
  // Whether serialization actually happens or not is dependent on backend used
  // and validity of this directory.
  // Set to nullptr in TfLiteGpuDelegateOptionsV2Default(), which implies the
  // delegate will not try serialization.
  //
  // NOTE: Users should ensure that this directory is private to the app to
  // avoid data access issues.
  const char* serialization_dir;

  // The unique nul-terminated token string that acts as a 'namespace' for
  // all serialization entries.
  // Should be unique to a particular model (graph & constants).
  // For an example of how to generate this from a TFLite model, see
  // StrFingerprint() in lite/delegates/serialization.h.
  //
  // Set to nullptr in TfLiteGpuDelegateOptionsV2Default(), which implies the
  // delegate will not try serialization.
  const char* model_token;
} TfLiteGpuDelegateOptionsV2;

// Populates TfLiteGpuDelegateOptionsV2 as follows:
//   is_precision_loss_allowed = false
//   inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER
//   priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION
//   priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
//   priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO
//   experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_QUANT
//   max_delegated_partitions = 1
TFL_CAPI_EXPORT TfLiteGpuDelegateOptionsV2 TfLiteGpuDelegateOptionsV2Default();

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_DELEGATE_OPTIONS_H_
