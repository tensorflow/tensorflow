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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_H_

#import <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct TfLiteDelegate TfLiteDelegate;

typedef enum {
  // waitUntilCompleted
  TFLGpuDelegateWaitTypePassive,
  // Minimize latency. It uses active spinning instead of mutex and consumes
  // additional CPU resources.
  TFLGpuDelegateWaitTypeActive,
  // Useful when the output is used with GPU pipeline then or if external
  // command encoder is set.
  TFLGpuDelegateWaitTypeDoNotWait,
  // Tries to avoid GPU sleep mode.
  TFLGpuDelegateWaitTypeAggressive,
} TFLGpuDelegateWaitType;

// Creates a new delegate instance that need to be destroyed with
// DeleteFlowDelegate when delegate is no longer used by tflite.
typedef struct {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss;
  TFLGpuDelegateWaitType wait_type;
} TFLGpuDelegateOptions;

// Creates a new delegate instance that need to be destroyed with
// `TFLDeleteTfLiteGpuDelegate` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
// .precision_loss_allowed = false,
// .wait_type = kPassive,
TfLiteDelegate* TFLGpuDelegateCreate(const TFLGpuDelegateOptions* options);

// Destroys a delegate created with `TFLGpuDelegateCreate` call.
void TFLGpuDelegateDelete(TfLiteDelegate* delegate);

// Binds Metal buffer to an input or an output tensor in the initialized
// delegate. Bound buffer should have sufficient storage to accommodate all
// elements of a tensor. Returns non-zero on success, or zero otherwise.
//
// *** Must be called *before* `Interpreter::ModifyGraphWithDelegate`. ***
bool TFLGpuDelegateBindMetalBufferToTensor(TfLiteDelegate* delegate,
                                           int tensor_index,
                                           id<MTLBuffer> metal_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_H_
