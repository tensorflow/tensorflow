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

#include <stdint.h>

#include "tensorflow/lite/c/c_api_internal.h"

// Creates a new delegate instance that need to be destroyed with
// DeleteFlowDelegate when delegate is no longer used by tflite.
struct GpuDelegateOptions {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss;

  enum class WaitType {
    // waitUntilCompleted
    kPassive,
    // Minimize latency. It uses active spinning instead of mutex and consumes
    // additional CPU resources.
    kActive,
    // Useful when the output is used with GPU pipeline then or if external
    // command encoder is set
    kDoNotWait,
  };
  WaitType wait_type;
};

// Creates a new delegate instance that need to be destroyed with
// `DeleteTfLiteGpuDelegate` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the following default values are used:
// .precision_loss_allowed = false,
// .wait_type = kPassive,
TfLiteDelegate* NewGpuDelegate(const GpuDelegateOptions* options);

// Destroys a delegate created with `NewGpuDelegate` call.
void DeleteGpuDelegate(TfLiteDelegate* delegate);

// Binds Metal buffer to an input or an output tensor in the initialized
// delegate.  Bound buffer should have sufficient storage to accommodate all
// elements of a tensor.  Returns non-zero on success, or zero otherwise.
//
// *** Must be called *before* `Interpreter::ModifyGraphWithDelegate`. ***
bool BindMetalBufferToTensor(TfLiteDelegate* delegate, int tensor_index,
                             id<MTLBuffer> metal_buffer);

// Binds user-defined MTLComputeCommandEncoder. The delegate puts all GPU tasks
// into this encoder instead of the internal encoder.
bool SetCommandEncoder(TfLiteDelegate* delegate,
                       id<MTLComputeCommandEncoder> encoder);

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_H_
