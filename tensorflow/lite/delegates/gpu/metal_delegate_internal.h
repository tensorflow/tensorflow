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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_INTERNAL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_INTERNAL_H_

#import <Metal/Metal.h>

#include <functional>

struct TfLiteDelegate;

// Binds Metal buffer to an input or an output tensor in the initialized
// delegate. Bound buffer should have sufficient storage to accommodate all
// elements of a tensor. For quantized model, the buffer is bound to internal
// dequantized float32 tensor.
// Returns non-zero on success, or zero otherwise.
//
// *** Must be called *after* `Interpreter::ModifyGraphWithDelegate`. ***
bool TFLGpuDelegateBindMetalBufferToTensor(TfLiteDelegate* delegate,
                                           int tensor_index,
                                           id<MTLBuffer> metal_buffer);

// Binds user-defined MTLComputeCommandEncoder. The delegate puts all GPU tasks
// into this encoder instead of the internal encoder.
// The callback is a user-defined function to take control over encoder and
// command buffer. Can be nullptr.
bool TFLGpuDelegateSetCommandEncoder(
    TfLiteDelegate* delegate, id<MTLComputeCommandEncoder> encoder,
    std::function<id<MTLComputeCommandEncoder>(bool is_last)> control_encoder);

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_DELEGATE_INTERNAL_H_
