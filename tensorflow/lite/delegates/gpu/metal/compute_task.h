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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_

#import <Metal/Metal.h>

#include <map>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

@interface TFLComputeTask : NSObject

/// Returns empty string or error if shader can't be compiled.
- (::tflite::gpu::Status)compileWithDevice:(id<MTLDevice>)device
                            taskDescriptor:(::tflite::gpu::metal::ComputeTaskDescriptorPtr)desc
                            runtimeOptions:(const ::tflite::gpu::metal::RuntimeOptions&)options;

/// Updates buffers for intermediate tensors only and dimensions for inputs/outputs/intermediate
/// tensors. Returns error if out of memory or a buffer is larger than MTLDevice can support.
- (::tflite::gpu::Status)
    setInputDimensionsWithDevice:(id<MTLDevice>)device
                       outputIDs:(const std::vector<::tflite::gpu::ValueId>&)outputIDs
                  runtimeOptions:(const ::tflite::gpu::metal::RuntimeOptions&)options
                      dimensions:(std::map<::tflite::gpu::ValueId, ::tflite::gpu::BHWC>*)dimensions
                         buffers:(std::map<::tflite::gpu::ValueId, id<MTLBuffer>>*)buffers;

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)encoder
       inputOutputBuffers:
           (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers;

@end

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
