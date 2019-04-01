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

/// Updates dimensions for inputs/outputs/intermediate tensors
- (::tflite::gpu::Status)
    setInputDimensionsWithDevice:(id<MTLDevice>)device
                      dimensions:(std::map<::tflite::gpu::ValueId, ::tflite::gpu::BHWC>*)dimensions;

/// Updates buffers for intermediate tensors only. Returns error if out of memory or a buffer is
/// larger than MTLDevice can support.
/// @param buffers is a map from intermediate tensors' ValueId to metal handles with corresponding
///        buffers.
/// @param outputIDs must match the output of added operations.
/// @param usageRecordIds is a map from intermediate tensors' ValueId to corresponding tensor usage
/// records ids.
/// @param sharedBufferIds contain shared buffer id for each tensor usage record id.
/// @param sharedBuffers contain metal handles to the allocated buffers for each shared buffer id.
/// TODO(ypisarchyk): probably we can decrease the number of parameters here
- (::tflite::gpu::Status)assignBuffers:(std::map<::tflite::gpu::ValueId, id<MTLBuffer>>*)buffers
                             outputIds:(const std::vector<::tflite::gpu::ValueId>&)outputIds
                        usageRecordIds:
                            (const std::map<::tflite::gpu::ValueId, size_t>&)usageRecordIds
                       sharedBufferIds:(const std::vector<size_t>&)sharedBufferIds
                         sharedBuffers:(const std::vector<id<MTLBuffer>>&)sharedBuffers;

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)encoder
       inputOutputBuffers:
           (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers;

@end

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
