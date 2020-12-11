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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_

#import <Metal/Metal.h>

#include <list>
#include <map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

/// Stages of model preprocessing:
/// 1. Operations' initialization. All operations are initialized and added into
///    model. Every operation is represented as a vector of
///    ComputeTaskDescriptors.
/// 2. Model compilation. Global list of ComputeTaskDescriptors is transformed
///    into the sorted list of sets of descriptors. A set can be transformed
///    later into a single GPU task.
/// 3. GPU compute tasks generation. Shader code generation happens here.
/// 4. Intermediate resource allocation.
/// Inference.
@interface TFLInferenceContext : NSObject

/// Compiles model: groups operations to be fused; validates model structure.
/// @param device Used to create resources: shaders, buffers. Also the device is used in
///             consecutive call setInputDimensions().
/// @param model Contains ordered vector of shader programs ready to be compiled for GPU and
///             with all supplementary buffers data.
/// @param inputBufferIDs IDs must match the input of added operations.
/// @param outputBufferIDs IDs must match the output of added operations.
/// @param runtimeOptions Options are used to specify data/calculations precision.
/// @return Status signals whether model is compiled successfully or not.
/// @discussion Previously added operations are distilled into sorted list of sets of
///             ComputeTaskDescriptors, which can be fused into a single GPU task.
- (absl::Status)compileModelWithDevice:(id<MTLDevice>)device
                                 model:(const tflite::gpu::metal::CompiledModel&)compiledModel
                        inputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)inputBufferIDs
                       outputBufferIDs:(const std::vector<tflite::gpu::ValueId>&)outputBufferIDs
                             precision:(tflite::gpu::CalculationsPrecision)precision;

/// Inserts all GPU compute tasks into the command encoder.
/// @param inputOutputBuffers Must be created and passed into the method with pairs ID:buffer
/// @discussion No GPU synchronization functions are used inside. All GPU resources must be created
///             with the same device which has been used in compileModelWithDevice() method.
- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
       inputOutputBuffers:
           (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers;

/// Inserts all GPU compute tasks into the command buffer. For every task will be used separate
///   encoder.
/// @param inputOutputBuffers Must be created and passed into the method with pairs ID:buffer
/// @discussion No GPU synchronization functions are used inside. All GPU resources must be created
///             with the same device which has been used in compileModelWithDevice() method.
- (void)encodeWithCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
             inputOutputBuffers:
                 (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers;

/// Adds all GPU compute tasks to the command queue. For every task will be used separate
///   encoder. Few encoders(flushPeriod) batched into compute buffer that sent for execution.
/// @param inputOutputBuffers Must be created and passed into the method with pairs ID:buffer
/// @discussion No GPU synchronization functions are used inside. All GPU resources must be created
///             with the same device which has been used in compileModelWithDevice() method.
- (void)encodeWithCommandQueue:(id<MTLCommandQueue>)commandQueue
            inputOutputBuffers:
                (const std::map<::tflite::gpu::ValueId, id<MTLBuffer>>&)inputOutputBuffers
             flushPeriodically:(int)flushPeriod;

@end

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_INFERENCE_CONTEXT_H_
