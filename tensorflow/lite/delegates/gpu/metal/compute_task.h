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
#include <set>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

@interface TFLComputeTask : NSObject

/// Returns empty string or error if shader can't be compiled.
- (absl::Status)compileWithDevice:(id<MTLDevice>)device
                   taskDescriptor:(const tflite::gpu::metal::NodeDescriptor&)desc
                        precision:(tflite::gpu::CalculationsPrecision)precision;

/// Updates parameters for inputs/outputs/intermediate tensors
- (absl::Status)updateParamsWithDevice:(id<MTLDevice>)device
                          tensorShapes:(const std::map<tflite::gpu::ValueId, tflite::gpu::BHWC>&)
                                           tensorShapes;

- (bool)hasInOutIds:(const std::set<::tflite::gpu::ValueId>&)ids;

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)encoder;

- (std::vector<tflite::gpu::ValueId>)getOutputIds;
- (std::vector<tflite::gpu::ValueId>)getInputIds;

- (void)setSrcTensor:(const tflite::gpu::metal::MetalSpatialTensor&)tensor withIndex:(int)index;

- (void)setDstTensor:(const tflite::gpu::metal::MetalSpatialTensor&)tensor withIndex:(int)index;

- (void)setDescription:(const std::string&)description;

@end

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_COMPUTE_TASK_H_
