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

#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"

#include <map>
#include <vector>

#include "third_party/absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

using ::tflite::gpu::BHWC;
using ::tflite::gpu::metal::ComputeTaskDescriptorPtr;
using ::tflite::gpu::metal::RuntimeOptions;
using ::tflite::gpu::OkStatus;
using ::tflite::gpu::Status;
using ::tflite::gpu::ValueId;

@implementation TFLInferenceContext {
  std::vector<TFLComputeTask*> _computeTasks;
  std::vector<ValueId> _outputIds;
  id<MTLDevice> _device;
  RuntimeOptions _options;
}

- (Status)compileModelWithDevice:(id<MTLDevice>)device
                 taskDescriptors:(const std::vector<ComputeTaskDescriptorPtr>&)taskDescriptors
                 outputBufferIDs:(const std::vector<ValueId>&)requestedOutputBufferIDs
                  runtimeOptions:(const RuntimeOptions&)options {
  _device = device;
  _outputIds = requestedOutputBufferIDs;
  _options = options;
  // Metal resources are created here.
  for (const auto& node : taskDescriptors) {
    TFLComputeTask* task = [[TFLComputeTask alloc] init];
    RETURN_IF_ERROR([task compileWithDevice:_device taskDescriptor:node runtimeOptions:_options]);
    _computeTasks.emplace_back(task);
  }
  return OkStatus();
}

- (Status)setInputDimensions:(const std::map<ValueId, BHWC>&)inputDimensions
            outputDimensions:(std::map<ValueId, BHWC>*)outputDimensions {
  // This maps contain all input/output/intermediate buffers shared across model.
  std::map<ValueId, BHWC> dimensions = inputDimensions;
  std::map<ValueId, id<MTLBuffer>> buffers;
  // Insert uninitialized input buffers. This buffers will be set externally.
  for (auto dimension : dimensions) {
    buffers[dimension.first] = nil;
  }
  for (auto& task : _computeTasks) {
    // The same device must be used here as well as on shader compilation stage.
    RETURN_IF_ERROR([task setInputDimensionsWithDevice:_device
                                             outputIDs:_outputIds
                                        runtimeOptions:_options
                                            dimensions:&dimensions
                                               buffers:&buffers]);
  }
  for (auto id : _outputIds) {
    (*outputDimensions)[id] = dimensions[id];
  }
  return OkStatus();
}

- (void)encodeWithEncoder:(id<MTLComputeCommandEncoder>)commandEncoder
       inputOutputBuffers:(const std::map<ValueId, id<MTLBuffer>>&)inputOutputBuffers
             encoderBlock:(id<MTLComputeCommandEncoder> (^)(bool isLast))encoderBlock {
  for (int i = 0; i < _computeTasks.size(); i++) {
    auto& task = _computeTasks[i];
    [task encodeWithEncoder:commandEncoder inputOutputBuffers:inputOutputBuffers];
    if (encoderBlock != nil) {
      commandEncoder = encoderBlock(i == _computeTasks.size() - 1);
    }
  }
}

@end
