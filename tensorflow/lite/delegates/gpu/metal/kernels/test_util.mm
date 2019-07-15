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

#include "tensorflow/lite/delegates/gpu/metal/kernels/test_util.h"

#import <Metal/Metal.h>

#include <map>
#include <vector>

#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/api.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

SingleOpModel::SingleOpModel(Operation&& operation, const std::vector<TensorRef<BHWC>>& inputs,
                             const std::vector<TensorRef<BHWC>>& outputs) {
  auto node = graph_.NewNode();
  node->operation = std::move(operation);

  for (int i = 0; i < inputs.size(); ++i) {
    auto input = graph_.NewValue();
    input->tensor = inputs[i];
    graph_.AddConsumer(node->id, input->id).IgnoreError();
    TensorFloat32 tensor;
    tensor.id = input->tensor.ref;
    tensor.shape = input->tensor.shape;
    inputs_.emplace_back(std::move(tensor));
  }

  for (int i = 0; i < outputs.size(); ++i) {
    auto output = graph_.NewValue();
    output->tensor = outputs[i];
    graph_.SetProducer(node->id, output->id).IgnoreError();
    TensorFloat32 tensor;
    tensor.id = output->tensor.ref;
    tensor.shape = output->tensor.shape;
    outputs_.emplace_back(std::move(tensor));
  }
}

Status SingleOpModel::Invoke() {
  std::vector<ValueId> input_ids;
  input_ids.reserve(inputs_.size());
  for (const auto& input : inputs_) {
    input_ids.push_back(input.id);
  }
  std::vector<ValueId> output_ids;
  output_ids.reserve(outputs_.size());
  for (const auto& output : outputs_) {
    output_ids.push_back(output.id);
  }

  RuntimeOptions options;
  options.storage_precision = RuntimeOptions::Precision::FP32;
  options.accumulator_precision = RuntimeOptions::Precision::FP32;
  CompiledModel compiled_model;
  RETURN_IF_ERROR(Compile(graph_, options, &compiled_model));
  std::string err = "res: ";
  CompiledModel optimized_model;
  RETURN_IF_ERROR(ValidateOptimizeModel(input_ids, output_ids, compiled_model, &optimized_model));

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  TFLInferenceContext* graph = [[TFLInferenceContext alloc] init];
  RETURN_IF_ERROR([graph compileModelWithDevice:device
                                taskDescriptors:optimized_model
                                outputBufferIDs:output_ids
                                 runtimeOptions:options]);
  std::map<ValueId, BHWC> input_dimensions;
  std::map<ValueId, id<MTLBuffer>> input_buffers;
  for (auto& input : inputs_) {
    input_dimensions[input.id] = input.shape;
    NSUInteger elements_count =
        input.shape.w * input.shape.h * AlignByN(input.shape.c, 4) * input.shape.b;
    std::vector<float> src_gpu(elements_count);
    id<MTLBuffer> input_buffer;
    RETURN_IF_ERROR(
        ConvertToPHWC4(absl::MakeConstSpan(input.data), input.shape, absl::MakeSpan(src_gpu)));
    input_buffer = [device newBufferWithBytes:src_gpu.data()
                                       length:(elements_count * sizeof(float))
                                      options:MTLResourceStorageModeShared];
    input_buffers[input.id] = input_buffer;
  }

  // Allocate internal buffers. Graph is ready to be executed.
  // Fills the output buffer IDs and dimensions.
  std::map<ValueId, BHWC> output_dimensions;
  QCHECK_OK([graph setInputDimensions:input_dimensions
                     outputDimensions:&output_dimensions
                      taskDescriptors:optimized_model]);

  std::map<ValueId, id<MTLBuffer>> output_buffers;
  for (const auto& outputDimension : output_dimensions) {
    // Uninitialized output buffer.
    const ValueId key = outputDimension.first;
    const BHWC& dims = outputDimension.second;
    const NSUInteger size = dims.b * dims.w * dims.h * AlignByN(dims.c, 4) * sizeof(float);
    output_buffers[key] = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
  }

  // Inference itself.
  std::map<ValueId, id<MTLBuffer>> inout_buffers(input_buffers.begin(), input_buffers.end());
  inout_buffers.insert(output_buffers.begin(), output_buffers.end());
  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> command_encoder = [command_buffer computeCommandEncoder];
  [graph encodeWithEncoder:command_encoder inputOutputBuffers:inout_buffers encoderBlock:nil];
  [command_encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (auto& output : outputs_) {
    const auto& dim = output_dimensions[output.id];
    NSUInteger elements_count = dim.w * dim.h * AlignByN(dim.c, 4) * dim.b;
    output.shape = dim;
    output.data.resize(output.shape.DimensionsProduct());
    float* output_pointer = reinterpret_cast<float*>([output_buffers[output.id] contents]);
    RETURN_IF_ERROR(ConvertFromPHWC4(absl::MakeConstSpan(output_pointer, elements_count),
                                     output.shape, absl::MakeSpan(output.data)));
  }
  return OkStatus();
}

Status CompareVectors(const std::vector<float>& reference, const std::vector<float>& output,
                      float max_error) {
  if (reference.size() != output.size()) {
    const std::string message = "CompareVectors: vectors size does not match for reference: " +
                          std::to_string(reference.size()) +
                          " vs. output: " + std::to_string(output.size());
    return tflite::gpu::InternalError(message);
  }
  for (int i = 0; i < reference.size(); i++) {
    float error = std::abs(reference[i] - output[i]);
    if (error > max_error) {
      const std::string message =
          "Reference: " + std::to_string(reference[i]) + ", output: " + std::to_string(output[i]) +
          ", error: " + std::to_string(error) + ", max allowed error: " + std::to_string(max_error);
      return tflite::gpu::InternalError(message);
    }
  }
  return OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
