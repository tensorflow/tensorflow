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

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/convert.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/api.h"
#include "tensorflow/lite/delegates/gpu/metal/compiled_model.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"
#include "tensorflow/lite/delegates/gpu/metal/inference_context.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

SingleOpModel::SingleOpModel(Operation&& operation,
                             const std::vector<TensorRef<BHWC>>& inputs,
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
    tensor.id = output->id;
    tensor.shape = output->tensor.shape;
    outputs_.emplace_back(std::move(tensor));
  }
}

absl::Status SingleOpModel::Invoke() {
  std::vector<ValueId> input_ids;
  input_ids.reserve(inputs_.size());
  for (const auto& input : inputs_) {
    input_ids.push_back(input.id);
  }
  std::vector<ValueId> output_ids;
  output_ids.reserve(outputs_.size());
  std::map<ValueId, BHWC> output_dimensions;
  for (const auto& output : outputs_) {
    output_ids.push_back(output.id);
    output_dimensions[output.id] = output.shape;
  }

  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  std::string device_name = std::string([[device name] UTF8String]);
  GpuInfo gpu_info;
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info);
  CalculationsPrecision precision = CalculationsPrecision::F32;
  CompiledModel compiled_model;
  RETURN_IF_ERROR(Compile(graph_, gpu_info, precision, &compiled_model));
  CompiledModel optimized_model;
  RETURN_IF_ERROR(ValidateOptimizeModel(input_ids, output_ids, compiled_model,
                                        &optimized_model));

  InferenceContext inference_context;
  RETURN_IF_ERROR(inference_context.CompileModelWithDevice(
      device, optimized_model, input_ids, output_ids, precision));
  std::map<ValueId, BHWC> input_dimensions;
  std::map<ValueId, id<MTLBuffer>> input_buffers;
  for (auto& input : inputs_) {
    input_dimensions[input.id] = input.shape;
    NSUInteger elements_count = input.shape.w * input.shape.h *
                                AlignByN(input.shape.c, 4) * input.shape.b;
    std::vector<float> src_gpu(elements_count);
    id<MTLBuffer> input_buffer;
    RETURN_IF_ERROR(ConvertToPHWC4(absl::MakeConstSpan(input.data), input.shape,
                                   absl::MakeSpan(src_gpu)));
    input_buffer = [device newBufferWithBytes:src_gpu.data()
                                       length:(elements_count * sizeof(float))
                                      options:MTLResourceStorageModeShared];
    input_buffers[input.id] = input_buffer;
  }

  std::map<ValueId, id<MTLBuffer>> output_buffers;
  for (const auto& outputDimension : output_dimensions) {
    // Uninitialized output buffer.
    const ValueId key = outputDimension.first;
    const BHWC& dims = outputDimension.second;
    const NSUInteger size =
        dims.b * dims.w * dims.h * AlignByN(dims.c, 4) * sizeof(float);
    output_buffers[key] =
        [device newBufferWithLength:size options:MTLResourceStorageModeShared];
  }

  // Inference itself.
  std::map<ValueId, id<MTLBuffer>> inout_buffers(input_buffers.begin(),
                                                 input_buffers.end());
  inout_buffers.insert(output_buffers.begin(), output_buffers.end());
  id<MTLCommandQueue> command_queue = [device newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> command_encoder =
      [command_buffer computeCommandEncoder];
  inference_context.EncodeWithEncoder(command_encoder, inout_buffers);
  [command_encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (auto& output : outputs_) {
    const auto& dim = output_dimensions[output.id];
    NSUInteger elements_count = dim.w * dim.h * AlignByN(dim.c, 4) * dim.b;
    output.shape = dim;
    output.data.resize(output.shape.DimensionsProduct());
    float* output_pointer =
        reinterpret_cast<float*>([output_buffers[output.id] contents]);
    RETURN_IF_ERROR(
        ConvertFromPHWC4(absl::MakeConstSpan(output_pointer, elements_count),
                         output.shape, absl::MakeSpan(output.data)));
  }
  return absl::OkStatus();
}

absl::Status CompareVectors(const std::vector<float>& reference,
                            const std::vector<float>& output, float max_error) {
  if (reference.size() != output.size()) {
    const std::string message =
        "CompareVectors: vectors size does not match for reference: " +
        std::to_string(reference.size()) +
        " vs. output: " + std::to_string(output.size());
    return absl::InternalError(message);
  }
  for (int i = 0; i < reference.size(); i++) {
    float error = std::abs(reference[i] - output[i]);
    if (error > max_error) {
      const std::string message =
          "Reference: " + std::to_string(reference[i]) +
          ", output: " + std::to_string(output[i]) +
          ", error: " + std::to_string(error) +
          ", max allowed error: " + std::to_string(max_error);
      return absl::InternalError(message);
    }
  }
  return absl::OkStatus();
}

absl::Status RunGraph(const std::vector<NodeDescriptor>& nodes,
                      id<MTLDevice> device,
                      const std::map<ValueId, TensorFloat32>& inputs,
                      std::map<ValueId, TensorFloat32>* outputs) {
  std::vector<ValueId> inputBufferIDs;
  inputBufferIDs.reserve(inputs.size());
  for (const auto& input : inputs) {
    inputBufferIDs.push_back(input.first);
  }
  std::vector<ValueId> outputBufferIDs;
  outputBufferIDs.reserve(outputs->size());
  for (const auto& output : *outputs) {
    outputBufferIDs.push_back(output.first);
  }
  std::map<ValueId, BHWC> outputDimensions;
  CompiledModel raw_model;
  raw_model.nodes = nodes;
  for (const auto& input : inputs) {
    raw_model.tensor_shapes[input.first] = input.second.shape;
  }
  for (const auto& output : *outputs) {
    outputDimensions[output.first] = output.second.shape;
    raw_model.tensor_shapes[output.first] = output.second.shape;
  }
  CompiledModel optimized_model;
  RETURN_IF_ERROR(ValidateOptimizeModel(inputBufferIDs, outputBufferIDs,
                                        raw_model, &optimized_model));

  CalculationsPrecision precision = CalculationsPrecision::F32;

  InferenceContext inference_context;
  RETURN_IF_ERROR(inference_context.CompileModelWithDevice(
      device, optimized_model, inputBufferIDs, outputBufferIDs, precision));
  std::map<ValueId, BHWC> inputDimensions;
  std::map<ValueId, std::vector<float>> inputBuffersCPU;
  std::map<ValueId, id<MTLBuffer>> inputBuffersGPU;
  for (auto& input : inputs) {
    const auto& src = input.second;
    inputDimensions[input.first] = src.shape;
    const int paddedDepth = AlignByN(src.shape.c, 4);
    NSUInteger elementsCount =
        src.shape.w * src.shape.h * paddedDepth * src.shape.b;
    std::vector<float> src_gpu(elementsCount);
    id<MTLBuffer> inputBuffer;
    RETURN_IF_ERROR(ConvertToPHWC4(absl::MakeConstSpan(src.data), src.shape,
                                   absl::MakeSpan(src_gpu)));
    inputBuffer = [device newBufferWithBytes:src_gpu.data()
                                      length:(elementsCount * sizeof(float))
                                     options:MTLResourceStorageModeShared];
    inputBuffersGPU[input.first] = inputBuffer;
  }

  std::map<ValueId, id<MTLBuffer>> outputBuffers;
  for (const auto& outputDimension : outputDimensions) {
    // Uninitialized output buffer.
    const ValueId key = outputDimension.first;
    const BHWC& dims = outputDimension.second;
    const NSUInteger outputDataSize =
        dims.b * dims.w * dims.h * AlignByN(dims.c, 4) * sizeof(float);
    outputBuffers[key] =
        [device newBufferWithLength:outputDataSize
                            options:MTLResourceStorageModeShared];
  }

  // Inference itself.
  std::map<ValueId, id<MTLBuffer>> inputOutputBuffers(inputBuffersGPU.begin(),
                                                      inputBuffersGPU.end());
  inputOutputBuffers.insert(outputBuffers.begin(), outputBuffers.end());
  id<MTLCommandQueue> commandQueue = [device newCommandQueue];
  id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  id<MTLComputeCommandEncoder> commandEncoder =
      [commandBuffer computeCommandEncoder];
  inference_context.EncodeWithEncoder(commandEncoder, inputOutputBuffers);
  [commandEncoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];

  for (auto& output : *outputs) {
    const auto& dim = outputDimensions[output.first];
    const int paddedDepth = AlignByN(dim.c, 4);
    NSUInteger elementsCount = dim.w * dim.h * paddedDepth * dim.b;
    auto& dst = output.second;
    dst.shape = dim;
    dst.data.resize(dst.shape.DimensionsProduct());
    float* outputPointer =
        reinterpret_cast<float*>([outputBuffers[output.first] contents]);
    RETURN_IF_ERROR(
        ConvertFromPHWC4(absl::MakeConstSpan(outputPointer, elementsCount),
                         dst.shape, absl::MakeSpan(dst.data)));
  }

  return absl::OkStatus();
}

MetalExecutionEnvironment::MetalExecutionEnvironment() {
  device_ = MTLCreateSystemDefaultDevice();
  std::string device_name = std::string([[device_ name] UTF8String]);
  GetGpuInfoFromDeviceDescription(device_name, GpuApi::kMetal, &gpu_info_);
}

std::vector<CalculationsPrecision>
MetalExecutionEnvironment::GetSupportedPrecisions() const {
  return {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
          CalculationsPrecision::F16};
}

std::vector<TensorStorageType> MetalExecutionEnvironment::GetSupportedStorages()
    const {
  return {TensorStorageType::BUFFER};
}

// returns storage types that support zero clamping when reading OOB in HW
// (Height/Width) dimensions.
std::vector<TensorStorageType>
MetalExecutionEnvironment::GetSupportedStoragesWithHWZeroClampSupport() const {
  return {};
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<ComputeTaskDescriptor>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu) {
  const OperationDef op_def = operation->definition;
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(
        CreateTensor(device_, src_shape, op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(src_cpu[i]));
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(
        CreateTensor(device_, dst_shape, op_def.dst_tensors[i], &dst[i]));
  }

  std::map<ValueId, BHWC> tensor_shapes;
  NodeDescriptor metal_node;
  metal_node.task = std::move(operation);
  metal_node.src_tensors_ids.resize(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    metal_node.src_tensors_ids[i] = i;
    tensor_shapes[i] = src_cpu[i].shape;
  }
  metal_node.dst_tensors_ids.resize(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    metal_node.dst_tensors_ids[i] = src_cpu.size() + i;
    tensor_shapes[src_cpu.size() + i] = dst_sizes[i];
  }
  metal_node.description = "test_op";
  metal_node.id = 0;

  std::string buffer_declarations;
  int index = 0;
  for (int i = 0; i < metal_node.task->dst_tensors_names.size(); ++i) {
    buffer_declarations += metal_node.task->dst_tensors_names[i] + "[[buffer(" +
                           std::to_string(index) + ")]],\n";
    index++;
  }
  for (int i = 0; i < metal_node.task->src_tensors_names.size(); ++i) {
    buffer_declarations += metal_node.task->src_tensors_names[i] + "[[buffer(" +
                           std::to_string(index) + ")]],\n";
    index++;
  }
  for (const auto& buffer : metal_node.task->immutable_buffers) {
    buffer_declarations +=
        buffer.declaration + "[[buffer(" + std::to_string(index) + ")]],\n";
    index++;
  }
  for (const auto& buffer : metal_node.task->uniform_buffers) {
    buffer_declarations +=
        buffer.declaration + "[[buffer(" + std::to_string(index) + ")]],\n";
    index++;
  }

  metal_node.task->shader_source = absl::Substitute(
      metal_node.task->shader_source, "$0", buffer_declarations + "$1", "");

  ComputeTask gpu_task;
  RETURN_IF_ERROR(
      gpu_task.CompileWithDevice(device_, metal_node, op_def.precision));
  RETURN_IF_ERROR(gpu_task.UpdateParamsWithDevice(device_, tensor_shapes));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(dst[i], i);
  }

  id<MTLCommandQueue> command_queue = [device_ newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  gpu_task.EncodeWithEncoder(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(dst_cpu[i]));
  }

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
