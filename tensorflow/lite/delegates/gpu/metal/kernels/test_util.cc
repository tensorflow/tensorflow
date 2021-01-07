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

  InferenceContext::CreateInferenceInfo create_info;
  create_info.precision = CalculationsPrecision::F32;
  create_info.storage_type = TensorStorageType::BUFFER;
  InferenceContext inference_context;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  RETURN_IF_ERROR(inference_context.InitFromGraph(create_info, graph_, device));

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
