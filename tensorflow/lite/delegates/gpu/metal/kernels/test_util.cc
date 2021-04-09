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
#include "tensorflow/lite/delegates/gpu/common/precision.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/compute_task.h"
#include "tensorflow/lite/delegates/gpu/metal/metal_spatial_tensor.h"

namespace tflite {
namespace gpu {
namespace metal {

std::vector<CalculationsPrecision>
MetalExecutionEnvironment::GetSupportedPrecisions() const {
  return {CalculationsPrecision::F32, CalculationsPrecision::F32_F16,
          CalculationsPrecision::F16};
}

std::vector<TensorStorageType> MetalExecutionEnvironment::GetSupportedStorages()
    const {
  return {TensorStorageType::BUFFER, TensorStorageType::IMAGE_BUFFER,
          TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_3D,
          TensorStorageType::TEXTURE_ARRAY};
}

// returns storage types that support zero clamping when reading OOB in HW
// (Height/Width) dimensions.
std::vector<TensorStorageType>
MetalExecutionEnvironment::GetSupportedStoragesWithHWZeroClampSupport() const {
  return {TensorStorageType::TEXTURE_2D, TensorStorageType::TEXTURE_3D,
          TensorStorageType::TEXTURE_ARRAY};
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<TensorFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWC>& dst_sizes,
    const std::vector<TensorFloat32*>& dst_cpu) {
  const OperationDef op_def = operation->GetDefinition();
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(device_.device(), src_cpu[i]));
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));
  }

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&device_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(&src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(&dst[i], i);
  }
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  gpu_task.Encode(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(device_.device(), dst_cpu[i]));
  }

  return absl::OkStatus();
}

absl::Status MetalExecutionEnvironment::ExecuteGPUOperation(
    const std::vector<Tensor5DFloat32>& src_cpu,
    std::unique_ptr<GPUOperation>&& operation,
    const std::vector<BHWDC>& dst_sizes,
    const std::vector<Tensor5DFloat32*>& dst_cpu) {
  const OperationDef op_def = operation->GetDefinition();
  std::vector<MetalSpatialTensor> src(src_cpu.size());
  for (int i = 0; i < src_cpu.size(); ++i) {
    auto src_shape = src_cpu[i].shape;
    if (src_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), src_shape,
                                 op_def.src_tensors[i], &src[i]));
    RETURN_IF_ERROR(src[i].WriteData(device_.device(), src_cpu[i]));
  }

  std::vector<MetalSpatialTensor> dst(dst_cpu.size());
  for (int i = 0; i < dst_cpu.size(); ++i) {
    auto dst_shape = dst_sizes[i];
    if (dst_shape.b != 1 && !op_def.IsBatchSupported()) {
      return absl::InvalidArgumentError(
          "Layout doesn't have Batch dimension, but shape.b != 1");
    }
    RETURN_IF_ERROR(CreateTensor(device_.device(), dst_shape,
                                 op_def.dst_tensors[i], &dst[i]));
  }

  ComputeTask gpu_task;
  gpu_task.Init(std::move(operation));
  RETURN_IF_ERROR(gpu_task.Compile(&device_));
  for (int i = 0; i < src_cpu.size(); ++i) {
    gpu_task.SetSrcTensor(&src[i], i);
  }
  for (int i = 0; i < dst_cpu.size(); ++i) {
    gpu_task.SetDstTensor(&dst[i], i);
  }
  RETURN_IF_ERROR(gpu_task.UpdateParams());

  id<MTLCommandQueue> command_queue = [device_.device() newCommandQueue];
  id<MTLCommandBuffer> command_buffer = [command_queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
  gpu_task.Encode(encoder);
  [encoder endEncoding];
  [command_buffer commit];
  [command_buffer waitUntilCompleted];

  for (int i = 0; i < dst_cpu.size(); ++i) {
    dst_cpu[i]->shape = dst_sizes[i];
    dst_cpu[i]->data = std::vector<float>(dst_sizes[i].DimensionsProduct(), 0);
    RETURN_IF_ERROR(dst[i].ReadData(device_.device(), dst_cpu[i]));
  }

  return absl::OkStatus();
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
