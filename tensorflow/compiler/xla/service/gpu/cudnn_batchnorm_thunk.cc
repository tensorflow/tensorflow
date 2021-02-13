/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_thunk.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_runner.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

namespace dnn = se::dnn;

CudnnBatchNormForwardInferenceThunk::CudnnBatchNormForwardInferenceThunk(
    ThunkInfo thunk_info, CudnnBatchNormConfig config,
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& offset,
    const BufferAllocation::Slice& mean,
    const BufferAllocation::Slice& variance,
    const BufferAllocation::Slice& output)
    : Thunk(Thunk::Kind::kCudnnBatchNormForwardInference, thunk_info),
      config_(std::move(config)),
      operand_(operand),
      scale_(scale),
      offset_(offset),
      mean_(mean),
      variance_(variance),
      output_(output) {}

Status CudnnBatchNormForwardInferenceThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());
  se::DeviceMemoryBase output_base =
      buffer_allocations.GetDeviceAddress(output_);
  se::DeviceMemoryBase operand = buffer_allocations.GetDeviceAddress(operand_);
  se::DeviceMemory<float> scale(buffer_allocations.GetDeviceAddress(scale_));
  se::DeviceMemory<float> offset(buffer_allocations.GetDeviceAddress(offset_));
  se::DeviceMemory<float> mean(buffer_allocations.GetDeviceAddress(mean_));
  se::DeviceMemory<float> variance(
      buffer_allocations.GetDeviceAddress(variance_));
  auto& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormForwardInference(
      config_, operand, output_base, scale, offset, mean, variance, &stream));

  if (!stream.ok()) {
    return InternalError("BatchNormalizationForward call failed.");
  }
  return Status::OK();
}

CudnnBatchNormForwardTrainingThunk::CudnnBatchNormForwardTrainingThunk(
    ThunkInfo thunk_info, CudnnBatchNormConfig config,
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& offset,
    const BufferAllocation::Slice& output_data,
    const BufferAllocation::Slice& output_mean,
    const BufferAllocation::Slice& output_inv_stddev)
    : Thunk(Thunk::Kind::kCudnnBatchNormForwardTraining, thunk_info),
      config_(std::move(config)),
      operand_(operand),
      scale_(scale),
      offset_(offset),
      output_data_(output_data),
      output_mean_(output_mean),
      output_inv_stddev_(output_inv_stddev) {}

Status CudnnBatchNormForwardTrainingThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase operand = buffer_allocations.GetDeviceAddress(operand_);
  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_data_);

  se::DeviceMemory<float> output_mean(
      buffer_allocations.GetDeviceAddress(output_mean_));
  se::DeviceMemory<float> output_inv_stddev(
      buffer_allocations.GetDeviceAddress(output_inv_stddev_));

  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());
  auto& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormForwardTraining(
      config_, operand, output_data, output_mean, output_inv_stddev,
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
      &stream));

  if (!stream.ok()) {
    return InternalError("BatchNormalizationTraining call failed.");
  }
  return Status::OK();
}

CudnnBatchNormBackwardThunk::CudnnBatchNormBackwardThunk(
    ThunkInfo thunk_info, CudnnBatchNormConfig config,
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& mean,
    const BufferAllocation::Slice& inv_stddev,
    const BufferAllocation::Slice& grad_output,
    const BufferAllocation::Slice& output_grad_data,
    const BufferAllocation::Slice& output_grad_scale,
    const BufferAllocation::Slice& output_grad_offset)
    : Thunk(Thunk::Kind::kCudnnBatchNormBackward, thunk_info),
      config_(std::move(config)),
      operand_(operand),
      scale_(scale),
      mean_(mean),
      inv_stddev_(inv_stddev),
      grad_output_(grad_output),
      output_grad_data_(output_grad_data),
      output_grad_scale_(output_grad_scale),
      output_grad_offset_(output_grad_offset) {}

Status CudnnBatchNormBackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase operand = buffer_allocations.GetDeviceAddress(operand_);
  se::DeviceMemoryBase output_grad_data =
      buffer_allocations.GetDeviceAddress(output_grad_data_);
  se::DeviceMemoryBase grad_output =
      buffer_allocations.GetDeviceAddress(grad_output_);
  se::DeviceMemory<float> output_grad_scale(
      buffer_allocations.GetDeviceAddress(output_grad_scale_));
  se::DeviceMemory<float> output_grad_offset(
      buffer_allocations.GetDeviceAddress(output_grad_offset_));

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());
  se::Stream* stream = params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormBackward(
      config_, operand, output_grad_data, grad_output, output_grad_scale,
      output_grad_offset,
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(inv_stddev_)),
      stream));

  if (!stream->ok()) {
    return InternalError("BatchNormalizationBackward call failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
