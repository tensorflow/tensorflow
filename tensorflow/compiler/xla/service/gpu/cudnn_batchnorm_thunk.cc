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
    ThunkInfo thunk_info, CudnnBatchNormForwardTrainingConfig config,
    std::vector<BufferAllocation::Slice> operand_slices,
    std::vector<BufferAllocation::Slice> output_slices)
    : Thunk(Thunk::Kind::kCudnnBatchNormForwardTraining, thunk_info),
      config_(std::move(config)),
      operand_slices_(operand_slices),
      output_slices_(output_slices) {}

Status CudnnBatchNormForwardTrainingThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  CHECK_LE(operand_slices_.size(), 4);
  CHECK_LE(output_slices_.size(), 5);
  se::DeviceMemoryBase operand =
      buffer_allocations.GetDeviceAddress(operand_slices_[0]);
  se::DeviceMemory<float> scale(
      buffer_allocations.GetDeviceAddress(operand_slices_[1]));
  se::DeviceMemory<float> offset(
      buffer_allocations.GetDeviceAddress(operand_slices_[2]));
  bool has_side_input = operand_slices_.size() == 4;
  se::DeviceMemoryBase side_input_base(nullptr);
  if (has_side_input) {
    side_input_base = buffer_allocations.GetDeviceAddress(operand_slices_[3]);
    VLOG(2) << "BatchNorm side input buffer slice: "
            << operand_slices_[3].ToString() << " with size "
            << side_input_base.size();
  }

  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_slices_[0]);
  se::DeviceMemory<float> output_mean(
      buffer_allocations.GetDeviceAddress(output_slices_[1]));
  se::DeviceMemory<float> output_inv_stddev(
      buffer_allocations.GetDeviceAddress(output_slices_[2]));

  bool use_reserve_space = output_slices_.size() == 5;
  se::DeviceMemoryBase reserve_space(nullptr);
  se::DeviceMemoryBase workspace(nullptr);
  if (use_reserve_space) {
    reserve_space = buffer_allocations.GetDeviceAddress(output_slices_[3]);
    VLOG(1) << "DeviceMemory reserve_space BatchNorm Forward - the size, in "
               "bytes, for the backing memory "
            << reserve_space.size();
    VLOG(2) << "BatchNorm forward reserve space buffer slice: "
            << output_slices_[3].ToString();
    VLOG(2) << "Reserve space device address in "
               "CudnnBatchNormForwardTrainingThunk: "
            << reserve_space.opaque();
    workspace = buffer_allocations.GetDeviceAddress(output_slices_[4]);
    VLOG(1) << "DeviceMemory workspace BatchNorm Forward - the size, in "
               "bytes, for the backing memory "
            << workspace.size();
  }
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());
  auto& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormForwardTraining(
      config_, operand, output_data, output_mean, output_inv_stddev, scale,
      offset, side_input_base, reserve_space, workspace, &stream));

  if (!stream.ok()) {
    return InternalError("BatchNormalizationTraining call failed.");
  }
  return Status::OK();
}

CudnnBatchNormBackwardThunk::CudnnBatchNormBackwardThunk(
    ThunkInfo thunk_info, CudnnBatchNormConfig config,
    std::vector<BufferAllocation::Slice> operand_slices,
    std::vector<BufferAllocation::Slice> output_slices)
    : Thunk(Thunk::Kind::kCudnnBatchNormBackward, thunk_info),
      config_(std::move(config)),
      operand_slices_(operand_slices),
      output_slices_(output_slices) {}

Status CudnnBatchNormBackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  CHECK_LE(operand_slices_.size(), 6);
  CHECK_GE(operand_slices_.size(), 5);
  CHECK_LE(output_slices_.size(), 4);
  CHECK_GE(output_slices_.size(), 3);

  // Operand Slices
  se::DeviceMemoryBase operand =
      buffer_allocations.GetDeviceAddress(operand_slices_[0]);
  se::DeviceMemory<float> scale(
      buffer_allocations.GetDeviceAddress(operand_slices_[1]));
  se::DeviceMemory<float> mean(
      buffer_allocations.GetDeviceAddress(operand_slices_[2]));
  se::DeviceMemory<float> inv_stddev(
      buffer_allocations.GetDeviceAddress(operand_slices_[3]));
  se::DeviceMemoryBase grad_output =
      buffer_allocations.GetDeviceAddress(operand_slices_[4]);

  // Output Slices
  se::DeviceMemoryBase output_grad_data =
      buffer_allocations.GetDeviceAddress(output_slices_[0]);
  se::DeviceMemory<float> output_grad_scale(
      buffer_allocations.GetDeviceAddress(output_slices_[1]));
  se::DeviceMemory<float> output_grad_offset(
      buffer_allocations.GetDeviceAddress(output_slices_[2]));

  bool use_reserve_space = operand_slices_.size() == 6;
  se::DeviceMemoryBase reserve_space_base(nullptr);
  se::DeviceMemoryBase workspace(nullptr);
  if (use_reserve_space) {
    reserve_space_base =
        buffer_allocations.GetDeviceAddress(operand_slices_[5]);
    VLOG(1) << "DeviceMemory reserve_space BatchNorm Backward - the size, in "
               "bytes, for the backing memory "
            << reserve_space_base.size();
    workspace = buffer_allocations.GetDeviceAddress(output_slices_[3]);
    VLOG(2) << "BatchNorm backward reserve space buffer slice: "
            << operand_slices_[5].ToString();
  }
  se::DeviceMemory<uint8> reserve_space(reserve_space_base);
  VLOG(2) << "Reserve space device address in CudnnBatchNormBackwardThunk: "
          << reserve_space.opaque();
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(profile_index());
  se::Stream* stream = params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormBackward(
      config_, operand, output_grad_data, grad_output, output_grad_scale,
      output_grad_offset, scale, mean, inv_stddev, reserve_space, workspace,
      stream));

  if (!stream->ok()) {
    return InternalError("BatchNormalizationBackward call failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
