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

namespace {
void CheckInputOutputPrimitivetypeAreValid(const HloInstruction* hlo) {
  // All input and output statistics variables must be F32. Also, the last
  // operand for CudnnBatchNormForwardInference, CudnnBatchNormForwardTraining,
  // and CudnnBatchNormBackward is the feature_index which must be S64.
  // The allowed types for non-statistics variables are as follows:
  // CudnnBatchNormForwardInference:
  //            operand[0]: {half, float}
  //                out[0]: {half, float}
  // CudnnBatchNormForwardTraining:
  //            operand[0]: {half, float}
  //                out[0]: {half, float}
  // CudnnBatchNormBackward:
  //            operand[0]: {half, float}
  //            operand[4]: {half, float}
  //                out[0]: {half, float}
  // Note non-statistics inputs and outputs mentioned above should be of the
  // same type.

  // Check Inputs.
  int64 num_operands = hlo->operand_count();
  PrimitiveType operand_primitive_type =
      hlo->operand(0)->shape().element_type();
  CHECK(operand_primitive_type == F16 || operand_primitive_type == F32)
      << "Not yet implemented";

  for (int i = 1; i < num_operands - 2; i++) {
    if (hlo->custom_call_target() == kCudnnBatchNormBackwardCallTarget &&
        i == 4) {
      // The first operand to batchnorm grad is the input and the 4th operand is
      // the grad_output, both of which can be Eigen::half.
      CHECK_EQ(hlo->operand(i)->shape().element_type(), operand_primitive_type)
          << "Invalid datatype";
      continue;
    }
    CHECK_EQ(hlo->operand(i)->shape().element_type(), F32)
        << "Not yet implemented";
  }

  // The last operand is the feature index which must be int64.
  CHECK_EQ(hlo->operand(num_operands - 1)->shape().element_type(), S64)
      << "Not yet implemented";

  // Check Outputs.
  if (hlo->shape().IsTuple()) {
    CHECK_EQ(hlo->shape().tuple_shapes(0).element_type(),
             operand_primitive_type)
        << "Invalid datatype";

    for (int j = 1; j < hlo->shape().tuple_shapes_size(); j++) {
      CHECK_EQ(hlo->shape().tuple_shapes(j).element_type(), F32)
          << "Not yet implemented";
    }
  } else {
    CHECK_EQ(hlo->shape().element_type(), operand_primitive_type)
        << "Invalid datatype";
  }
}
}  // namespace

CudnnBatchNormForwardInferenceThunk::CudnnBatchNormForwardInferenceThunk(
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& offset,
    const BufferAllocation::Slice& mean,
    const BufferAllocation::Slice& variance, float epsilon, int64 feature_index,
    const BufferAllocation::Slice& output, const HloInstruction* hlo)
    : Thunk(Thunk::Kind::kCudnnBatchNormForwardInference, hlo),
      operand_(operand),
      scale_(scale),
      offset_(offset),
      mean_(mean),
      variance_(variance),
      epsilon_(epsilon),
      feature_index_(feature_index),
      output_(output) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kCustomCall);
  CHECK_EQ(hlo->custom_call_target(),
           kCudnnBatchNormForwardInferenceCallTarget);
  CHECK(
      LayoutUtil::LayoutsInShapesEqual(hlo->shape(), hlo->operand(0)->shape()));
  CheckInputOutputPrimitivetypeAreValid(hlo);
}

Status CudnnBatchNormForwardInferenceThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& buffer_allocations = *params.buffer_allocations;
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
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
      hlo_instruction(), operand, output_base, scale, offset, mean, variance,
      epsilon_, feature_index_, &stream));

  if (!stream.ok()) {
    return InternalError("BatchNormalizationForward call failed.");
  }
  return Status::OK();
}

CudnnBatchNormForwardTrainingThunk::CudnnBatchNormForwardTrainingThunk(
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& offset,
    float epsilon, int64 feature_index,
    const BufferAllocation::Slice& output_data,
    const BufferAllocation::Slice& output_mean,
    const BufferAllocation::Slice& output_inv_stddev,
    const BufferAllocation::Slice& output_tuple, const HloInstruction* hlo)
    : Thunk(Thunk::Kind::kCudnnBatchNormForwardTraining, hlo),
      operand_(operand),
      scale_(scale),
      offset_(offset),
      epsilon_(epsilon),
      feature_index_(feature_index),
      output_data_(output_data),
      output_mean_(output_mean),
      output_inv_stddev_(output_inv_stddev),
      output_tuple_(output_tuple) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kCustomCall);
  CHECK_EQ(hlo->custom_call_target(), kCudnnBatchNormForwardTrainingCallTarget);
  CHECK_EQ(hlo->shape().tuple_shapes_size(), 3);
  CHECK(LayoutUtil::LayoutsInShapesEqual(hlo->shape().tuple_shapes(0),
                                         hlo->operand(0)->shape()));
  CheckInputOutputPrimitivetypeAreValid(hlo);
}

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
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  auto& stream = *params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormForwardTraining(
      hlo_instruction(), operand, output_data, output_mean, output_inv_stddev,
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
      epsilon_, feature_index_, &stream));

  // Write the output tuple.
  const int kNumOutputs = 3;
  auto ptrs = absl::make_unique<void*[]>(kNumOutputs);
  ptrs[0] = output_data.opaque();
  ptrs[1] = output_mean.opaque();
  ptrs[2] = output_inv_stddev.opaque();
  se::DeviceMemory<void*> tuple_addr(
      buffer_allocations.GetDeviceAddress(output_tuple_));
  SafeH2DMemcpy(tuple_addr, std::move(ptrs), kNumOutputs, &stream);
  if (!stream.ok()) {
    return InternalError("BatchNormalizationTraining call failed.");
  }
  return Status::OK();
}

CudnnBatchNormBackwardThunk::CudnnBatchNormBackwardThunk(
    const BufferAllocation::Slice& operand,
    const BufferAllocation::Slice& scale, const BufferAllocation::Slice& mean,
    const BufferAllocation::Slice& inv_stddev,
    const BufferAllocation::Slice& grad_output, float epsilon,
    int64 feature_index, const BufferAllocation::Slice& output_grad_data,
    const BufferAllocation::Slice& output_grad_scale,
    const BufferAllocation::Slice& output_grad_offset,
    const BufferAllocation::Slice& output_tuple, const HloInstruction* hlo)
    : Thunk(Thunk::Kind::kCudnnBatchNormBackward, hlo),
      operand_(operand),
      scale_(scale),
      mean_(mean),
      inv_stddev_(inv_stddev),
      grad_output_(grad_output),
      epsilon_(epsilon),
      feature_index_(feature_index),
      output_grad_data_(output_grad_data),
      output_grad_scale_(output_grad_scale),
      output_grad_offset_(output_grad_offset),
      output_tuple_(output_tuple) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kCustomCall);
  CHECK_EQ(hlo->custom_call_target(), kCudnnBatchNormBackwardCallTarget);
  CHECK_EQ(hlo->shape().tuple_shapes_size(), 3);
  CHECK(LayoutUtil::LayoutsInShapesEqual(hlo->shape().tuple_shapes(0),
                                         hlo->operand(0)->shape()));
  CHECK(LayoutUtil::LayoutsInShapesEqual(hlo->shape().tuple_shapes(0),
                                         hlo->operand(4)->shape()));
  CheckInputOutputPrimitivetypeAreValid(hlo);
}

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
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  se::Stream* stream = params.stream;
  TF_RETURN_IF_ERROR(RunCudnnBatchNormBackward(
      hlo_instruction(), operand, output_grad_data, grad_output,
      output_grad_scale, output_grad_offset,
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
      se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(inv_stddev_)),
      epsilon_, feature_index_, stream));

  // Write the output tuple.
  const int kNumOutputs = 3;
  auto ptrs = absl::make_unique<void*[]>(kNumOutputs);
  ptrs[0] = output_grad_data.opaque();
  ptrs[1] = output_grad_scale.opaque();
  ptrs[2] = output_grad_offset.opaque();
  se::DeviceMemory<void*> tuple_addr(
      buffer_allocations.GetDeviceAddress(output_tuple_));
  SafeH2DMemcpy(tuple_addr, std::move(ptrs), kNumOutputs, stream);

  if (!stream->ok()) {
    return InternalError("BatchNormalizationBackward call failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
