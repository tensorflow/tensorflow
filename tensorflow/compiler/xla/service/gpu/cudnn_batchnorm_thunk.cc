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
void CheckInputOutputPrimitivetypeValid(const HloInstruction* hlo) {
  // Check Inputs.
  int64 num_operands = hlo->operand_count();
  PrimitiveType operand_dtype = hlo->operand(0)->shape().element_type();
  CHECK((operand_dtype == F16) || (operand_dtype == F32))
      << "Not yet implemented";

  for (auto i = 1; i < num_operands - 2; i++) {
    if (hlo->custom_call_target() == kCudnnBatchNormBackwardCallTarget &&
        (i == 4)) {
      // The first operand to batchnorm grad is the input and the 4th operand is
      // the grad_output, both of which can be Eigen::half.
      CHECK_EQ(hlo->operand(i)->shape().element_type(), operand_dtype)
          << "Invalid datatype";
      continue;
    }
    CHECK_EQ(hlo->operand(i)->shape().element_type(), F32)
        << "Not yet implemented";
  }

  // The last operand is the feature index which must be int64.
  CHECK_EQ(hlo->operand(num_operands - 1)->shape().element_type(), S64)
      << "Not yet impelemented";

  // Check Outputs.
  if (hlo->shape().IsTuple()) {
    CHECK_EQ(hlo->shape().tuple_shapes(0).element_type(), operand_dtype)
        << "Invalid datatype";

    for (auto j = 1; j < hlo->shape().tuple_shapes_size(); j++) {
      CHECK_EQ(hlo->shape().tuple_shapes(j).element_type(), F32)
          << "Not yet implemented";
    }
  } else {
    CHECK_EQ(hlo->shape().element_type(), operand_dtype)
        << "Invalid datatype";
  }
}
}  // namespace

static std::pair<dnn::BatchDescriptor /*input_desc*/,
                 dnn::BatchDescriptor /*scale_offset_desc*/>
MakeDescriptors(const Shape& shape, int64 feature_index) {
  std::vector<int64> logical_to_physical =
      LayoutUtil::MakeLogicalToPhysical(shape.layout());

  auto physical_dim_size = [&](int64 physical_dim) {
    return shape.dimensions(LayoutUtil::Major(shape.layout(), physical_dim));
  };

  // Batchnorm only cares about the location of the depth (aka "feature") dim.
  // The other dims are all treated the same.  Thus we can use the kBatchDepthYX
  // cudnn layout for any XLA shape+layout, even XLA shapes that don't have
  // exactly 4 dimensions: We put everything that comes before the feature dim
  // into "batch", and everything that comes after the feature dim into "Y".
  int64 batch_size = 1;
  int64 y_size = 1;
  int64 physical_dim;
  for (physical_dim = 0; physical_dim != logical_to_physical[feature_index];
       ++physical_dim) {
    CHECK_LT(physical_dim, shape.dimensions_size());
    batch_size *= physical_dim_size(physical_dim);
  }
  ++physical_dim;  // Skip the feature dimension.
  for (; physical_dim < shape.dimensions_size(); ++physical_dim) {
    y_size *= physical_dim_size(physical_dim);
  }

  dnn::BatchDescriptor input_desc;
  input_desc.set_layout(dnn::DataLayout::kBatchDepthYX)
      .set_count(batch_size)
      .set_feature_map_count(shape.dimensions(feature_index))
      .set_height(y_size)
      .set_width(1);

  dnn::BatchDescriptor scale_offset_desc;
  scale_offset_desc.set_layout(dnn::DataLayout::kBatchDepthYX)
      .set_feature_map_count(input_desc.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_count(1);

  return std::make_pair(input_desc, scale_offset_desc);
}

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
  CheckInputOutputPrimitivetypeValid(hlo);
}

Status CudnnBatchNormForwardInferenceThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  dnn::BatchDescriptor operand_desc;
  dnn::BatchDescriptor scale_offset_desc;
  std::tie(operand_desc, scale_offset_desc) =
      MakeDescriptors(hlo_instruction()->shape(), feature_index_);

  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  se::DeviceMemoryBase output_base =
      buffer_allocations.GetDeviceAddress(output_);
  PrimitiveType output_primitive_type =
      hlo_instruction()->shape().element_type();

  switch (output_primitive_type) {
    case F16: {
      auto output = se::DeviceMemory<Eigen::half>(output_base);
      stream.ThenBatchNormalizationForward(
          se::DeviceMemory<Eigen::half>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(variance_)),
          /*side_input=*/null_device_ptr,
          operand_desc,                         //
          scale_offset_desc,                    //
          epsilon_,                             //
          se::dnn::ActivationMode::kNone,       //
          &output,                              //
          /*batch_mean=*/nullptr,               //
          /*batch_var=*/nullptr,                //
          /*saved_mean=*/nullptr,               //
          /*saved_inv_var=*/nullptr,            //
          /*is_training=*/false,                //
          /*var_to_inv_var=*/nullptr,           //
          /*inv_var_to_var=*/nullptr,           //
          /*reserve_space_allocator=*/nullptr,  //
          /*workspace_allocator=*/nullptr);
      break;
    }
    case F32: {
      auto output = se::DeviceMemory<float>(output_base);
      stream.ThenBatchNormalizationForward(
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(variance_)),
          /*side_input=*/null_device_ptr,
          operand_desc,                         //
          scale_offset_desc,                    //
          epsilon_,                             //
          se::dnn::ActivationMode::kNone,       //
          &output,                              //
          /*batch_mean=*/nullptr,               //
          /*batch_var=*/nullptr,                //
          /*saved_mean=*/nullptr,               //
          /*saved_inv_var=*/nullptr,            //
          /*is_training=*/false,                //
          /*var_to_inv_var=*/nullptr,           //
          /*inv_var_to_var=*/nullptr,           //
          /*reserve_space_allocator=*/nullptr,  //
          /*workspace_allocator=*/nullptr);
      break;
    }
    default:
      LOG(FATAL) << hlo_instruction()->ToString();
  }

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
  CheckInputOutputPrimitivetypeValid(hlo);
}

Status CudnnBatchNormForwardTrainingThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  dnn::BatchDescriptor operand_desc;
  dnn::BatchDescriptor scale_offset_desc;
  // The BatchNormTraining HLO outputs a tuple of three elements: output data,
  // batch mean, and batch variance.  We want to make our descriptors based on
  // the shape of the output data.
  std::tie(operand_desc, scale_offset_desc) = MakeDescriptors(
      hlo_instruction()->shape().tuple_shapes(0), feature_index_);

  se::DeviceMemoryBase output_data_base =
      buffer_allocations.GetDeviceAddress(output_data_);
  PrimitiveType output_primitive_type =
      hlo_instruction()->shape().tuple_shapes(0).element_type();
  se::DeviceMemory<float> output_mean(
      buffer_allocations.GetDeviceAddress(output_mean_));
  se::DeviceMemory<float> output_inv_stddev(
      buffer_allocations.GetDeviceAddress(output_inv_stddev_));

  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  switch (output_primitive_type) {
    case F16: {
      auto output_data = se::DeviceMemory<Eigen::half>(output_data_base);
      stream.ThenBatchNormalizationForward(
          se::DeviceMemory<Eigen::half>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
          /*estimated_mean=*/null_device_ptr,
          /*estimated_variance=*/null_device_ptr,
          /*side_input=*/null_device_ptr,
          operand_desc,                          //
          scale_offset_desc,                     //
          epsilon_,                              //
          se::dnn::ActivationMode::kNone,        //
          &output_data,                          //
          /*batch_mean=*/&null_device_ptr,       //
          /*batch_var=*/&null_device_ptr,        //
          /*saved_mean=*/&output_mean,           //
          /*saved_inv_var=*/&output_inv_stddev,  //
          /*is_training=*/true,                  //
          /*var_to_inv_var=*/nullptr,            //
          /*inv_var_to_var=*/nullptr,            //
          /*reserve_space_allocator=*/nullptr,   //
          /*workspace_allocator=*/nullptr);
      break;
    }
    case F32: {
      auto output_data = se::DeviceMemory<float>(output_data_base);
      stream.ThenBatchNormalizationForward(
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(offset_)),
          /*estimated_mean=*/null_device_ptr,
          /*estimated_variance=*/null_device_ptr,
          /*side_input=*/null_device_ptr,
          operand_desc,                          //
          scale_offset_desc,                     //
          epsilon_,                              //
          se::dnn::ActivationMode::kNone,        //
          &output_data,                          //
          /*batch_mean=*/&null_device_ptr,       //
          /*batch_var=*/&null_device_ptr,        //
          /*saved_mean=*/&output_mean,           //
          /*saved_inv_var=*/&output_inv_stddev,  //
          /*is_training=*/true,                  //
          /*var_to_inv_var=*/nullptr,            //
          /*inv_var_to_var=*/nullptr,            //
          /*reserve_space_allocator=*/nullptr,   //
          /*workspace_allocator=*/nullptr);
      break;
    }
    default:
      LOG(FATAL) << hlo_instruction()->ToString();
  }

  // Write the output tuple.
  const int kNumOutputs = 3;
  auto ptrs = absl::make_unique<void*[]>(kNumOutputs);
  ptrs[0] = output_data_base.opaque();
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
  CheckInputOutputPrimitivetypeValid(hlo);
}

Status CudnnBatchNormBackwardThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  dnn::BatchDescriptor operand_desc;
  dnn::BatchDescriptor scale_offset_desc;

  // This call outputs a tuple of three elements: grad data, grad offset, and
  // grad scale.  We want to make our descriptors based on the shape of the grad
  // data.
  std::tie(operand_desc, scale_offset_desc) = MakeDescriptors(
      hlo_instruction()->shape().tuple_shapes(0), feature_index_);
  se::DeviceMemoryBase output_grad_data_base =
      buffer_allocations.GetDeviceAddress(output_grad_data_);
  PrimitiveType output_primitive_type =
      hlo_instruction()->shape().tuple_shapes(0).element_type();
  se::DeviceMemory<float> output_grad_scale(
      buffer_allocations.GetDeviceAddress(output_grad_scale_));
  se::DeviceMemory<float> output_grad_offset(
      buffer_allocations.GetDeviceAddress(output_grad_offset_));

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());

  switch (output_primitive_type) {
    case F16: {
      auto output_grad_data =
          se::DeviceMemory<Eigen::half>(output_grad_data_base);
      stream.ThenBatchNormalizationBackward(
          se::DeviceMemory<Eigen::half>(
              buffer_allocations.GetDeviceAddress(grad_output_)),
          se::DeviceMemory<Eigen::half>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(inv_stddev_)),
          operand_desc, scale_offset_desc, epsilon_, &output_grad_data,
          &output_grad_scale, &output_grad_offset, nullptr, nullptr);
      break;
    }
    case F32: {
      auto output_grad_data = se::DeviceMemory<float>(output_grad_data_base);
      stream.ThenBatchNormalizationBackward(
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(grad_output_)),
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(operand_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(scale_)),
          se::DeviceMemory<float>(buffer_allocations.GetDeviceAddress(mean_)),
          se::DeviceMemory<float>(
              buffer_allocations.GetDeviceAddress(inv_stddev_)),
          operand_desc, scale_offset_desc, epsilon_, &output_grad_data,
          &output_grad_scale, &output_grad_offset, nullptr, nullptr);
      break;
    }
    default:
      LOG(FATAL) << hlo_instruction()->ToString();
  }

  // Write the output tuple.
  const int kNumOutputs = 3;
  auto ptrs = absl::make_unique<void*[]>(kNumOutputs);
  ptrs[0] = output_grad_data_base.opaque();
  ptrs[1] = output_grad_scale.opaque();
  ptrs[2] = output_grad_offset.opaque();
  se::DeviceMemory<void*> tuple_addr(
      buffer_allocations.GetDeviceAddress(output_tuple_));
  SafeH2DMemcpy(tuple_addr, std::move(ptrs), kNumOutputs, &stream);

  if (!stream.ok()) {
    return InternalError("BatchNormalizationBackward call failed.");
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
