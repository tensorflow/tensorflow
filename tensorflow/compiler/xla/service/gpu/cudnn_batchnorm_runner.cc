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

#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_runner.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {
StatusOr<CudnnBatchNormKind> GetCudnnBatchNormKind(
    const HloInstruction* instr) {
  absl::string_view target = instr->custom_call_target();
  if (target == kCudnnBatchNormForwardInferenceCallTarget) {
    return CudnnBatchNormKind::kCudnnBatchNormForwardInference;
  }
  if (target == kCudnnBatchNormForwardTrainingCallTarget) {
    return CudnnBatchNormKind::kCudnnBatchNormForwardTraining;
  }
  if (target == kCudnnBatchNormBackwardCallTarget) {
    return CudnnBatchNormKind::kCudnnBatchNormBackward;
  }
  return InternalError("Unexpected call target: %s", target);
}

void AssignNonOptionalParams(CudnnBatchNormParams& params,
                             const se::DeviceMemoryBase& operand,
                             const se::DeviceMemory<float>& scale,
                             float epsilon) {
  params.operand = operand;
  params.scale = scale;
  params.epsilon = epsilon;
}

static std::pair<se::dnn::BatchDescriptor /*input_desc*/,
                 se::dnn::BatchDescriptor /*scale_offset_desc*/>
MakeBatchNormDescriptors(const Shape& shape, int64 feature_index) {
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

  se::dnn::BatchDescriptor input_desc;
  input_desc.set_layout(se::dnn::DataLayout::kBatchDepthYX)
      .set_count(batch_size)
      .set_feature_map_count(shape.dimensions(feature_index))
      .set_height(y_size)
      .set_width(1);

  se::dnn::BatchDescriptor scale_offset_desc;
  scale_offset_desc.set_layout(se::dnn::DataLayout::kBatchDepthYX)
      .set_feature_map_count(input_desc.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_count(1);

  return std::make_pair(input_desc, scale_offset_desc);
}

template <typename ElemType>
Status RunCudnnBatchNormImplInternal(CudnnBatchNormParams& params,
                                     se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  switch (params.kind) {
    case CudnnBatchNormKind::kCudnnBatchNormForwardInference: {
      auto output_buf =
          se::DeviceMemory<ElemType>(params.forward_inference->output);
      stream->ThenBatchNormalizationForward(
          se::DeviceMemory<ElemType>(params.operand),
          params.scale,                                         //
          params.forward_inference->offset,                     //
          params.forward_inference->mean,                       //
          params.forward_inference->variance,                   //
          /*side_input=*/null_device_ptr, params.operand_desc,  //
          params.scale_offset_desc, params.epsilon,             //
          se::dnn::ActivationMode::kNone,                       //
          &output_buf,                                          //
          /*batch_mean=*/nullptr,                               //
          /*batch_var=*/nullptr,                                //
          /*saved_mean=*/nullptr,                               //
          /*saved_inv_var=*/nullptr,                            //
          /*is_training=*/false,                                //
          /*var_to_inv_var=*/nullptr,                           //
          /*inv_var_to_var=*/nullptr,                           //
          /*reserve_space_allocator=*/nullptr,                  //
          /*workspace_allocator=*/nullptr);
      break;
    }
    case CudnnBatchNormKind::kCudnnBatchNormForwardTraining: {
      auto output_data =
          se::DeviceMemory<ElemType>(params.forward_training->output_data);
      stream->ThenBatchNormalizationForward(
          se::DeviceMemory<ElemType>(params.operand),
          params.scale,                                                   //
          params.forward_training->offset,                                //
          /*estimated_mean=*/null_device_ptr,                             //
          /*estimated_variance=*/null_device_ptr,                         //
          /*side_input=*/null_device_ptr,                                 //
          params.operand_desc,                                            //
          params.scale_offset_desc,                                       //
          params.epsilon,                                                 //
          se::dnn::ActivationMode::kNone,                                 //
          &output_data,                                                   //
          /*batch_mean=*/&null_device_ptr,                                //
          /*batch_var=*/&null_device_ptr,                                 //
          /*saved_mean=*/&params.forward_training->output_mean,           //
          /*saved_inv_var=*/&params.forward_training->output_inv_stddev,  //
          /*is_training=*/true,                                           //
          /*var_to_inv_var=*/nullptr,                                     //
          /*inv_var_to_var=*/nullptr,                                     //
          /*reserve_space_allocator=*/nullptr,                            //
          /*workspace_allocator=*/nullptr);
      break;
    }
    case CudnnBatchNormKind::kCudnnBatchNormBackward: {
      auto output_grad_data =
          se::DeviceMemory<ElemType>(params.backward->output_grad_data);
      stream->ThenBatchNormalizationBackward(
          se::DeviceMemory<ElemType>(params.backward->grad_output),  //
          se::DeviceMemory<ElemType>(params.operand),                //
          params.scale,                                              //
          params.backward->mean,                                     //
          params.backward->inv_stddev,                               //
          params.operand_desc,                                       //
          params.scale_offset_desc,                                  //
          params.epsilon,                                            //
          &output_grad_data,                                         //
          &params.backward->output_grad_scale,                       //
          &params.backward->output_grad_offset,                      //
          /*reserve_space_allocator=*/nullptr,                       //
          /*workspace_allocator=*/nullptr);
      break;
    }
    default:
      return InternalError("Invalid Batchnorm kind");
  }

  return Status::OK();
}

Status RunCudnnBatchNormImpl(const HloInstruction* batchnorm,
                             CudnnBatchNormParams& params, se::Stream* stream) {
  PrimitiveType output_primitive_type =
      batchnorm->shape().IsTuple()
          ? batchnorm->shape().tuple_shapes(0).element_type()
          : batchnorm->shape().element_type();
  switch (output_primitive_type) {
    case F16:
      return RunCudnnBatchNormImplInternal<Eigen::half>(params, stream);
    case F32:
      return RunCudnnBatchNormImplInternal<float>(params, stream);
    default:
      LOG(FATAL) << batchnorm->ToString();
  }
}

}  // namespace

Status RunCudnnBatchNormForwardInference(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::DeviceMemory<float> mean,
    se::DeviceMemory<float> variance, float epsilon, int64 feature_index,
    se::Stream* stream) {
  CudnnBatchNormParams inference_params;
  TF_ASSIGN_OR_RETURN(inference_params.kind, GetCudnnBatchNormKind(batchnorm));
  std::tie(inference_params.operand_desc, inference_params.scale_offset_desc) =
      MakeBatchNormDescriptors(batchnorm->shape(), feature_index);

  AssignNonOptionalParams(inference_params, operand, scale, epsilon);
  inference_params.forward_inference.emplace();
  inference_params.forward_inference->offset = offset;
  inference_params.forward_inference->mean = mean;
  inference_params.forward_inference->variance = variance;
  inference_params.forward_inference->output = output;

  return RunCudnnBatchNormImpl(batchnorm, inference_params, stream);
}

Status RunCudnnBatchNormForwardTraining(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_data, se::DeviceMemory<float> output_mean,
    se::DeviceMemory<float> output_inv_stddev, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, float epsilon, int64 feature_index,
    se::Stream* stream) {
  CudnnBatchNormParams forward_params;
  TF_ASSIGN_OR_RETURN(forward_params.kind, GetCudnnBatchNormKind(batchnorm));
  // The BatchNormTraining HLO outputs a tuple of three elements: output data,
  // batch mean, and batch variance.  We want to make our descriptors based on
  // the shape of the output data.
  std::tie(forward_params.operand_desc, forward_params.scale_offset_desc) =
      MakeBatchNormDescriptors(batchnorm->shape().tuple_shapes(0),
                               feature_index);

  AssignNonOptionalParams(forward_params, operand, scale, epsilon);
  forward_params.forward_training.emplace();
  forward_params.forward_training->offset = offset;
  forward_params.forward_training->output_data = output_data;
  forward_params.forward_training->output_mean = output_mean;
  forward_params.forward_training->output_inv_stddev = output_inv_stddev;

  return RunCudnnBatchNormImpl(batchnorm, forward_params, stream);
}

Status RunCudnnBatchNormBackward(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_grad_data, se::DeviceMemoryBase grad_output,
    se::DeviceMemory<float> output_grad_scale,
    se::DeviceMemory<float> output_grad_offset, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> mean, se::DeviceMemory<float> inv_stddev,
    float epsilon, int64 feature_index, se::Stream* stream) {
  CudnnBatchNormParams backward_params;
  TF_ASSIGN_OR_RETURN(backward_params.kind, GetCudnnBatchNormKind(batchnorm));
  // This call outputs a tuple of three elements: grad data, grad offset, and
  // grad scale.  We want to make our descriptors based on the shape of the grad
  // data.
  std::tie(backward_params.operand_desc, backward_params.scale_offset_desc) =
      MakeBatchNormDescriptors(batchnorm->shape().tuple_shapes(0),
                               feature_index);

  AssignNonOptionalParams(backward_params, operand, scale, epsilon);
  backward_params.backward.emplace();
  backward_params.backward->output_grad_data = output_grad_data;
  backward_params.backward->grad_output = grad_output;
  backward_params.backward->output_grad_scale = output_grad_scale;
  backward_params.backward->output_grad_offset = output_grad_offset;
  backward_params.backward->mean = mean;
  backward_params.backward->inv_stddev = inv_stddev;

  return RunCudnnBatchNormImpl(batchnorm, backward_params, stream);
}

}  // namespace gpu
}  // namespace xla
