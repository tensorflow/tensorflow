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

struct CudnnBatchNormParamsCommon {
  se::DeviceMemoryBase operand;
  se::dnn::BatchDescriptor operand_desc;
  se::dnn::BatchDescriptor scale_offset_desc;
  se::DeviceMemory<float> scale;
  float epsilon;
};

struct CudnnBatchNormForwardInferenceParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase output;
  se::DeviceMemory<float> offset;
  se::DeviceMemory<float> mean;
  se::DeviceMemory<float> variance;
};

struct CudnnBatchNormForwardTrainingParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase output_data;
  se::DeviceMemory<float> offset;
  se::DeviceMemory<float> output_mean;
  se::DeviceMemory<float> output_inv_stddev;
};

struct CudnnBatchNormBackwardParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase output_grad_data;
  se::DeviceMemoryBase grad_output;
  se::DeviceMemory<float> output_grad_scale;
  se::DeviceMemory<float> output_grad_offset;
  se::DeviceMemory<float> mean;
  se::DeviceMemory<float> inv_stddev;
};

struct DnnBatchDescriptors {
  se::dnn::BatchDescriptor input_desc;
  se::dnn::BatchDescriptor scale_offset_desc;
};

DnnBatchDescriptors MakeBatchNormDescriptors(const Shape& shape,
                                             int64 feature_index) {
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

  DnnBatchDescriptors batch_descs;
  batch_descs.input_desc.set_layout(se::dnn::DataLayout::kBatchDepthYX)
      .set_count(batch_size)
      .set_feature_map_count(shape.dimensions(feature_index))
      .set_height(y_size)
      .set_width(1);

  batch_descs.scale_offset_desc.set_layout(se::dnn::DataLayout::kBatchDepthYX)
      .set_feature_map_count(batch_descs.input_desc.feature_map_count())
      .set_height(1)
      .set_width(1)
      .set_count(1);

  return batch_descs;
}

void AssignCommonParams(const HloInstruction* batchnorm,
                        CudnnBatchNormParamsCommon* params,
                        const se::DeviceMemoryBase& operand,
                        const se::DeviceMemory<float>& scale, float epsilon,
                        int64 feature_index) {
  // The BatchNormTraining HLO outputs a tuple of three elements: output data,
  // batch mean, and batch variance.  We want to make our descriptors based on
  // the shape of the output data. Batchnorm backward call outputs a tuple of
  // three elements: grad data, grad offset, and grad scale.  We want to make
  // our descriptors based on the shape of the grad data.
  const Shape& shape = batchnorm->shape().IsTuple()
                           ? batchnorm->shape().tuple_shapes(0)
                           : batchnorm->shape();
  DnnBatchDescriptors batch_descs =
      MakeBatchNormDescriptors(shape, feature_index);
  params->operand_desc = batch_descs.input_desc;
  params->scale_offset_desc = batch_descs.scale_offset_desc;
  params->operand = operand;
  params->scale = scale;
  params->epsilon = epsilon;
}

template <typename ElemType>
void RunCudnnBatchNormForwardInferenceImpl(
    CudnnBatchNormForwardInferenceParams* params, se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto output_buf = se::DeviceMemory<ElemType>(params->output);
  stream->ThenBatchNormalizationForward(
      se::DeviceMemory<ElemType>(params->common.operand),
      params->common.scale,                                         //
      params->offset,                                               //
      params->mean,                                                 //
      params->variance,                                             //
      /*side_input=*/null_device_ptr, params->common.operand_desc,  //
      params->common.scale_offset_desc,                             //
      static_cast<double>(params->common.epsilon),                  //
      // TODO(b/137108598): Extend method to allow use of non-trivial
      // exponential averaging.
      /*exponential_average_factor=*/1.0,
      se::dnn::ActivationMode::kNone,       //
      &output_buf,                          //
      /*batch_mean=*/nullptr,               //
      /*batch_var=*/nullptr,                //
      /*saved_mean=*/nullptr,               //
      /*saved_inv_var=*/nullptr,            //
      /*is_training=*/false,                //
      /*var_to_inv_var=*/nullptr,           //
      /*inv_var_to_var=*/nullptr,           //
      /*reserve_space_allocator=*/nullptr,  //
      /*workspace_allocator=*/nullptr);
}

template <typename ElemType>
void RunCudnnBatchNormForwardTrainingImpl(
    CudnnBatchNormForwardTrainingParams* params, se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto output_data = se::DeviceMemory<ElemType>(params->output_data);
  stream->ThenBatchNormalizationForward(
      se::DeviceMemory<ElemType>(params->common.operand),
      params->common.scale,                    //
      params->offset,                          //
      /*estimated_mean=*/null_device_ptr,      //
      /*estimated_variance=*/null_device_ptr,  //
      /*side_input=*/null_device_ptr,          //
      params->common.operand_desc,             //
      params->common.scale_offset_desc,        //
      params->common.epsilon,                  //
      // TODO(b/137108598): Extend method to allow use of non-trivial
      // exponential averaging.
      /*exponential_average_factor=*/1.0,
      se::dnn::ActivationMode::kNone,                //
      &output_data,                                  //
      /*batch_mean=*/&null_device_ptr,               //
      /*batch_var=*/&null_device_ptr,                //
      /*saved_mean=*/&params->output_mean,           //
      /*saved_inv_var=*/&params->output_inv_stddev,  //
      /*is_training=*/true,                          //
      /*var_to_inv_var=*/nullptr,                    //
      /*inv_var_to_var=*/nullptr,                    //
      /*reserve_space_allocator=*/nullptr,           //
      /*workspace_allocator=*/nullptr);
}

template <typename ElemType>
void RunCudnnBatchNormBackwardImpl(CudnnBatchNormBackwardParams* params,
                                   se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  auto output_grad_data = se::DeviceMemory<ElemType>(params->output_grad_data);
  stream->ThenBatchNormalizationBackward(
      se::DeviceMemory<ElemType>(params->grad_output),     //
      se::DeviceMemory<ElemType>(params->common.operand),  //
      params->common.scale,                                //
      params->mean,                                        //
      params->inv_stddev,                                  //
      params->common.operand_desc,                         //
      params->common.scale_offset_desc,                    //
      params->common.epsilon,                              //
      &output_grad_data,                                   //
      &params->output_grad_scale,                          //
      &params->output_grad_offset,                         //
      /*reserve_space_allocator=*/nullptr,                 //
      /*workspace_allocator=*/nullptr);
}

}  // namespace

Status RunCudnnBatchNormForwardInference(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::DeviceMemory<float> mean,
    se::DeviceMemory<float> variance, float epsilon, int64 feature_index,
    se::Stream* stream) {
  CudnnBatchNormForwardInferenceParams inference_params;
  AssignCommonParams(batchnorm, &inference_params.common, operand, scale,
                     epsilon, feature_index);
  inference_params.offset = offset;
  inference_params.mean = mean;
  inference_params.variance = variance;
  inference_params.output = output;

  PrimitiveType output_primitive_type = batchnorm->shape().element_type();
  switch (output_primitive_type) {
    case F16:
      RunCudnnBatchNormForwardInferenceImpl<Eigen::half>(&inference_params,
                                                         stream);
      break;
    case F32:
      RunCudnnBatchNormForwardInferenceImpl<float>(&inference_params, stream);
      break;
    default:
      return Unimplemented("Primitive type not implemented for \"%s\" ",
                           batchnorm->ToString());
  }
  return Status::OK();
}

Status RunCudnnBatchNormForwardTraining(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_data, se::DeviceMemory<float> output_mean,
    se::DeviceMemory<float> output_inv_stddev, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, float epsilon, int64 feature_index,
    se::Stream* stream) {
  CudnnBatchNormForwardTrainingParams forward_params;
  AssignCommonParams(batchnorm, &forward_params.common, operand, scale, epsilon,
                     feature_index);
  forward_params.offset = offset;
  forward_params.output_data = output_data;
  forward_params.output_mean = output_mean;
  forward_params.output_inv_stddev = output_inv_stddev;

  PrimitiveType output_primitive_type =
      batchnorm->shape().tuple_shapes(0).element_type();
  switch (output_primitive_type) {
    case F16:
      RunCudnnBatchNormForwardTrainingImpl<Eigen::half>(&forward_params,
                                                        stream);
      break;
    case F32:
      RunCudnnBatchNormForwardTrainingImpl<float>(&forward_params, stream);
      break;
    default:
      return Unimplemented("Primitive type not implemented for \"%s\" ",
                           batchnorm->ToString());
  }
  return Status::OK();
}

Status RunCudnnBatchNormBackward(
    const HloInstruction* batchnorm, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_grad_data, se::DeviceMemoryBase grad_output,
    se::DeviceMemory<float> output_grad_scale,
    se::DeviceMemory<float> output_grad_offset, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> mean, se::DeviceMemory<float> inv_stddev,
    float epsilon, int64 feature_index, se::Stream* stream) {
  CudnnBatchNormBackwardParams backward_params;
  AssignCommonParams(batchnorm, &backward_params.common, operand, scale,
                     epsilon, feature_index);
  backward_params.output_grad_data = output_grad_data;
  backward_params.grad_output = grad_output;
  backward_params.output_grad_scale = output_grad_scale;
  backward_params.output_grad_offset = output_grad_offset;
  backward_params.mean = mean;
  backward_params.inv_stddev = inv_stddev;

  PrimitiveType output_primitive_type =
      batchnorm->shape().tuple_shapes(0).element_type();
  switch (output_primitive_type) {
    case F16:
      RunCudnnBatchNormBackwardImpl<Eigen::half>(&backward_params, stream);
      break;
    case F32:
      RunCudnnBatchNormBackwardImpl<float>(&backward_params, stream);
      break;
    default:
      return Unimplemented("Primitive type not implemented for \"%s\" ",
                           batchnorm->ToString());
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
