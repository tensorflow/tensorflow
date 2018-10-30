/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifdef TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <string.h>
#include <map>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/gpu_fusion_ops.h"

#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

//---------------------------------------------------------------------------

template <typename Device, typename T>
class ROCmFusionKernelBatchNormActivationInference : public OpKernel {
 public:
  explicit ROCmFusionKernelBatchNormActivationInference(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    float epsilon;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    string activation_mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(ctx, GetActivationModeFromString(activation_mode_str,
                                                    &activation_mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);
    const Tensor& mean = ctx->input(3);
    const Tensor& variance = ctx->input(4);

    OP_REQUIRES(ctx, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(ctx, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(ctx, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(ctx, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(ctx, variance.dims() == 1,
                errors::InvalidArgument("variance must be 1-dimensional",
                                        variance.shape().DebugString()));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    const int64 batch_size = GetTensorDim(x, data_format_, 'N');
    const int64 height = GetTensorDim(x, data_format_, 'H');
    const int64 width = GetTensorDim(x, data_format_, 'W');
    const int64 channels = GetTensorDim(x, data_format_, 'C');

    if (x.shape().num_elements() != 0) {
      Tensor fusion_input = x;
      Tensor fusion_output = *y;

      // if the data format is NHWC, we need to
      // 1. convert the input tensor to NCHW format
      // 2, allocate a temporary tensor to hold the fusion op output (which will
      // be in NCHW format)
      if (data_format_ == FORMAT_NHWC) {
        // allocate a temporary tensor to store the NCHW input
        Tensor transformed_input;
        TensorShape nchw_shape_input =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input,
                                    &transformed_input));

        // convert the input tensor to NCHW format for the GPU
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());

        fusion_input = transformed_input;

        // allocate a temporary tensor to store the NCHW output
        Tensor transformed_output;
        TensorShape nchw_shape_output =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_output,
                                    &transformed_output));

        fusion_output = transformed_output;
      }

      se::dnn::BatchDescriptor x_desc;
      x_desc.set_count(batch_size)
          .set_feature_map_count(channels)
          .set_height(height)
          .set_width(width)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::BatchDescriptor scale_offset_mean_variance_desc;
      scale_offset_mean_variance_desc.set_count(1)
          .set_feature_map_count(channels)
          .set_height(1)
          .set_width(1)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      auto x_data = AsDeviceMemory(fusion_input.template flat<T>().data(),
                                   fusion_input.template flat<T>().size());
      auto scale_data = AsDeviceMemory(scale.template flat<T>().data(),
                                       scale.template flat<T>().size());

      auto offset_data = AsDeviceMemory(offset.template flat<T>().data(),
                                        offset.template flat<T>().size());

      auto mean_data = AsDeviceMemory(mean.template flat<T>().data(),
                                      mean.template flat<T>().size());

      auto variance_data = AsDeviceMemory(variance.template flat<T>().data(),
                                          variance.template flat<T>().size());

      auto dnn_activation_mode = GetDnnActivationMode(activation_mode_);

      auto y_data = AsDeviceMemory(fusion_output.template flat<T>().data(),
                                   fusion_output.template flat<T>().size());

      auto* stream = ctx->op_device_context()->stream();

      bool miopen_status =
          stream
              ->ThenFusedBatchNormActivationInference(
                  x_desc, x_data, scale_offset_mean_variance_desc, scale_data,
                  offset_data, mean_data, variance_data, epsilon_,
                  dnn_activation_mode, &y_data)
              .ok();

      if (!miopen_status) {
        ctx->SetStatus(
            errors::Internal("MIOpen BnA (inference) FusionOp launch Failure"));
      }

      // if the data format is NHWC, we need to convert the fusion op output
      // back to NHWC format
      if (data_format_ == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_output).tensor<T, 4>(),
            y->tensor<T, 4>());
      }
    }
  }

 private:
  double epsilon_;
  TensorFormat data_format_;
  ActivationMode activation_mode_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationInference")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    ROCmFusionKernelBatchNormActivationInference<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationInference")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
    ROCmFusionKernelBatchNormActivationInference<GPUDevice, Eigen::half>);

//---------------------------------------------------------------------------

template <typename Device, typename T>
class ROCmFusionKernelBatchNormActivationForward : public OpKernel {
 public:
  explicit ROCmFusionKernelBatchNormActivationForward(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    float epsilon;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    string activation_mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(ctx, GetActivationModeFromString(activation_mode_str,
                                                    &activation_mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& x = ctx->input(0);
    const Tensor& scale = ctx->input(1);
    const Tensor& offset = ctx->input(2);

    OP_REQUIRES(ctx, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(ctx, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(ctx, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));

    Tensor* y = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &y));

    Tensor* batch_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, scale.shape(), &batch_mean));

    Tensor* batch_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, scale.shape(), &batch_var));

    Tensor* saved_mean = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, scale.shape(), &saved_mean));

    Tensor* saved_var = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, scale.shape(), &saved_var));

    const int64 batch_size = GetTensorDim(x, data_format_, 'N');
    const int64 height = GetTensorDim(x, data_format_, 'H');
    const int64 width = GetTensorDim(x, data_format_, 'W');
    const int64 channels = GetTensorDim(x, data_format_, 'C');

    if (x.shape().num_elements() != 0) {
      Tensor fusion_input = x;
      Tensor fusion_output = *y;

      // if the data format is NHWC, we need to
      // 1. convert the input tensor to NCHW format
      // 2, allocate a temporary tensor to hold the fusion op output (which will
      // be in NCHW format)
      if (data_format_ == FORMAT_NHWC) {
        // allocate a temporary tensor to store the NCHW input
        Tensor transformed_input;
        TensorShape nchw_shape_input =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input,
                                    &transformed_input));

        // convert the input tensor to NCHW format for the GPU
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());

        fusion_input = transformed_input;

        // allocate a temporary tensor to store the NCHW output
        Tensor transformed_output;
        TensorShape nchw_shape_output =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_output,
                                    &transformed_output));

        fusion_output = transformed_output;
      }

      se::dnn::BatchDescriptor x_desc;
      x_desc.set_count(batch_size)
          .set_feature_map_count(channels)
          .set_height(height)
          .set_width(width)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::BatchDescriptor scale_offset_mean_variance_desc;
      scale_offset_mean_variance_desc.set_count(1)
          .set_feature_map_count(channels)
          .set_height(1)
          .set_width(1)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      auto x_data = AsDeviceMemory(fusion_input.template flat<T>().data(),
                                   fusion_input.template flat<T>().size());
      auto scale_data = AsDeviceMemory(scale.template flat<T>().data(),
                                       scale.template flat<T>().size());

      auto offset_data = AsDeviceMemory(offset.template flat<T>().data(),
                                        offset.template flat<T>().size());

      auto dnn_activation_mode = GetDnnActivationMode(activation_mode_);

      auto y_data = AsDeviceMemory(fusion_output.template flat<T>().data(),
                                   fusion_output.template flat<T>().size());

      auto batch_mean_data =
          AsDeviceMemory(batch_mean->template flat<T>().data(),
                         batch_mean->template flat<T>().size());

      auto batch_var_data =
          AsDeviceMemory(batch_var->template flat<T>().data(),
                         batch_var->template flat<T>().size());

      auto saved_mean_data =
          AsDeviceMemory(saved_mean->template flat<T>().data(),
                         saved_mean->template flat<T>().size());

      auto saved_var_data =
          AsDeviceMemory(saved_var->template flat<T>().data(),
                         saved_var->template flat<T>().size());

      auto* stream = ctx->op_device_context()->stream();

      bool miopen_status =
          stream
              ->ThenFusedBatchNormActivationForward(
                  x_desc, x_data, scale_offset_mean_variance_desc, scale_data,
                  offset_data, epsilon_, dnn_activation_mode, &y_data,
                  &batch_mean_data, &batch_var_data, &saved_mean_data,
                  &saved_var_data)
              .ok();

      if (!miopen_status) {
        ctx->SetStatus(
            errors::Internal("MIOpen BnA (forward) FusionOp launch Failure"));
      }

      // if the data format is NHWC, we need to convert the fusion op output
      // back to NHWC format
      if (data_format_ == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_output).tensor<T, 4>(),
            y->tensor<T, 4>());
      }
    }
  }

 private:
  double epsilon_;
  TensorFormat data_format_;
  ActivationMode activation_mode_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationForward")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    ROCmFusionKernelBatchNormActivationForward<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationForward")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
    ROCmFusionKernelBatchNormActivationForward<GPUDevice, Eigen::half>);

//---------------------------------------------------------------------------

template <typename Device, typename T>
class ROCmFusionKernelBatchNormActivationBackward : public OpKernel {
 public:
  explicit ROCmFusionKernelBatchNormActivationBackward(
      OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    float epsilon;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;

    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));

    string activation_mode_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("activation_mode", &activation_mode_str));
    OP_REQUIRES_OK(ctx, GetActivationModeFromString(activation_mode_str,
                                                    &activation_mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& y_act_backprop = ctx->input(0);
    const Tensor& y_act = ctx->input(1);
    const Tensor& x_bn = ctx->input(2);
    const Tensor& scale = ctx->input(3);
    const Tensor& offset = ctx->input(4);
    const Tensor& mean = ctx->input(5);
    const Tensor& variance = ctx->input(6);

    OP_REQUIRES(ctx, y_act_backprop.dims() == 4,
                errors::InvalidArgument("y_act_backprop must be 4-dimensional",
                                        y_act_backprop.shape().DebugString()));
    OP_REQUIRES(ctx, y_act.dims() == 4,
                errors::InvalidArgument("y_act must be 4-dimensional",
                                        y_act.shape().DebugString()));
    OP_REQUIRES(ctx, x_bn.dims() == 4,
                errors::InvalidArgument("x_bn must be 4-dimensional",
                                        x_bn.shape().DebugString()));
    OP_REQUIRES(ctx, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(ctx, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(ctx, mean.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        mean.shape().DebugString()));
    OP_REQUIRES(ctx, variance.dims() == 1,
                errors::InvalidArgument("variance must be 1-dimensional",
                                        variance.shape().DebugString()));

    Tensor* x_bn_backprop = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, y_act_backprop.shape(), &x_bn_backprop));

    Tensor* scale_backprop = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(1, scale.shape(), &scale_backprop));

    Tensor* offset_backprop = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(2, offset.shape(), &offset_backprop));

    const int64 batch_size = GetTensorDim(x_bn, data_format_, 'N');
    const int64 height = GetTensorDim(x_bn, data_format_, 'H');
    const int64 width = GetTensorDim(x_bn, data_format_, 'W');
    const int64 channels = GetTensorDim(x_bn, data_format_, 'C');

    if (x_bn.shape().num_elements() != 0) {
      Tensor fusion_input = y_act_backprop;
      Tensor fusion_output = *x_bn_backprop;

      // if the data format is NHWC, we need to
      // 1. convert the input tensor to NCHW format
      // 2, allocate a temporary tensor to hold the fusion op output (which will
      // be in NCHW format)
      if (data_format_ == FORMAT_NHWC) {
        // allocate a temporary tensor to store the NCHW input
        Tensor transformed_input;
        TensorShape nchw_shape_input =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_input,
                                    &transformed_input));

        // convert the input tensor to NCHW format for the GPU
        functor::NHWCToNCHW<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());

        fusion_input = transformed_input;

        // allocate a temporary tensor to store the NCHW output
        Tensor transformed_output;
        TensorShape nchw_shape_output =
            ShapeFromFormat(FORMAT_NCHW, batch_size, height, width, channels);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, nchw_shape_output,
                                    &transformed_output));

        fusion_output = transformed_output;
      }

      se::dnn::BatchDescriptor y_act_backprop_desc;
      y_act_backprop_desc.set_count(batch_size)
          .set_feature_map_count(channels)
          .set_height(height)
          .set_width(width)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      se::dnn::BatchDescriptor scale_offset_mean_variance_desc;
      scale_offset_mean_variance_desc.set_count(1)
          .set_feature_map_count(channels)
          .set_height(1)
          .set_width(1)
          .set_layout(se::dnn::DataLayout::kBatchDepthYX);

      auto y_act_backprop_data =
          AsDeviceMemory(fusion_input.template flat<T>().data(),
                         fusion_input.template flat<T>().size());

      auto y_act_data = AsDeviceMemory(y_act.template flat<T>().data(),
                                       y_act.template flat<T>().size());

      auto x_bn_data = AsDeviceMemory(x_bn.template flat<T>().data(),
                                      x_bn.template flat<T>().size());

      auto scale_data = AsDeviceMemory(scale.template flat<T>().data(),
                                       scale.template flat<T>().size());

      auto offset_data = AsDeviceMemory(offset.template flat<T>().data(),
                                        offset.template flat<T>().size());

      auto mean_data = AsDeviceMemory(mean.template flat<T>().data(),
                                      mean.template flat<T>().size());

      auto variance_data = AsDeviceMemory(variance.template flat<T>().data(),
                                          variance.template flat<T>().size());

      auto dnn_activation_mode = GetDnnActivationMode(activation_mode_);

      auto x_bn_backprop_data =
          AsDeviceMemory(fusion_output.template flat<T>().data(),
                         fusion_output.template flat<T>().size());

      auto scale_backprop_data =
          AsDeviceMemory(scale_backprop->template flat<T>().data(),
                         scale_backprop->template flat<T>().size());

      auto offset_backprop_data =
          AsDeviceMemory(offset_backprop->template flat<T>().data(),
                         offset_backprop->template flat<T>().size());

      auto* stream = ctx->op_device_context()->stream();

      bool miopen_status =
          stream
              ->ThenFusedBatchNormActivationBackward(
                  y_act_backprop_desc, y_act_backprop_data, y_act_data,
                  dnn_activation_mode, x_bn_data,
                  scale_offset_mean_variance_desc, scale_data, offset_data,
                  mean_data, variance_data, &x_bn_backprop_data,
                  &scale_backprop_data, &offset_backprop_data)
              .ok();

      if (!miopen_status) {
        ctx->SetStatus(
            errors::Internal("MIOpen BnA (backward) FusionOp launch Failure"));
      }

      // if the data format is NHWC, we need to convert the fusion op output
      // back to NHWC format
      if (data_format_ == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(fusion_output).tensor<T, 4>(),
            x_bn_backprop->tensor<T, 4>());
      }
    }
  }

 private:
  double epsilon_;
  TensorFormat data_format_;
  ActivationMode activation_mode_;
};

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationBackward")
        .Device(DEVICE_GPU)
        .TypeConstraint<float>("T"),
    ROCmFusionKernelBatchNormActivationBackward<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("_ROCmFusedBatchNormActivationBackward")
        .Device(DEVICE_GPU)
        .TypeConstraint<Eigen::half>("T"),
    ROCmFusionKernelBatchNormActivationBackward<GPUDevice, Eigen::half>);

//---------------------------------------------------------------------------

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
