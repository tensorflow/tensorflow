/* Copyright 2015 Google Inc. All Rights Reserved.
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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "tensorflow/core/kernels/conv_2d.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#else// GOOGLE_CUDA
#error "Requires GOOGLE_CUDA"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if GOOGLE_CUDA
namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory,
                                                    uint64 size) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory),
                                                size * sizeof(T));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace
#endif  // GOOGLE_CUDA

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchBatchNormTraining;

template <typename T>
struct LaunchBatchNormTraining<GPUDevice, T> {
  static void launch(OpKernelContext* ctx,
                     const float epsilon,
                     const TensorFormat data_format,
                     const Tensor& input_param,
                     const Tensor& scale_param, const Tensor& bias_param,
                     Tensor* output,
                     Tensor* save_mean,
                     Tensor* save_inv_var) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream avalible"));

    Tensor input = input_param;

    const int64 in_batch = GetTensorDim(input, data_format, 'N');
    const int64 in_depths = GetTensorDim(input, data_format, 'C');
    const int64 in_cols = GetTensorDim(input, data_format, 'H');
    const int64 in_rows = GetTensorDim(input, data_format, 'W');

    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
      .set_feature_map_count(in_depths)
      .set_height(in_rows)
      .set_width(in_cols)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::BatchDescriptor scale_bias_mean_var_desc;
    scale_bias_mean_var_desc.set_count(1)
      .set_feature_map_count(in_depths)
      .set_height(1)
      .set_width(1)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    Tensor transformed_output;
    perftools::gputools::DeviceMemory<T> output_ptr;

    if (data_format == FORMAT_NHWC) {
      // Convert the input tensor from NHWC to NCHW.
      Tensor transformed_input;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_input));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());
      input = transformed_input;

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_output));

      output_ptr = AsDeviceMemory(transformed_output.template flat<T>().data(),
                                  transformed_output.template flat<T>().size());
    } else {
      output_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                  output->template flat<T>().size());
    }

    auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());


    auto scale_ptr = AsDeviceMemory(scale_param.template flat<T>().data(),
                                    scale_param.template flat<T>().size());
    auto bias_ptr = AsDeviceMemory(bias_param.template flat<T>().data(),
                                    bias_param.template flat<T>().size());

    auto save_mean_ptr = AsDeviceMemory(save_mean->template flat<T>().data(),
                                        save_mean->template flat<T>().size());
    auto save_inv_var_ptr = AsDeviceMemory(save_inv_var->template flat<T>().data(),
                                        save_inv_var->template flat<T>().size());

    bool cudnn_launch_status =
      stream
        ->ThenBatchNormTrainingForward(epsilon,
                                    input_desc,
                                    input_ptr,
                                    scale_bias_mean_var_desc,
                                    scale_ptr,
                                    bias_ptr,
                                    input_desc,
                                    &output_ptr,
                                    &save_mean_ptr,
                                    &save_inv_var_ptr)
        .ok();

      if (!cudnn_launch_status) {
        ctx->SetStatus(errors::Internal(
            "cuDNN launch failure : input shape(", input.shape().DebugString(),
            ")"));
      }

    if (data_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
          output->tensor<T, 4>());
    }
  }
};

template <typename Device, typename T>
class BatchNormTrainingOp : public OpKernel {
  public:
    explicit BatchNormTrainingOp(OpKernelConstruction* context) : OpKernel(context) {
      const DataType dt = DataTypeToEnum<T>::v();
      OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt}, {dt, dt, dt}));
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));

      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }

    void Compute(OpKernelContext* context) override {
      const Tensor& input = context->input(0);
      const Tensor& scale = context->input(1);
      const Tensor& bias = context->input(2);

      OP_REQUIRES(context, input.shape().dims() == 4,
          errors::InvalidArgument("input must be 4-dimensional", input.shape().DebugString()));
      OP_REQUIRES(context, scale.shape().dims() == 1,
          errors::InvalidArgument("scale must be 1-dimensional", scale.shape().DebugString()));
      OP_REQUIRES(context, bias.shape().dims() == 1,
          errors::InvalidArgument("bias must be 1-dimensional", bias.shape().DebugString()));

      Tensor* output = nullptr;
      TensorShape out_shape = input.shape();
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

      const int64 in_depths = GetTensorDim(input, data_format_, 'C');
      OP_REQUIRES(context, in_depths == scale.dim_size(0),
          errors::InvalidArgument("scale size does not match channels of input"));
      OP_REQUIRES(context, in_depths == bias.dim_size(0),
          errors::InvalidArgument("bias size does not match channels of input"));

      TensorShape save_mean_var_shape = TensorShape({in_depths});


      Tensor* save_mean = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, save_mean_var_shape, &save_mean));
      Tensor* save_inv_var = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, save_mean_var_shape, &save_inv_var));

      LaunchBatchNormTraining<Device, T>::launch(
          context, epsilon_, data_format_,
          input, scale, bias, output,
          save_mean, save_inv_var);
    }

  private:
    float epsilon_;
    TensorFormat data_format_;

    TF_DISALLOW_COPY_AND_ASSIGN(BatchNormTrainingOp);
};

template <typename Device, typename T>
struct LaunchBatchNormTrainingGrad;

template <typename T>
struct LaunchBatchNormTrainingGrad<GPUDevice, T> {
  static void launch(OpKernelContext* ctx,
                     const float epsilon,
                     const TensorFormat data_format,
                     const Tensor& input_param,
                     const Tensor& output_grad_param,
                     const Tensor& scale_param,
                     const Tensor& saved_mean,
                     const Tensor& saved_inv_var,
                     Tensor* input_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream avalible"));

    Tensor input = input_param;
    Tensor output_grad = output_grad_param;

    const int64 in_batch = GetTensorDim(input, data_format, 'N');
    const int64 in_depths = GetTensorDim(input, data_format, 'C');
    const int64 in_cols = GetTensorDim(input, data_format, 'H');
    const int64 in_rows = GetTensorDim(input, data_format, 'W');

    perftools::gputools::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
      .set_feature_map_count(in_depths)
      .set_height(in_rows)
      .set_width(in_cols)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::BatchDescriptor output_desc = input_desc;

    perftools::gputools::dnn::BatchDescriptor scale_bias_mean_var_desc;
    scale_bias_mean_var_desc.set_count(1)
      .set_feature_map_count(in_depths)
      .set_height(1)
      .set_width(1)
      .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    Tensor transformed_input_grad;
    perftools::gputools::DeviceMemory<T> input_grad_ptr;

    if (data_format == FORMAT_NHWC) {
      // Convert the input tensor from NHWC to NCHW.
      Tensor transformed_input;
      Tensor transformed_output_grad;
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_input));
      // TODO I don't think 2 temp allocations are needed.
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_output_grad));

      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(input).tensor<T, 4>(),
          transformed_input.tensor<T, 4>());

      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          ctx->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(output_grad).tensor<T, 4>(),
          transformed_output_grad.tensor<T, 4>());

      input = transformed_input;
      output_grad = transformed_output_grad;

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, in_batch,
                                                  in_rows, in_cols, in_depths),
                                  &transformed_input_grad));

      input_grad_ptr = AsDeviceMemory(transformed_input_grad.template flat<T>().data(),
                                      transformed_input_grad.template flat<T>().size());
    } else {
      input_grad_ptr = AsDeviceMemory(input_grad->template flat<T>().data(),
                                      input_grad->template flat<T>().size());
    }

    auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());
    auto output_grad_ptr = AsDeviceMemory(output_grad.template flat<T>().data(),
                                     output_grad.template flat<T>().size());
    auto scale_ptr = AsDeviceMemory(scale_param.template flat<T>().data(),
                                    scale_param.template flat<T>().size());
    auto saved_mean_ptr = AsDeviceMemory(saved_mean.template flat<T>().data(),
                                         saved_mean.template flat<T>().size());
    auto saved_inv_var_ptr = AsDeviceMemory(saved_inv_var.template flat<T>().data(),
                                         saved_inv_var.template flat<T>().size());

    auto scale_grad_ptr = AsDeviceMemory(scale_grad->template flat<T>().data(),
                                         scale_grad->template flat<T>().size());
    auto bias_grad_ptr = AsDeviceMemory(bias_grad->template flat<T>().data(),
                                        bias_grad->template flat<T>().size());
    bool cudnn_launch_status =
      stream
        ->ThenBatchNormTrainingBackward(epsilon,
                                             input_desc,
                                             input_ptr,
                                             output_desc,
                                             output_grad_ptr,
                                             scale_bias_mean_var_desc,
                                             scale_ptr,
                                             saved_mean_ptr,
                                             saved_inv_var_ptr,
                                             &input_grad_ptr,
                                             &scale_grad_ptr,
                                             &bias_grad_ptr)
        .ok();

      if (!cudnn_launch_status) {
        ctx->SetStatus(errors::Internal(
            "cuDNN launch failure : input shape(", input.shape().DebugString(),
            ")"));
      }

      if (data_format == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T, 4>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(transformed_input_grad).tensor<T, 4>(),
            input_grad->tensor<T, 4>());
      }
  }
};

template <typename Device, typename T>
class BatchNormTrainingGradOp : public OpKernel {
  public:
    explicit BatchNormTrainingGradOp(OpKernelConstruction* context) : OpKernel(context) {
      const DataType dt = DataTypeToEnum<T>::v();
      OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt, dt, dt}, {dt, dt, dt}));
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));

      string data_format;
      OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
      OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                  errors::InvalidArgument("Invalid data format"));
    }

    void Compute(OpKernelContext* context) override {
      const Tensor& input = context->input(0);
      const Tensor& output_grad = context->input(1);
      const Tensor& scale = context->input(2);
      const Tensor& saved_mean = context->input(3);
      const Tensor& saved_inv_var = context->input(4);

      OP_REQUIRES(context, input.shape().dims() == 4,
          errors::InvalidArgument("input must be 4-dimensional", input.shape().DebugString()));
      OP_REQUIRES(context, output_grad.shape().dims() == 4,
          errors::InvalidArgument("output_grad must be 4-dimensional", input.shape().DebugString()));
      OP_REQUIRES(context, scale.shape().dims() == 1,
          errors::InvalidArgument("scale must be 1-dimensional", scale.shape().DebugString()));
      OP_REQUIRES(context, saved_mean.shape().dims() == 1,
          errors::InvalidArgument("saved_mean must be 1-dimensional", saved_mean.shape().DebugString()));
      OP_REQUIRES(context, saved_inv_var.shape().dims() == 1,
          errors::InvalidArgument("saved_inv_var must be 1-dimensional", saved_inv_var.shape().DebugString()));

      OP_REQUIRES(context, input.shape() == output_grad.shape(),
          errors::InvalidArgument("input shape(", input.shape().DebugString(),
          "(does not match output_grad shape(", output_grad.shape().DebugString(), ")"));

      const int64 in_depths = GetTensorDim(input, data_format_, 'C');
      OP_REQUIRES(context, in_depths == scale.dim_size(0),
          errors::InvalidArgument("scale size does not match channels of input"));
      OP_REQUIRES(context, in_depths == saved_mean.dim_size(0),
          errors::InvalidArgument("saved_mean size does not match channels of input"));
      OP_REQUIRES(context, in_depths == saved_inv_var.dim_size(0),
          errors::InvalidArgument("saved_inv_var size does not match channels of input"));

      Tensor* input_grad = nullptr;
      TensorShape input_shape = input.shape();
      OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grad));

      TensorShape scale_bias_shape = scale.shape();
      Tensor* scale_grad = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, scale_bias_shape, &scale_grad));

      Tensor* bias_grad = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, scale_bias_shape, &bias_grad));

      LaunchBatchNormTrainingGrad<Device, T>::launch(
          context, epsilon_, data_format_,
          input, output_grad, scale, saved_mean, saved_inv_var,
          input_grad, scale_grad, bias_grad);
    }

  private:
    float epsilon_;
    TensorFormat data_format_;

    TF_DISALLOW_COPY_AND_ASSIGN(BatchNormTrainingGradOp);
};

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("BatchNormTraining").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    BatchNormTrainingOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(
    Name("BatchNormTrainingGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    BatchNormTrainingGradOp<GPUDevice, float>);

#endif // GOOGLE_CUDA

} // namespace tensorflow
