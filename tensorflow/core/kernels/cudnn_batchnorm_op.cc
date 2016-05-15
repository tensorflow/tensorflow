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

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
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
                     const float exponential_average_factor,
                     const Tensor& input,
                     const Tensor& scale_param, const Tensor& bias_param,
                     Tensor* output,
                     Tensor* running_mean,
                     Tensor* running_inv_var,
                     Tensor* save_mean,
                     Tensor* save_inv_var) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream avalible"));

    TensorFormat data_format = FORMAT_NCHW;
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

    auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());
    auto output_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                     output->template flat<T>().size());
    auto scale_ptr = AsDeviceMemory(scale_param.template flat<T>().data(),
                                    scale_param.template flat<T>().size());
    auto bias_ptr = AsDeviceMemory(bias_param.template flat<T>().data(),
                                    bias_param.template flat<T>().size());

    auto running_mean_ptr = AsDeviceMemory(running_mean->template flat<T>().data(),
                                    running_mean->template flat<T>().size());

    auto running_inv_var_ptr = AsDeviceMemory(running_inv_var->template flat<T>().data(),
                                    running_inv_var->template flat<T>().size());

    auto save_mean_ptr = AsDeviceMemory(save_mean->template flat<T>().data(),
                                        save_mean->template flat<T>().size());
    auto save_inv_var_ptr = AsDeviceMemory(save_inv_var->template flat<T>().data(),
                                        save_inv_var->template flat<T>().size());

    bool cudnn_launch_status =
      stream
        ->ThenBatchNormTrainingForward(epsilon,
                                    exponential_average_factor,
                                    input_desc,
                                    input_ptr,
                                    scale_bias_mean_var_desc,
                                    scale_ptr,
                                    bias_ptr,
                                    input_desc,
                                    &output_ptr,
                                    &running_mean_ptr,
                                    &running_inv_var_ptr,
                                    &save_mean_ptr,
                                    &save_inv_var_ptr)
        .ok();

      if (!cudnn_launch_status) {
        ctx->SetStatus(errors::Internal(
            "cuDNN launch failure : input shape(", input.shape().DebugString(),
            ")"));
      }
  }
};

template <typename Device, typename T>
class BatchNormTrainingOp : public OpKernel {
  public:
    explicit BatchNormTrainingOp(OpKernelConstruction* context) : OpKernel(context) {
      const DataType dt = DataTypeToEnum<T>::v();
      const DataType dt_ref = DataTypeToEnum<T>::ref();
      OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt, dt_ref, dt_ref}, {dt, dt, dt}));

      //Do some type checking here
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
      OP_REQUIRES_OK(context, context->GetAttr("exponential_average_factor",
                                               &exponential_average_factor_));
    }

    void Compute(OpKernelContext* context) override {
      //TODO a whole bunch of error checking
      const Tensor& input = context->input(0);
      const Tensor& scale = context->input(1);
      const Tensor& bias = context->input(2);

      Tensor running_mean = context->mutable_input(3, true);
      Tensor running_inv_var = context->mutable_input(4, true);

      Tensor* output = nullptr;
      TensorShape out_shape = input.shape();
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

      TensorFormat data_format = FORMAT_NCHW;
      const int64 in_depths = GetTensorDim(input, data_format, 'C');
      TensorShape save_mean_var_shape = TensorShape({in_depths});

      Tensor* save_mean = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, save_mean_var_shape, &save_mean));
      Tensor* save_inv_var = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, save_mean_var_shape, &save_inv_var));

      //TODO support other dimensions
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional", input.shape().DebugString()));

      LaunchBatchNormTraining<Device, T>::launch(
          context, epsilon_, exponential_average_factor_,
          input, scale, bias, output,
          &running_mean, &running_inv_var, save_mean, save_inv_var);
    }

  private:
    float epsilon_;
    float exponential_average_factor_;

    TF_DISALLOW_COPY_AND_ASSIGN(BatchNormTrainingOp);
};

template <typename Device, typename T>
struct LaunchBatchNormTrainingGrad;

template <typename T>
struct LaunchBatchNormTrainingGrad<GPUDevice, T> {
  static void launch(OpKernelContext* ctx,
                     const float epsilon,
                     const Tensor& input,
                     const Tensor& output_grad,
                     const Tensor& scale_param,
                     const Tensor& saved_mean,
                     const Tensor& saved_inv_var,
                     Tensor* input_grad,
                     Tensor* scale_grad,
                     Tensor* bias_grad) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream avalible"));

    TensorFormat data_format = FORMAT_NCHW;
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

    auto input_grad_ptr = AsDeviceMemory(input_grad->template flat<T>().data(),
                                         input_grad->template flat<T>().size());
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
  }
};

template <typename Device, typename T>
class BatchNormTrainingGradOp : public OpKernel {
  public:
    explicit BatchNormTrainingGradOp(OpKernelConstruction* context) : OpKernel(context) {
      const DataType dt = DataTypeToEnum<T>::v();
      OP_REQUIRES_OK(context, context->MatchSignature({dt, dt, dt, dt, dt}, {dt, dt, dt}));

      //Do some type checking here
      OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* context) override {
      //TODO a whole bunch of error checking
      const Tensor& input = context->input(0);
      const Tensor& output_grad = context->input(1);
      const Tensor& scale = context->input(2);
      const Tensor& saved_mean = context->input(3);
      const Tensor& saved_inv_var = context->input(4);

      Tensor* input_grad = nullptr;
      TensorShape input_shape = input.shape();
      OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &input_grad));

      TensorShape scale_bias_shape = scale.shape();
      Tensor* scale_grad = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(1, scale_bias_shape, &scale_grad));

      Tensor* bias_grad = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(2, scale_bias_shape, &bias_grad));

      //TODO support other dimensions
      OP_REQUIRES(context, input.dims() == 4,
                  errors::InvalidArgument("input must be 4-dimensional", input.shape().DebugString()));

      LaunchBatchNormTrainingGrad<Device, T>::launch(
          context, epsilon_, input, output_grad, scale, saved_mean, saved_inv_var,
          input_grad, scale_grad, bias_grad);
    }

  private:
    float epsilon_;

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
