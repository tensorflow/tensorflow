/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

// Functor used by FusedBatchNormOp to do the computations.
template <typename Device, typename T>
struct FusedBatchNorm;
// Functor used by FusedBatchNormGradOp to do the computations.
template <typename Device, typename T>
struct FusedBatchNormGrad;

template <typename T>
struct FusedBatchNorm<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  const Tensor& estimated_mean_input,
                  const Tensor& estimated_variance_input, T epsilon,
                  Tensor* y_output, Tensor* batch_mean_output,
                  Tensor* batch_var_output, Tensor* saved_mean_output,
                  Tensor* saved_var_output, TensorFormat tensor_format,
                  bool is_training) {
    CHECK(tensor_format == FORMAT_NHWC)
        << "The CPU implementation of FusedBatchNorm only supports "
        << "NHWC tensor format for now.";
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<T>::ConstVec scale(scale_input.vec<T>());
    typename TTypes<T>::ConstVec offset(offset_input.vec<T>());
    typename TTypes<T>::ConstVec estimated_mean(estimated_mean_input.vec<T>());
    typename TTypes<T>::ConstVec estimated_variance(
        estimated_variance_input.vec<T>());
    typename TTypes<T, 4>::Tensor y(y_output->tensor<T, 4>());
    typename TTypes<T>::Vec batch_mean(batch_mean_output->vec<T>());
    typename TTypes<T>::Vec batch_var(batch_var_output->vec<T>());
    typename TTypes<T>::Vec saved_mean(saved_mean_output->vec<T>());
    typename TTypes<T>::Vec saved_var(saved_var_output->vec<T>());

    const CPUDevice& d = context->eigen_device<CPUDevice>();

    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduce_dims({0});
    Eigen::array<int, 2> bcast_spec({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0> > reduce_dims;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > bcast_spec;
    bcast_spec.set(0, rest_size);
#endif

    auto x_rest_by_depth = x.reshape(rest_by_depth);
    const int rest_size_minus_one = (rest_size > 1) ? (rest_size - 1) : 1;
    T rest_size_inv = static_cast<T>(1.0f / static_cast<T>(rest_size));
    // This adjustment is for Bessel's correction
    T rest_size_adjust =
        static_cast<T>(rest_size) / static_cast<T>(rest_size_minus_one);

    Eigen::Tensor<T, 1, Eigen::RowMajor> mean(depth);
    Eigen::Tensor<T, 1, Eigen::RowMajor> variance(depth);
    if (is_training) {
      mean.device(d) = (x_rest_by_depth.sum(reduce_dims) * rest_size_inv);
      batch_mean.device(d) = mean;
      saved_mean.device(d) = mean;
    } else {
      mean.device(d) = estimated_mean;
    }

    auto x_centered =
        x_rest_by_depth - mean.reshape(one_by_depth).broadcast(bcast_spec);

    if (is_training) {
      variance.device(d) = x_centered.square().sum(reduce_dims) * rest_size_inv;
      batch_var.device(d) = variance * rest_size_adjust;
      saved_var.device(d) = variance;
    } else {
      mean.device(d) = estimated_mean;
      variance.device(d) = estimated_variance;
    }

    auto scaling_factor = ((variance + epsilon).rsqrt() * scale)
                              .eval()
                              .reshape(one_by_depth)
                              .broadcast(bcast_spec);
    auto x_scaled = x_centered * scaling_factor;
    auto x_shifted =
        x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec);

    y.reshape(rest_by_depth).device(d) = x_shifted;
  }
};

template <typename T>
struct FusedBatchNormGrad<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& mean_input, const Tensor& variance_input,
                  T epsilon, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output, Tensor* offset_backprop_output,
                  TensorFormat tensor_format) {
    CHECK(tensor_format == FORMAT_NHWC)
        << "The CPU implementation of FusedBatchNorm only support "
        << "NHWC tensor format for now.";
    typename TTypes<T, 4>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<T>::ConstVec scale(scale_input.vec<T>());
    typename TTypes<T>::ConstVec mean(mean_input.vec<T>());
    typename TTypes<T>::ConstVec variance(variance_input.vec<T>());
    typename TTypes<T, 4>::Tensor x_backprop(x_backprop_output->tensor<T, 4>());
    typename TTypes<T>::Vec scale_backprop(scale_backprop_output->vec<T>());
    typename TTypes<T>::Vec offset_backprop(offset_backprop_output->vec<T>());

    // Note: the following formulas are used to to compute the gradients for
    // back propagation.
    // x_backprop = scale * rsqrt(variance + epsilon) *
    //              [y_backprop - mean(y_backprop) - (x - mean(x)) *
    //              mean(y_backprop * (x - mean(x))) / (variance + epsilon)]
    // scale_backprop = sum(y_backprop *
    //                  (x - mean(x)) * rsqrt(variance + epsilon))
    // offset_backprop = sum(y_backprop)

    const CPUDevice& d = context->eigen_device<CPUDevice>();
    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);
    Eigen::array<int, 1> reduce_dims({0});
    Eigen::array<int, 2> bcast_spec({rest_size, 1});
#else
    Eigen::IndexList<Eigen::type2index<1>, Eigen::Index> one_by_depth;
    one_by_depth.set(1, depth);
    Eigen::IndexList<Eigen::type2index<0> > reduce_dims;
    Eigen::IndexList<Eigen::Index, Eigen::type2index<1> > bcast_spec;
    bcast_spec.set(0, rest_size);
#endif

    auto x_rest_by_depth = x.reshape(rest_by_depth);
    T rest_size_inv = static_cast<T>(1.0f / static_cast<T>(rest_size));

    auto x_mean_rest_by_depth =
        mean.reshape(one_by_depth).broadcast(bcast_spec);
    auto x_centered = (x_rest_by_depth - x_mean_rest_by_depth).eval();
    auto coef0 = (variance + epsilon).rsqrt();
    auto coef0_rest_by_depth =
        coef0.eval().reshape(one_by_depth).broadcast(bcast_spec);
    auto x_scaled = x_centered * coef0_rest_by_depth;

    auto y_backprop_rest_by_depth = y_backprop.eval().reshape(rest_by_depth);
    scale_backprop.device(d) =
        (y_backprop_rest_by_depth * x_scaled).sum(reduce_dims);
    auto y_backprop_sum = y_backprop_rest_by_depth.sum(reduce_dims);
    offset_backprop.device(d) = y_backprop_sum;

    auto y_backprop_sum_one_by_depth =
        y_backprop_sum.eval().reshape(one_by_depth);
    auto y_backprop_mean_one_by_depth =
        y_backprop_sum_one_by_depth * rest_size_inv;
    auto y_backprop_mean_rest_by_depth =
        y_backprop_mean_one_by_depth.broadcast(bcast_spec);
    auto y_backprop_centered =
        y_backprop_rest_by_depth - y_backprop_mean_rest_by_depth;
    auto coef1 =
        (scale * coef0).eval().reshape(one_by_depth).broadcast(bcast_spec);
    auto coef2 = (coef0.square() *
                  (y_backprop_rest_by_depth * x_centered).mean(reduce_dims))
                     .eval()
                     .reshape(one_by_depth)
                     .broadcast(bcast_spec);
    x_backprop.reshape(rest_by_depth).device(d) =
        coef1 * (y_backprop_centered - x_centered * coef2);
  }
};

#if GOOGLE_CUDA
template <typename T>
struct FusedBatchNorm<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& x,
                  const Tensor& scale, const Tensor& offset,
                  const Tensor& estimated_mean,
                  const Tensor& estimated_variance, T epsilon, Tensor* y,
                  Tensor* batch_mean, Tensor* batch_var, Tensor* saved_mean,
                  Tensor* saved_inv_var, TensorFormat tensor_format,
                  bool is_training) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream avalible"));

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');
    VLOG(2) << "FusedBatchNorm:"
            << " batch_size: " << batch_size << " channels: " << channels
            << " height: " << height << " width:" << width
            << " x shape: " << x.shape().DebugString()
            << " scale shape: " << scale.shape().DebugString()
            << " offset shape: " << offset.shape().DebugString()
            << " tensor format: " << tensor_format;

    Tensor x_maybe_transformed = x;
    Tensor x_transformed;
    Tensor y_transformed;
    perftools::gputools::DeviceMemory<T> y_ptr;

    if (tensor_format == FORMAT_NCHW) {
      y_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*y);
    } else if (tensor_format == FORMAT_NHWC) {
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &x_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_maybe_transformed).tensor<T, 4>(),
          x_transformed.tensor<T, 4>());
      x_maybe_transformed = x_transformed;

      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &y_transformed));
      y_ptr = StreamExecutorUtil::AsDeviceMemory<T>(y_transformed);
    } else {
      context->SetStatus(
          errors::Internal("Unsupported tensor format: ", tensor_format));
      return;
    }

    perftools::gputools::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<T>(scale);
    auto offset_ptr = StreamExecutorUtil::AsDeviceMemory<T>(offset);
    auto estimated_mean_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(estimated_mean);
    auto estimated_variance_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(estimated_variance);
    auto batch_mean_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*batch_mean);

    auto batch_var_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*batch_var);
    auto saved_mean_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*saved_mean);
    auto saved_inv_var_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(*saved_inv_var);

    GPUDevice d = context->eigen_device<GPUDevice>();
    using perftools::gputools::DeviceMemory;
    Tensor inv_var;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        estimated_variance.shape(), &inv_var));
    auto inv_var_ptr = StreamExecutorUtil::AsDeviceMemory<T>(inv_var);
    std::function<const DeviceMemory<T>&()> var_to_inv_var =
        [d, epsilon, estimated_variance,
         &inv_var_ptr]() -> const DeviceMemory<T>& {
      auto estimated_variance_ptr =
          StreamExecutorUtil::AsDeviceMemory<T>(estimated_variance);
      const T* variance =
          static_cast<const T*>(estimated_variance_ptr.opaque());
      T* inv_variance = static_cast<T*>(inv_var_ptr.opaque());
      int channels = inv_var_ptr.ElementCount();
      VarianceToInvVariance<T>()(d, variance, epsilon, channels, inv_variance);
      return inv_var_ptr;
    };
    const int64 sample_size = batch_size * height * width;
    std::function<void()> inv_var_to_var = [d, &batch_var_ptr, epsilon,
                                            sample_size]() {
      T* variance = static_cast<T*>(batch_var_ptr.opaque());
      int channels = batch_var_ptr.ElementCount();
      InvVarianceToVariance<T>()(d, epsilon, sample_size, channels, variance);
    };

    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationForward(
                x_ptr, scale_ptr, offset_ptr, estimated_mean_ptr,
                estimated_variance_ptr, x_desc, scale_offset_desc,
                static_cast<double>(epsilon), &y_ptr, &batch_mean_ptr,
                &batch_var_ptr, &saved_mean_ptr, &saved_inv_var_ptr,
                is_training, std::move(var_to_inv_var),
                std::move(inv_var_to_var))
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(
          errors::Internal("cuDNN launch failure : input shape (",
                           x.shape().DebugString(), ")"));
    }
    if (tensor_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(y_transformed).tensor<T, 4>(),
          y->tensor<T, 4>());
    }
  }
};

template <typename T>
struct FusedBatchNormGrad<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop,
                  const Tensor& x, const Tensor& scale, const Tensor& mean,
                  const Tensor& variance, T epsilon, Tensor* x_backprop,
                  Tensor* scale_backprop, Tensor* offset_backprop,
                  TensorFormat tensor_format) {
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream avalible"));

    const int64 batch_size = GetTensorDim(x, tensor_format, 'N');
    const int64 channels = GetTensorDim(x, tensor_format, 'C');
    const int64 height = GetTensorDim(x, tensor_format, 'H');
    const int64 width = GetTensorDim(x, tensor_format, 'W');

    VLOG(2) << "FusedBatchNormGrad:"
            << " batch_size: " << batch_size << " channels: " << channels
            << " height: " << height << " width: " << width
            << " y_backprop shape: " << y_backprop.shape().DebugString()
            << " x shape: " << x.shape().DebugString()
            << " scale shape: " << scale.shape().DebugString()
            << " tensor format: " << tensor_format;

    // Inputs
    Tensor y_backprop_maybe_transformed = y_backprop;
    Tensor x_maybe_transformed = x;
    Tensor y_backprop_transformed;
    Tensor x_transformed;

    // Outputs
    Tensor x_backprop_transformed;
    perftools::gputools::DeviceMemory<T> x_backprop_ptr;

    if (tensor_format == FORMAT_NCHW) {
      x_backprop_ptr = StreamExecutorUtil::AsDeviceMemory<T>(*x_backprop);
    } else if (tensor_format == FORMAT_NHWC) {
      // Transform inputs from 'NHWC' to 'NCHW'
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &y_backprop_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(y_backprop_maybe_transformed)
              .tensor<T, 4>(),
          y_backprop_transformed.tensor<T, 4>());
      y_backprop_maybe_transformed = y_backprop_transformed;

      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &x_transformed));
      functor::NHWCToNCHW<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_maybe_transformed).tensor<T, 4>(),
          x_transformed.tensor<T, 4>());
      x_maybe_transformed = x_transformed;

      // Allocate memory for transformed outputs in 'NCHW'
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<T>::value,
                                  ShapeFromFormat(FORMAT_NCHW, batch_size,
                                                  height, width, channels),
                                  &x_backprop_transformed));
      x_backprop_ptr =
          StreamExecutorUtil::AsDeviceMemory<T>(x_backprop_transformed);
    } else {
      context->SetStatus(
          errors::Internal("Unsupported tensor format: ", tensor_format));
      return;
    }

    perftools::gputools::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    perftools::gputools::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);

    auto y_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(y_backprop_maybe_transformed);
    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<T>(scale);
    auto mean_ptr = StreamExecutorUtil::AsDeviceMemory<T>(mean);
    auto variance_ptr = StreamExecutorUtil::AsDeviceMemory<T>(variance);
    auto scale_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(*scale_backprop);
    auto offset_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(*offset_backprop);

    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationBackward(
                y_backprop_ptr, x_ptr, scale_ptr, mean_ptr, variance_ptr,
                x_desc, scale_offset_desc, static_cast<double>(epsilon),
                &x_backprop_ptr, &scale_backprop_ptr, &offset_backprop_ptr)
            .ok();

    if (!cudnn_launch_status) {
      context->SetStatus(
          errors::Internal("cuDNN launch failure : input shape (",
                           x.shape().DebugString(), ")"));
    }
    if (tensor_format == FORMAT_NHWC) {
      functor::NCHWToNHWC<GPUDevice, T, 4>()(
          context->eigen_device<GPUDevice>(),
          const_cast<const Tensor&>(x_backprop_transformed).tensor<T, 4>(),
          x_backprop->tensor<T, 4>());
    }
  }
};
#endif  // GOOGLE_CUDA
}  // namespace functor

template <typename Device, typename T>
class FusedBatchNormOp : public OpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& x = context->input(0);
    const Tensor& scale = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& estimated_mean = context->input(3);
    const Tensor& estimated_variance = context->input(4);

    OP_REQUIRES(context, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, offset.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        offset.shape().DebugString()));
    OP_REQUIRES(context, estimated_mean.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        estimated_mean.shape().DebugString()));
    OP_REQUIRES(
        context, estimated_variance.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                estimated_variance.shape().DebugString()));
    if (is_training_) {
      OP_REQUIRES(
          context, estimated_mean.dim_size(0) == 0,
          errors::InvalidArgument("estimated_mean empty for training",
                                  estimated_mean.shape().DebugString()));
      OP_REQUIRES(context, estimated_variance.dim_size(0) == 0,
                  errors::InvalidArgument(
                      "estimated_variance must be empty for training",
                      estimated_variance.shape().DebugString()));
    }

    Tensor* y = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, x.shape(), &y));
    Tensor* batch_mean = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, scale.shape(), &batch_mean));
    Tensor* batch_var = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, scale.shape(), &batch_var));
    Tensor* saved_mean = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(3, scale.shape(), &saved_mean));
    Tensor* saved_inv_var = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(4, scale.shape(), &saved_inv_var));

    functor::FusedBatchNorm<Device, T>()(
        context, x, scale, offset, estimated_mean, estimated_variance, epsilon_,
        y, batch_mean, batch_var, saved_mean, saved_inv_var, tensor_format_,
        is_training_);
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
};

template <typename Device, typename T>
class FusedBatchNormGradOp : public OpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = T(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& y_backprop = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& scale = context->input(2);
    const Tensor& saved_mean = context->input(3);
    // The Eigen implementation saves variance in the forward pass,  while cuDNN
    // saves inverted variance.
    const Tensor& saved_maybe_inv_var = context->input(4);

    OP_REQUIRES(context, y_backprop.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        y_backprop.shape().DebugString()));
    OP_REQUIRES(context, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(context, saved_mean.dims() == 1,
                errors::InvalidArgument("saved mean must be 1-dimensional",
                                        saved_mean.shape().DebugString()));
    OP_REQUIRES(
        context, saved_maybe_inv_var.dims() == 1,
        errors::InvalidArgument("saved variance must be 1-dimensional",
                                saved_maybe_inv_var.shape().DebugString()));

    Tensor* x_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, x.shape(), &x_backprop));

    const TensorShape& scale_offset_shape = scale.shape();
    Tensor* scale_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, scale_offset_shape,
                                                     &scale_backprop));
    Tensor* offset_backprop = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, scale_offset_shape,
                                                     &offset_backprop));
    // two placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    Tensor* placeholder_1 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(3, TensorShape({}), &placeholder_1));
    Tensor* placeholder_2 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, TensorShape({}), &placeholder_2));

    functor::FusedBatchNormGrad<Device, T>()(
        context, y_backprop, x, scale, saved_mean, saved_maybe_inv_var,
        epsilon_, x_backprop, scale_backprop, offset_backprop, tensor_format_);
  }

 private:
  T epsilon_;
  TensorFormat tensor_format_;
};

REGISTER_KERNEL_BUILDER(Name("FusedBatchNorm").Device(DEVICE_CPU),
                        FusedBatchNormOp<CPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGrad").Device(DEVICE_CPU),
                        FusedBatchNormGradOp<CPUDevice, float>);
#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("FusedBatchNorm").Device(DEVICE_GPU),
                        FusedBatchNormOp<GPUDevice, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGrad").Device(DEVICE_GPU),
                        FusedBatchNormGradOp<GPUDevice, float>);
#endif

}  // namespace tensorflow
