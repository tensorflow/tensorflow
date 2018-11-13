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
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

// Functor used by FusedBatchNormOp to do the computations.
template <typename Device, typename T, typename U>
struct FusedBatchNorm;
// Functor used by FusedBatchNormGradOp to do the computations when
// is_training=True.
template <typename Device, typename T, typename U>
struct FusedBatchNormGrad;

template <bool IsSame, typename Y, typename X, typename T>
struct CastIfNecessary {
  static inline void process(
      Y& y, X& x_shifted, const Eigen::DSizes<Eigen::Index, 2>& rest_by_depth,
      const CPUDevice& d) {
    y.reshape(rest_by_depth).device(d) = x_shifted.template cast<T>();
  }
};

template <typename Y, typename X, typename T>
struct CastIfNecessary<true, Y, X, T> {
  static inline void process(
      Y& y, X& x_shifted, const Eigen::DSizes<Eigen::Index, 2>& rest_by_depth,
      const CPUDevice& d) {
    y.reshape(rest_by_depth).device(d) = x_shifted;
  }
};

template <typename T, typename U>
struct FusedBatchNorm<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x_input,
                  const Tensor& scale_input, const Tensor& offset_input,
                  const Tensor& estimated_mean_input,
                  const Tensor& estimated_variance_input, U epsilon,
                  Tensor* y_output, Tensor* batch_mean_output,
                  Tensor* batch_var_output, Tensor* saved_mean_output,
                  Tensor* saved_var_output, TensorFormat tensor_format,
                  bool is_training) {
    OP_REQUIRES(context, tensor_format == FORMAT_NHWC,
                errors::Internal("The CPU implementation of FusedBatchNorm "
                                 "only supports NHWC tensor format for now."));
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_mean(estimated_mean_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_variance(
        estimated_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor y(y_output->tensor<T, 4>());
    typename TTypes<U>::Vec batch_mean(batch_mean_output->vec<U>());
    typename TTypes<U>::Vec batch_var(batch_var_output->vec<U>());
    typename TTypes<U>::Vec saved_mean(saved_mean_output->vec<U>());
    typename TTypes<U>::Vec saved_var(saved_var_output->vec<U>());

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

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    const int rest_size_minus_one = (rest_size > 1) ? (rest_size - 1) : 1;
    U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));
    // This adjustment is for Bessel's correction
    U rest_size_adjust =
        static_cast<U>(rest_size) / static_cast<U>(rest_size_minus_one);

    Eigen::Tensor<U, 1, Eigen::RowMajor> mean(depth);
    Eigen::Tensor<U, 1, Eigen::RowMajor> variance(depth);
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
      variance.device(d) = estimated_variance;
    }

    auto scaling_factor = ((variance + epsilon).rsqrt() * scale)
                              .eval()
                              .reshape(one_by_depth)
                              .broadcast(bcast_spec);
    auto x_scaled = x_centered * scaling_factor;
    auto x_shifted =
        x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec);

    // Explicitly checks the types of T and U and only casts x_shifted when
    // T != U. (Not doing so caused a 35-50% performance slowdown for
    // some compiler flags.)
    CastIfNecessary<std::is_same<T, U>::value, decltype(y), decltype(x_shifted),
                    T>::process(y, x_shifted, rest_by_depth, d);
  }
};

template <typename T, typename U>
struct FusedBatchNormGrad<CPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop_input,
                  const Tensor& x_input, const Tensor& scale_input,
                  const Tensor& mean_input, const Tensor& variance_input,
                  U epsilon, Tensor* x_backprop_output,
                  Tensor* scale_backprop_output, Tensor* offset_backprop_output,
                  TensorFormat tensor_format) {
    OP_REQUIRES(context, tensor_format == FORMAT_NHWC,
                errors::Internal("The CPU implementation of FusedBatchNormGrad "
                                 "only supports NHWC tensor format for now."));
    typename TTypes<T, 4>::ConstTensor y_backprop(
        y_backprop_input.tensor<T, 4>());
    typename TTypes<T, 4>::ConstTensor x(x_input.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec mean(mean_input.vec<U>());
    typename TTypes<U>::ConstVec variance(variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor x_backprop(x_backprop_output->tensor<T, 4>());
    typename TTypes<U>::Vec scale_backprop(scale_backprop_output->vec<U>());
    typename TTypes<U>::Vec offset_backprop(offset_backprop_output->vec<U>());

    // Note: the following formulas are used to compute the gradients for
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

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    U rest_size_inv = static_cast<U>(1.0f / static_cast<U>(rest_size));

    auto x_mean_rest_by_depth =
        mean.reshape(one_by_depth).broadcast(bcast_spec);
    auto x_centered = (x_rest_by_depth - x_mean_rest_by_depth).eval();
    auto coef0 = (variance + epsilon).rsqrt();
    auto coef0_rest_by_depth =
        coef0.eval().reshape(one_by_depth).broadcast(bcast_spec);
    auto x_scaled = x_centered * coef0_rest_by_depth;

    auto y_backprop_rest_by_depth =
        y_backprop.eval().reshape(rest_by_depth).template cast<U>();
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
        (coef1 * (y_backprop_centered - x_centered * coef2)).template cast<T>();
  }
};

#if GOOGLE_CUDA
template <typename T, typename U>
struct FusedBatchNorm<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& x,
                  const Tensor& scale, const Tensor& offset,
                  const Tensor& estimated_mean,
                  const Tensor& estimated_variance, U epsilon, Tensor* y,
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

    // If input is empty, return NaN mean/variance
    if (x.shape().num_elements() == 0) {
      functor::SetNanFunctor<U> f;
      f(context->eigen_device<GPUDevice>(), batch_mean->flat<U>());
      f(context->eigen_device<GPUDevice>(), batch_var->flat<U>());
      return;
    }

    Tensor x_maybe_transformed = x;
    Tensor x_transformed;
    Tensor y_transformed;
    se::DeviceMemory<T> y_ptr;

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

    se::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    se::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<U>(scale);
    auto offset_ptr = StreamExecutorUtil::AsDeviceMemory<U>(offset);
    auto estimated_mean_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(estimated_mean);
    auto estimated_variance_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(estimated_variance);
    auto batch_mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*batch_mean);

    auto batch_var_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*batch_var);
    auto saved_mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(*saved_mean);
    auto saved_inv_var_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*saved_inv_var);

    GPUDevice d = context->eigen_device<GPUDevice>();
    using se::DeviceMemory;
    Tensor inv_var;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<U>::value,
                                        estimated_variance.shape(), &inv_var));
    auto inv_var_ptr = StreamExecutorUtil::AsDeviceMemory<U>(inv_var);
    std::function<const DeviceMemory<U>&()> var_to_inv_var =
        [d, epsilon, estimated_variance,
         &inv_var_ptr]() -> const DeviceMemory<U>& {
      auto estimated_variance_ptr =
          StreamExecutorUtil::AsDeviceMemory<U>(estimated_variance);
      const U* variance =
          static_cast<const U*>(estimated_variance_ptr.opaque());
      U* inv_variance = static_cast<U*>(inv_var_ptr.opaque());
      int channels = inv_var_ptr.ElementCount();
      VarianceToInvVariance<U>()(d, variance, epsilon, channels, inv_variance);
      return inv_var_ptr;
    };
    const int64 sample_size = batch_size * height * width;
    std::function<void()> inv_var_to_var = [d, &batch_var_ptr, epsilon,
                                            sample_size]() {
      U* variance = static_cast<U*>(batch_var_ptr.opaque());
      int channels = batch_var_ptr.ElementCount();
      InvVarianceToVariance<U>()(d, epsilon, sample_size, channels, variance);
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

template <typename T, typename U>
struct FusedBatchNormGrad<GPUDevice, T, U> {
  void operator()(OpKernelContext* context, const Tensor& y_backprop,
                  const Tensor& x, const Tensor& scale, const Tensor& mean,
                  const Tensor& inv_variance, U epsilon, Tensor* x_backprop,
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
    se::DeviceMemory<T> x_backprop_ptr;

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

    se::dnn::BatchDescriptor x_desc;
    x_desc.set_count(batch_size)
        .set_feature_map_count(channels)
        .set_height(height)
        .set_width(width)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    se::dnn::BatchDescriptor scale_offset_desc;
    scale_offset_desc.set_count(1)
        .set_feature_map_count(channels)
        .set_height(1)
        .set_width(1)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    auto y_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<T>(y_backprop_maybe_transformed);
    auto x_ptr = StreamExecutorUtil::AsDeviceMemory<T>(x_maybe_transformed);
    auto scale_ptr = StreamExecutorUtil::AsDeviceMemory<U>(scale);
    auto mean_ptr = StreamExecutorUtil::AsDeviceMemory<U>(mean);
    auto inv_variance_ptr = StreamExecutorUtil::AsDeviceMemory<U>(inv_variance);
    auto scale_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*scale_backprop);
    auto offset_backprop_ptr =
        StreamExecutorUtil::AsDeviceMemory<U>(*offset_backprop);

    // the cudnn kernel outputs inverse variance in forward and reuse it in
    // backward
    bool cudnn_launch_status =
        stream
            ->ThenBatchNormalizationBackward(
                y_backprop_ptr, x_ptr, scale_ptr, mean_ptr, inv_variance_ptr,
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

// Forward declarations of the functor specializations for GPU.
#define DECLARE_GPU_SPEC(T, U)                                           \
  template <>                                                            \
  void FusedBatchNormFreezeGrad<GPUDevice, T, U>::operator()(            \
      const GPUDevice& d, const Tensor& y_backprop_input,                \
      const Tensor& x_input, const Tensor& scale_input,                  \
      const Tensor& mean_input, const Tensor& variance_input, U epsilon, \
      Tensor* x_backprop_output, Tensor* scale_backprop_output,          \
      Tensor* offset_backprop_output, typename TTypes<U>::Vec scratch1,  \
      typename TTypes<U>::Vec scratch2);                                 \
  extern template struct FusedBatchNormFreezeGrad<GPUDevice, T, U>;
DECLARE_GPU_SPEC(float, float);
DECLARE_GPU_SPEC(Eigen::half, float);

#endif  // GOOGLE_CUDA
}  // namespace functor

template <typename Device, typename T, typename U>
class FusedBatchNormOp : public OpKernel {
 public:
  explicit FusedBatchNormOp(OpKernelConstruction* context) : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
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
          errors::InvalidArgument("estimated_mean must be empty for training",
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
    Tensor* saved_maybe_inv_var = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(4, scale.shape(),
                                                     &saved_maybe_inv_var));

    functor::FusedBatchNorm<Device, T, U>()(
        context, x, scale, offset, estimated_mean, estimated_variance, epsilon_,
        y, batch_mean, batch_var, saved_mean, saved_maybe_inv_var,
        tensor_format_, is_training_);
  }

 private:
  U epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
};

template <typename Device, typename T, typename U>
class FusedBatchNormGradOp : public OpKernel {
 public:
  explicit FusedBatchNormGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = U(epsilon);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& y_backprop = context->input(0);
    const Tensor& x = context->input(1);
    const Tensor& scale = context->input(2);
    // When is_training=True, batch mean and variance/inverted variance are
    // saved in the forward pass to be reused here. When is_training=False,
    // population mean and variance need to be forwarded here to compute the
    // gradients.
    const Tensor& saved_mean_or_pop_mean = context->input(3);
    // The Eigen implementation saves variance in the forward pass, while cuDNN
    // saves inverted variance.
    const Tensor& saved_maybe_inv_var_or_pop_var = context->input(4);

    OP_REQUIRES(context, y_backprop.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        y_backprop.shape().DebugString()));
    OP_REQUIRES(context, x.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        x.shape().DebugString()));
    OP_REQUIRES(context, scale.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale.shape().DebugString()));
    OP_REQUIRES(
        context, saved_mean_or_pop_mean.dims() == 1,
        errors::InvalidArgument("saved mean must be 1-dimensional",
                                saved_mean_or_pop_mean.shape().DebugString()));
    OP_REQUIRES(context, saved_maybe_inv_var_or_pop_var.dims() == 1,
                errors::InvalidArgument(
                    "saved variance must be 1-dimensional",
                    saved_maybe_inv_var_or_pop_var.shape().DebugString()));

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
    // Two placeholders for estimated_mean and estimated_variance, which are
    // used for inference and thus not needed here for gradient computation.
    // They are filled with zeros so as to avoid NaN outputs.
    Tensor* placeholder_1 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(3, TensorShape({}), &placeholder_1));
    functor::SetZeroFunctor<Device, float> f;
    f(context->eigen_device<Device>(), placeholder_1->flat<U>());
    Tensor* placeholder_2 = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(4, TensorShape({}), &placeholder_2));
    f(context->eigen_device<Device>(), placeholder_2->flat<U>());

    // If input is empty, set gradients w.r.t scale/offset to zero.
    if (x.shape().num_elements() == 0) {
      functor::SetZeroFunctor<Device, U> f;
      f(context->eigen_device<Device>(), scale_backprop->flat<U>());
      f(context->eigen_device<Device>(), offset_backprop->flat<U>());
      return;
    }

    if (is_training_) {
      functor::FusedBatchNormGrad<Device, T, U>()(
          context, y_backprop, x, scale, saved_mean_or_pop_mean,
          saved_maybe_inv_var_or_pop_var, epsilon_, x_backprop, scale_backprop,
          offset_backprop, tensor_format_);

    } else {
      // Necessary layout conversion is currently done in python.
      CHECK(tensor_format_ == FORMAT_NHWC)
          << "The implementation of FusedBatchNormGrad with is_training=False "
             "only support "
          << "NHWC tensor format for now.";
      Tensor scratch1, scratch2;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            scale_offset_shape, &scratch1));
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<U>::value,
                                            scale_offset_shape, &scratch2));
      functor::FusedBatchNormFreezeGrad<Device, T, U>()(
          context->eigen_device<Device>(), y_backprop, x, scale,
          saved_mean_or_pop_mean, saved_maybe_inv_var_or_pop_var, epsilon_,
          x_backprop, scale_backprop, offset_backprop, scratch1.vec<U>(),
          scratch2.vec<U>());
    }
  }

 private:
  U epsilon_;
  TensorFormat tensor_format_;
  bool is_training_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedBatchNormOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNormGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedBatchNormGradOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<CPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<CPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<CPUDevice, Eigen::half, float>);

#if GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNorm").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    FusedBatchNormOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedBatchNormGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    FusedBatchNormGradOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<GPUDevice, float, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormOp<GPUDevice, Eigen::half, float>);

REGISTER_KERNEL_BUILDER(Name("FusedBatchNormGradV2")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<Eigen::half>("T")
                            .TypeConstraint<float>("U"),
                        FusedBatchNormGradOp<GPUDevice, Eigen::half, float>);

#endif

}  // namespace tensorflow
