/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#include "tensorflow/core/kernels/conv_ops_impl.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Explicit instantiation.
template struct LaunchConv2DOp<CPUDevice, Eigen::bfloat16>;
template struct Conv2DOp<CPUDevice, Eigen::bfloat16>;

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
#if !defined(USE_GEMM_FOR_CONV)
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"),
    Conv2DOp<CPUDevice, bfloat16>);
#endif  // USE_GEMM_FOR_CONV
REGISTER_KERNEL_BUILDER(
    Name("Conv").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"),
    ConvOp<CPUDevice, bfloat16>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                 \
  template <>                                                               \
  void SpatialConvolution<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,             \
      typename TTypes<T, 4>::ConstTensor input,                             \
      typename TTypes<T, 4>::ConstTensor filter, int row_stride,            \
      int col_stride, int row_dilation, int col_dilation,                   \
      const Eigen::PaddingType& padding,                                    \
      const Eigen::NoOpOutputKernel& output_kernel);                        \
  template <>                                                               \
  void SpatialConvolution<GPUDevice, T>::operator()(                        \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,             \
      typename TTypes<T, 4>::ConstTensor input,                             \
      typename TTypes<T, 4>::ConstTensor filter, int row_stride,            \
      int col_stride, int row_dilation, int col_dilation, int padding_top,  \
      int padding_bottom, int padding_left, int padding_right,              \
      const Eigen::NoOpOutputKernel& output_kernel);                        \
  extern template struct SpatialConvolution<GPUDevice, T>;                  \
  template <>                                                               \
  void MatMulConvFunctor<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T, 2>::Tensor out,                \
      typename TTypes<T, 2>::ConstTensor in0,                               \
      typename TTypes<T, 2>::ConstTensor in1,                               \
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair, \
      const Eigen::NoOpOutputKernel& output_kernel);                        \
  extern template struct MatMulConvFunctor<GPUDevice, T>;                   \
  template <>                                                               \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(                   \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,             \
      typename TTypes<T, 4, int>::ConstTensor in,                           \
      typename TTypes<T, 4, int>::Tensor out);                              \
  extern template struct TransformFilter<GPUDevice, T, int, 4>;             \
  template <>                                                               \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(                   \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,             \
      typename TTypes<T, 5, int>::ConstTensor in,                           \
      typename TTypes<T, 5, int>::Tensor out);                              \
  extern template struct TransformFilter<GPUDevice, T, int, 5>;             \
  template <>                                                               \
  void PadInput<GPUDevice, T, int, 4>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,       \
      const std::array<int, 2>& padding_left,                               \
      const std::array<int, 2>& padding_right,                              \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format,     \
      const T& padding_value);                                              \
  extern template struct PadInput<GPUDevice, T, int, 4>;                    \
  template <>                                                               \
  void PadInput<GPUDevice, T, int, 5>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in,       \
      const std::array<int, 3>& padding_left,                               \
      const std::array<int, 3>& padding_right,                              \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat data_format,     \
      const T& padding_value);                                              \
  extern template struct PadInput<GPUDevice, T, int, 5>

DECLARE_GPU_SPEC(Eigen::bfloat16);
DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC

}  // namespace functor

template <>
void LaunchConvOp<GPUDevice, Eigen::bfloat16>::operator()(
    OpKernelContext* context, bool cudnn_use_autotune, const Tensor& input,
    const Tensor& filter, const std::vector<int64>& dilations,
    const std::vector<int64>& strides, const Padding padding,
    const std::vector<int64_t>& explicit_paddings, TensorFormat data_format,
    Tensor* output) {
  // Get spatial dims for dilations and strides.
  int spatial_dims = input.dims() - 2;
  gtl::InlinedVector<int64_t, 3> strides_spatial(spatial_dims);
  gtl::InlinedVector<int64_t, 3> dilations_spatial(spatial_dims);
  for (int i = 0; i < spatial_dims; ++i) {
    strides_spatial[i] =
        GetTensorDim(strides, data_format, static_cast<char>(i + '0'));
    dilations_spatial[i] =
        GetTensorDim(dilations, data_format, static_cast<char>(i + '0'));
  }
  auto* stream = context->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);

  if (cast_to_float) {
    Tensor casted_input = input;
    Tensor casted_filter = filter;
    Tensor casted_out = *output;

    const GPUDevice& device = context->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, input.shape(),
                                                   &casted_input));
    cast(device, casted_input.template flat<float>(),
         input.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, filter.shape(),
                                                   &casted_filter));
    cast(device, casted_filter.template flat<float>(),
         filter.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, output->shape(),
                                                   &casted_out));

    LaunchConvOpImpl<float>(context, cudnn_use_autotune, casted_input,
                            casted_filter, dilations_spatial, strides_spatial,
                            padding, explicit_paddings, data_format,
                            &casted_out);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_out_const = casted_out;
    cast_back(device, output->template flat<Eigen::bfloat16>(),
              casted_out_const.template flat<float>());
    return;
  }

  LaunchConvOpImpl<Eigen::bfloat16>(context, cudnn_use_autotune, input, filter,
                                    dilations_spatial, strides_spatial, padding,
                                    explicit_paddings, data_format, output);
}

template <>
void LaunchConv2DOp<GPUDevice, Eigen::bfloat16>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* output,
    TensorFormat data_format) {
  // Cast strides and dilations.
  gtl::InlinedVector<int64_t, 3> casted_strides = {row_stride, col_stride};
  gtl::InlinedVector<int64_t, 3> casted_dilations = {row_dilation,
                                                     col_dilation};

  auto* stream = ctx->op_device_context()->stream();
  const bool cast_to_float = !IsBF16SupportedInOps(stream);

  if (cast_to_float) {
    Tensor casted_input = input_param;
    Tensor casted_filter = filter;
    Tensor casted_out = *output;

    const GPUDevice& device = ctx->eigen_device<GPUDevice>();
    functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_FLOAT, input_param.shape(), &casted_input));
    cast(device, casted_input.template flat<float>(),
         input_param.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DT_FLOAT, filter.shape(), &casted_filter));
    cast(device, casted_filter.template flat<float>(),
         filter.template flat<Eigen::bfloat16>());

    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DT_FLOAT, output->shape(), &casted_out));

    LaunchConvOpImpl<float>(
        ctx, cudnn_use_autotune, casted_input, casted_filter, casted_dilations,
        casted_strides, padding, explicit_paddings, data_format, &casted_out);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_out_const = casted_out;
    cast_back(device, output->template flat<Eigen::bfloat16>(),
              casted_out_const.template flat<float>());
    return;
  }

  LaunchConvOpImpl<Eigen::bfloat16>(
      ctx, cudnn_use_autotune, input_param, filter, casted_dilations,
      casted_strides, padding, explicit_paddings, data_format, output);
}

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
    Conv2DOp<GPUDevice, Eigen::bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("Conv").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
    ConvOp<GPUDevice, Eigen::bfloat16>);

// Explicit instantiation.
template struct LaunchConv2DOp<GPUDevice, Eigen::bfloat16>;
template struct LaunchConvOp<GPUDevice, Eigen::bfloat16>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
