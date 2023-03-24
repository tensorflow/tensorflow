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
  void PadInput<GPUDevice, T, int, 4>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,       \
      const std::array<int, 2>& padding_left,                               \
      const std::array<int, 2>& padding_right,                              \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format,     \
      const T& padding_value);                                              \
  extern template struct PadInput<GPUDevice, T, int, 4>

DECLARE_GPU_SPEC(Eigen::bfloat16);
DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC

}  // namespace functor

template <>
void LaunchConv2DOp<GPUDevice, Eigen::bfloat16>::operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter, int row_dilation,
    int col_dilation, int row_stride, int col_stride, const Padding& padding,
    const std::vector<int64_t>& explicit_paddings, Tensor* output,
    TensorFormat data_format) {
  // Performant bfloat16 operations are supported for Ampere+ GPUs. For
  // pre-Ampere GPUs, we cast inputs to float and outputs back to bfloat16.
  auto* stream = ctx->op_device_context()->stream();
  const bool cast_to_float = !stream->GetCudaComputeCapability().IsAtLeast(
      se::CudaComputeCapability::AMPERE);

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

    LaunchConv2DOpImpl<float>(ctx, use_cudnn, cudnn_use_autotune, casted_input,
                              casted_filter, row_dilation, col_dilation,
                              row_stride, col_stride, padding,
                              explicit_paddings, &casted_out, data_format);

    functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
    const Tensor& casted_out_const = casted_out;
    cast_back(device, output->template flat<Eigen::bfloat16>(),
              casted_out_const.template flat<float>());
    return;
  }

  LaunchConv2DOpImpl<Eigen::bfloat16>(
      ctx, use_cudnn, cudnn_use_autotune, input_param, filter, row_dilation,
      col_dilation, row_stride, col_stride, padding, explicit_paddings, output,
      data_format);
}

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
    Conv2DOp<GPUDevice, Eigen::bfloat16>);

// Explicit instantiation.
template struct LaunchConv2DOp<GPUDevice, Eigen::bfloat16>;

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
