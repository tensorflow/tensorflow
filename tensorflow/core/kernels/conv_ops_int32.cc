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
template struct LaunchConv2DOp<CPUDevice, int32>;
template struct Conv2DOp<CPUDevice, int32>;

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
#if !defined(USE_GEMM_FOR_CONV)
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    Conv2DOp<CPUDevice, int32>);
#endif  // USE_GEMM_FOR_CONV
REGISTER_KERNEL_BUILDER(
    Name("Conv").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    ConvOp<CPUDevice, int32>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <>
struct LaunchConv2DOp<GPUDevice, int32> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_dilation,
                  int col_dilation, int row_stride, int col_stride,
                  const Padding& padding,
                  const std::vector<int64_t>& explicit_paddings, Tensor* output,
                  TensorFormat data_format) {
    if (data_format != FORMAT_NHWC) {
      ctx->SetStatus(
          errors::Unimplemented("The Conv2D op currently only supports the "
                                "NHWC tensor format for integer types. "
                                "The op was given the format: ",
                                ToString(data_format)));
      return;
    }
    const int64_t in_depth = GetTensorDim(input, data_format, 'C');
    OP_REQUIRES(ctx, in_depth == filter.dim_size(2),
                errors::Unimplemented(
                    "The Conv2D op currently does not support grouped "
                    "convolutions for integer types. A grouped convolution was "
                    "attempted to be run because the input depth of ",
                    in_depth, " does not match the filter input depth of ",
                    filter.dim_size(2)));
    OP_REQUIRES(
        ctx, filter.NumElements() > 0,
        errors::InvalidArgument("filter must not have zero elements "
                                "(i.e. all dimensions must be non-zero)"));

    for (int64_t explicit_padding : explicit_paddings) {
      if (!FastBoundsCheck(explicit_padding, std::numeric_limits<int>::max())) {
        ctx->SetStatus(errors::InvalidArgument("filter too large"));
        return;
      }
    }
    LaunchGeneric<GPUDevice, int32>()(
        ctx, input, filter, row_stride, col_stride, row_dilation, col_dilation,
        padding, explicit_paddings, output, data_format);
  }
};

template <>
struct LaunchConvOp<GPUDevice, int32> {
  void operator()(OpKernelContext* context, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter,
                  const std::vector<int64>& dilations,
                  const std::vector<int64>& strides, const Padding padding,
                  const std::vector<int64_t>& explicit_paddings,
                  TensorFormat data_format, Tensor* output) {
    // Cuda backend does not support int32. For 2D we fall back to Conv2D Eigen
    // based implementation and for 3D we throw an error.
    int spatial_dims = input.dims() - 2;
    if (spatial_dims == 2) {
      LaunchConv2DOp<GPUDevice, int32>()(
          context, true, cudnn_use_autotune, input, filter, dilations[1],
          dilations[2], strides[1], strides[2], padding, explicit_paddings,
          output, data_format);
    } else if (spatial_dims == 3) {
      context->SetStatus(absl::UnimplementedError(
          "3D Convolution does not support int32 data type."));
    } else {
      context->SetStatus(absl::InternalError("Invalid spatial dimensions."));
    }
  }
};

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

DECLARE_GPU_SPEC(int32);
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<int32>("T"),
    Conv2DOp<GPUDevice, int32>);
REGISTER_KERNEL_BUILDER(
    Name("Conv").Device(DEVICE_GPU).TypeConstraint<int32>("T"),
    ConvOp<GPUDevice, int32>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
