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

// Functors for 3d convolution.

#ifndef TENSORFLOW_CORE_KERNELS_CONV_3D_H_
#define TENSORFLOW_CORE_KERNELS_CONV_3D_H_

#include <array>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_cuboid_convolutions.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"

namespace tensorflow {
namespace functor {

// Applies a 3D convolution to a batch of multi-channel volumes.
template <typename Device, typename T>
struct CuboidConvolution;

// Backward input pass for the cuboid convolution.
template <typename Device, typename T>
struct CuboidConvolutionBackwardInput;

// Backward filter pass for the cuboid convolution.
template <typename Device, typename T>
struct CuboidConvolutionBackwardFilter;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct CuboidConvolution<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 5>::Tensor output,
                  typename TTypes<T, 5>::ConstTensor input,
                  typename TTypes<T, 5>::ConstTensor filter, int stride_planes,
                  int stride_rows, int stride_cols,
                  const Eigen::PaddingType& padding) {
    output.device(d) = Eigen::CuboidConvolution(
        input, filter, stride_planes, stride_rows, stride_cols, padding);
  }
};

template <typename T>
struct CuboidConvolutionBackwardInput<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T, 5>::Tensor input_backward,
                  typename TTypes<T, 5>::ConstTensor filter,
                  typename TTypes<T, 5>::ConstTensor output_backward,
                  int stride_planes, int stride_rows, int stride_cols) {
    // Need to swap the order of plane/row/col strides when calling Eigen.
    input_backward.device(d) = Eigen::CuboidConvolutionBackwardInput(
        filter, output_backward,
        input_backward.dimension(3),  // input_planes
        input_backward.dimension(2),  // input_rows
        input_backward.dimension(1),  // input_cols
        stride_cols, stride_rows, stride_planes);
  }
};

template <typename T>
struct CuboidConvolutionBackwardFilter<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<T, 5>::Tensor filter_backward,
                  typename TTypes<T, 5>::ConstTensor input,
                  typename TTypes<T, 5>::ConstTensor output_backward,
                  int stride_planes, int stride_rows, int stride_cols) {
    // Need to swap the order of plane/row/col strides when calling Eigen.
    filter_backward.device(d) = Eigen::CuboidConvolutionBackwardKernel(
        input, output_backward,
        filter_backward.dimension(2),  // kernel_planes
        filter_backward.dimension(1),  // kernel_rows
        filter_backward.dimension(0),  // kernel_cols
        stride_cols, stride_rows, stride_planes);
  }
};

}  // namespace functor

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
struct LaunchConv3DOp;

template <typename T>
struct LaunchConv3DOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, bool cudnn_use_autotune,
                     const Tensor& input, const Tensor& filter,
                     const std::array<int64, 3>& dilations,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    OP_REQUIRES(context, data_format == FORMAT_NHWC,
                absl::InvalidArgumentError("CPU implementation of Conv3D "
                                           "currently only supports the NHWC "
                                           "tensor format."));
    OP_REQUIRES(
        context, dilations[0] == 1 && dilations[1] == 1 && dilations[2] == 1,
        absl::InvalidArgumentError("CPU implementation of Conv3D "
                                   "currently only supports dilated rates "
                                   "of 1."));
    OP_REQUIRES(context, filter.dim_size(3) == input.dim_size(input.dims() - 1),
                absl::InvalidArgumentError(absl::StrCat(
                    "Number of channels in filter (", filter.dim_size(3),
                    ") must match last dimension of input (",
                    input.dim_size(input.dims() - 1), ")")));
    functor::CuboidConvolution<CPUDevice, T>()(
        context->eigen_device<CPUDevice>(), output->tensor<T, 5>(),
        input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
        strides[0], BrainPadding2EigenPadding(padding));
  }
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_3D_H_
