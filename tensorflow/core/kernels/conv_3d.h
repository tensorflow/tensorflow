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

#ifndef TENSORFLOW_KERNELS_CONV_3D_H_
#define TENSORFLOW_KERNELS_CONV_3D_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"
#include "tensorflow/core/kernels/eigen_backward_cuboid_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {


template <typename Device, typename Input, typename Filter, typename Output>
void CuboidConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int depth_stride, int row_stride,
                            int col_stride, const Eigen::PaddingType& padding) {
  // Need to shuffle DHW -> WHD for eigen
  output.device(d) =
      Eigen::CuboidConvolution(input, filter, col_stride, row_stride,
                               depth_stride, padding);
};

template <typename Device, typename T>
struct CuboidConvolution {
  void operator()(const Device& d, typename TTypes<T, 5>::Tensor output,
                  typename TTypes<T, 5>::ConstTensor input,
                  typename TTypes<T, 5>::ConstTensor filter, int depth_stride,
                  int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    CuboidConvolutionFunc(d, output, input, filter, depth_stride, row_stride,
                          col_stride, padding);
  }
};

template <typename Device, typename T>
struct CuboidConvolutionBackwardInput {
  void operator()(const Device& d, typename TTypes<T, 5>::Tensor input_backward,
                  typename TTypes<T, 5>::ConstTensor kernel,
                  typename TTypes<T, 5>::ConstTensor output_backward,
                  int input_depth, int input_rows, int input_cols,
                  int depth_stride, int row_stride, int col_stride) {
    // Need to shuffle DHW -> WHD for eigen
    input_backward.device(d) = Eigen::CuboidConvolutionBackwardInput(
        kernel, output_backward,
        input_cols, input_rows, input_depth,
        col_stride, row_stride, depth_stride);
  }
};

template <typename Device, typename T>
struct CuboidConvolutionBackwardKernel {
  void operator()(const Device& d,
                  typename TTypes<T, 5>::Tensor kernel_backward,
                  typename TTypes<T, 5>::ConstTensor input,
                  typename TTypes<T, 5>::ConstTensor output_backward,
                  int kernel_depth, int kernel_rows, int kernel_cols,
                  int depth_stride, int row_stride, int col_stride) {
    // Need to shuffle DHW -> WHD for eigen
    kernel_backward.device(d) = Eigen::CuboidConvolutionBackwardKernel(
        input, output_backward,
        kernel_cols, kernel_rows, kernel_depth,
        col_stride, row_stride, depth_stride);
  }
};


}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_3D_H_
