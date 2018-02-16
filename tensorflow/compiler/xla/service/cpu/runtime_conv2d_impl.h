/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_IMPL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_IMPL_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_spatial_convolutions.h"
#include "tensorflow/core/platform/types.h"

// 'tensorflow' namespace is used so that int64 and other types don't require
// qualification.
namespace tensorflow {
namespace xla {

template <typename EigenDevice, typename ScalarType>
void EigenConvImpl(const EigenDevice& device, ScalarType* out, ScalarType* lhs,
                   ScalarType* rhs, int64 input_batch, int64 input_rows,
                   int64 input_cols, int64 input_channels, int64 kernel_rows,
                   int64 kernel_cols, int64 kernel_channels,
                   int64 kernel_filters, int64 output_rows, int64 output_cols,
                   int64 row_stride, int64 col_stride, int64 padding_top,
                   int64 padding_bottom, int64 padding_left,
                   int64 padding_right, int64 lhs_row_dilation,
                   int64 lhs_col_dilation, int64 rhs_row_dilation,
                   int64 rhs_col_dilation) {
  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 4, Eigen::RowMajor>,
                         Eigen::Aligned>
      input(lhs, input_batch, input_rows, input_cols, input_channels);

  const Eigen::TensorMap<Eigen::Tensor<const ScalarType, 4, Eigen::RowMajor>,
                         Eigen::Aligned>
      kernel(rhs, kernel_rows, kernel_cols, kernel_channels, kernel_filters);

  Eigen::TensorMap<Eigen::Tensor<ScalarType, 4, Eigen::RowMajor>,
                   Eigen::Aligned>
      output(out, input_batch, output_rows, output_cols, kernel_filters);

  Eigen::array<Eigen::IndexPair<int64>, 1> contract_dims;
  contract_dims[0] = Eigen::IndexPair<int64>(1, 0);

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  //   kernels
  // - the second dimension (dims[1]): everything else
  Eigen::DSizes<int64, 2> pre_contract_dims;
  pre_contract_dims[0] = output_cols * output_rows * input_batch;
  pre_contract_dims[1] = kernel_channels * kernel_cols * kernel_rows;

  // Molds the output of the contraction into the shape expected by the user:
  Eigen::DSizes<int64, 4> post_contract_dims;
  post_contract_dims[0] = input_batch;
  post_contract_dims[1] = output_rows;
  post_contract_dims[2] = output_cols;
  post_contract_dims[3] = kernel_filters;

  Eigen::DSizes<int64, 2> kernel_dims;
  kernel_dims[0] = kernel_channels * kernel_cols * kernel_rows;
  kernel_dims[1] = kernel_filters;

  // The row and column dimensions must be flipped when passed to Eigen.
  output.device(device) =
      input
          .extract_image_patches(kernel_cols, kernel_rows, col_stride,
                                 row_stride, rhs_col_dilation, rhs_row_dilation,
                                 lhs_col_dilation, lhs_row_dilation,
                                 padding_left, padding_right, padding_top,
                                 padding_bottom, static_cast<ScalarType>(0.0f))
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims);
}

}  // namespace xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_RUNTIME_CONV2D_IMPL_H_
