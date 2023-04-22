/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_AVGPOOLING_OP_H_
#define TENSORFLOW_CORE_KERNELS_AVGPOOLING_OP_H_
// Functor definition for AvgPoolingOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SpatialAvgPooling {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input, int window_rows,
                  int window_cols, int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    if (Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
      // Use 32bit indexing to speed up the computations
      To32Bit(output).swap_layout().device(d) = Eigen::SpatialAvgPooling(
          To32Bit(input).swap_layout(), window_cols, window_rows, col_stride,
          row_stride, padding);
    } else {
      // Because we swap the layout, we swap the row/cols as well
      output.swap_layout().device(d) = Eigen::SpatialAvgPooling(
          input.swap_layout(), window_cols, window_rows, col_stride, row_stride,
          padding);
    }
  }
};

}  // namespace functor

typedef Eigen::GpuDevice GPUDevice;

// Launch a custom GPU kernels from Yanqing for the avgpooling backward
// operation that works NHWC data formats. Arguments:
//   top_diff: backprop to the output of the pooling layer
//   num: number of input batches
//   height: input height
//   width: input width
//   channels: number of input channels
//   pooled_height: the height of the output to the pooling layer
//   pooled_width: the width of the output to the pooling layer
//   kernel_h: the height of the pooling kernel
//   kernel_w: the width of the pooling kernel
//   stride_h: the height of the vertical stride
//   stride_w: the width of the horizontal stride
//   pad_t: padding size to the top side
//   pad_l: padding size to the left side
//   bottom_diff: backprop to the input of the pooling layer.
template <typename T>
bool RunAvePoolBackwardNHWC(const T* const top_diff, const int num,
                            const int height, const int width,
                            const int channels, const int pooled_height,
                            const int pooled_width, const int kernel_h,
                            const int kernel_w, const int stride_h,
                            const int stride_w, const int pad_t,
                            const int pad_l, T* const bottom_diff,
                            const GPUDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_AVGPOOLING_OP_H_
