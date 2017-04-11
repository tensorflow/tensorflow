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

#ifndef TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
#define TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
// Functor definition for MaxPoolingOp, must be compilable by nvcc.

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_pooling.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SpatialMaxPooling {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input, int window_rows,
                  int window_cols, int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    // Because we swap the layout, we swap the row/cols as well
    output.swap_layout().device(d) =
        Eigen::SpatialMaxPooling(input.swap_layout(), window_cols, window_rows,
                                 col_stride, row_stride, padding);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
