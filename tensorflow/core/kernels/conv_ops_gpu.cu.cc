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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/conv_2d.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
struct SpatialConvolution<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, const Eigen::PaddingType& padding) {
    SpatialConvolutionFunc(d, To32Bit(output), To32Bit(input), To32Bit(filter),
                           row_stride, col_stride, padding);
  }
};

template struct SpatialConvolution<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
