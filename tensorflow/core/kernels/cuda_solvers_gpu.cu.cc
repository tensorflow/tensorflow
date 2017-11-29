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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/cuda_solvers.h"

#include <complex>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// TODO(rmlarsen): Add a faster custom kernel similar to
// SwapDimension1And2InTensor3 in tensorflow/core/kernels/conv_ops_gpu_3.cu.cc
template <typename Scalar>
struct AdjointBatchFunctor<GPUDevice, Scalar> {
  void operator()(const GPUDevice& d,
                  typename TTypes<Scalar, 3>::ConstTensor input,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const Eigen::array<int, 3> perm({0, 2, 1});
    To32Bit(output).device(d) = To32Bit(input).shuffle(perm).conjugate();
  }
};

// Instantiate implementations for the 4 numeric types.
template struct AdjointBatchFunctor<GPUDevice, float>;
template struct AdjointBatchFunctor<GPUDevice, double>;
template struct AdjointBatchFunctor<GPUDevice, std::complex<float>>;
template struct AdjointBatchFunctor<GPUDevice, std::complex<double>>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
