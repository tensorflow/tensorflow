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

#ifndef TENSORFLOW_KERNELS_RANDOM_OP_H_
#define TENSORFLOW_KERNELS_RANDOM_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {

class OpKernelContext;

namespace functor {

template <typename Device, class Distribution>
struct FillPhiloxRandom;

typedef Eigen::ThreadPoolDevice CPUDevice;
// Declares the partially CPU-specialized functor struct.
//
// NOTE: Due to inlining done by the compiler, you may need to add
// explicit instantiation of the functor in random_op.cc.  See example
// functor::FillPhiloxRandom<CPUDevice, random::UniformDistribution>.
template <class Distribution>
struct FillPhiloxRandom<CPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const CPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist);
};

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
// Declares the partially GPU-specialized functor struct.
template <class Distribution>
struct FillPhiloxRandom<GPUDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist);
};
#endif  // GOOGLE_CUDA

#if TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
// Declares the partially SYCL-specialized functor struct.
template <class Distribution>
struct FillPhiloxRandom<SYCLDevice, Distribution> {
  void operator()(OpKernelContext* ctx, const SYCLDevice& d,
                  random::PhiloxRandom gen,
                  typename Distribution::ResultElementType* data, int64 size,
                  Distribution dist);
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_RANDOM_OP_H_
