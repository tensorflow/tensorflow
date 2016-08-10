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

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_RANDOM_OP_H_
