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

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda
#endif

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_TO_DENSE_GPU_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_TO_DENSE_GPU_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

namespace functor {
template <typename T, typename Index>
struct LaunchSparseToDense {
  void operator()(OpKernelContext* c, bool validate_indices,
                  const Index* indices, const T* values, const int num_elems,
                  const int num_values, const Index* shape, const int num_dims,
                  const T default_value, Index dense_size, T* dense);
};

}  // namespace functor

}  // namespace tensorflow
#endif // TENSORFLOW_CORE_KERNELS_SPARSE_TO_DENSE_GPU_H_
