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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TRANSPOSE_OP_FUNCTOR_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TRANSPOSE_OP_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
void Transpose(const Device& d, typename TTypes<T, NDIMS>::Tensor out,
               typename TTypes<T, NDIMS>::ConstTensor in, const int* perm) {
  // perm[] is a permutation of 0, 1, ..., NDIMS-1. perm[] is on CPU.
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  out.device(d) = in.shuffle(p);
}

template <typename Device, typename T, int NDIMS>
struct TransposeFunctor {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor out,
                  typename TTypes<T, NDIMS>::ConstTensor in, const int* perm);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_TRANSPOSE_OP_FUNCTOR_H_
