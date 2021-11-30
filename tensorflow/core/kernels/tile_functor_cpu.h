/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_CPU_H_
#define TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_CPU_H_

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/tile_functor.h"

namespace tensorflow {
namespace internal {

template <typename Device, typename T>
void TileSimpleImpl(const Device& d, Tensor* out, const Tensor& in) {
  const int ndims = in.dims();
  const int64_t nelem = out->NumElements();
  gtl::InlinedVector<int64_t, 8> in_strides =
      ComputeStride<int64_t>(in.shape());
  gtl::InlinedVector<int64_t, 8> out_strides =
      ComputeStride<int64_t>(out->shape());
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();

  for (int64_t o_idx = 0; o_idx < nelem; ++o_idx) {
    int64_t i_idx = 0;
    int64_t t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += t / out_strides[i] % in.dim_size(i) * in_strides[i];
      t %= out_strides[i];
    }
    q[o_idx] = p[i_idx];
  }
}

template <typename T>
void TileSimple(const Eigen::ThreadPoolDevice& d, Tensor* out,
                const Tensor& in) {
  return TileSimpleImpl<Eigen::ThreadPoolDevice, T>(d, out, in);
}

}  // namespace internal
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TILE_FUNCTOR_CPU_H_
