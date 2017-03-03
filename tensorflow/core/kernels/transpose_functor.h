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

#ifndef TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
#define TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.dims() == out->dims()
// REQUIRES: in.dims() == perm.size()
// REQUIRES: in.dim_size(perm[i]) == out->dim_size(i)
template <typename Device>
Status DoTranspose(const Device& device, const Tensor& in,
                   const gtl::ArraySlice<int32> perm, Tensor* out);

// Implementation details.
namespace internal {

// Helper to compute 'strides' given a tensor 'shape'. I.e.,
// strides[i] = prod(shape.dim_size[(i+1):])
template <typename Index>
void ComputeStride(const TensorShape& shape, Index* strides) {
  const int ndims = shape.dims();
  Index stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<Index>(shape.dim_size(i));
  }
}

// Device-specific naive implementation for transpose.
template <typename Device, typename T>
void TransposeSimple(const Device& d, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out);

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, Tensor* out);

template <typename Device, typename T>
void Transpose(const Device& d, const Tensor& in,
               const gtl::ArraySlice<int32> perm, Tensor* out) {
  switch (in.dims()) {
    case 2:
      TransposeUsingEigen<Device, T, 2>(d, in, perm, out);
      break;
    case 3:
      TransposeUsingEigen<Device, T, 3>(d, in, perm, out);
      break;
    case 4:
      TransposeUsingEigen<Device, T, 4>(d, in, perm, out);
      break;
    default:
      TransposeSimple<Device, T>(d, in, perm, out);
      break;
  }
}
}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
