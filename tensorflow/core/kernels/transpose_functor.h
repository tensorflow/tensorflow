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

typedef gtl::InlinedVector<int64, 8> TransposeDimsVec;
typedef gtl::InlinedVector<int32, 8> TransposePermsVec;

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

// Helper function that takes a tensor shape, a permutation, combines the
// neighboring shapes if their indices in the permutation are consecutive.
// The function outputs the combined shape and new permutation.
// Example: Tensor shape {2, 3, 4, 5, 120} and permutation {0, 4, 1, 2, 3} will
// produce new shape {2, 60, 120} and new permutation {0, 2, 1}.
inline void ReduceTransposeDimensions(const TensorShape& shape,
                                      gtl::ArraySlice<int32> perm,
                                      TransposePermsVec* new_perm,
                                      TransposeDimsVec* new_dims) {
  CHECK_EQ(shape.dims(), perm.size());
  if (shape.dims() == 1) {
    // If input dimension is already 1, no need to reduce dimension.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    (*new_dims)[0] = shape.dim_size(0);
    return;
  }
  TransposePermsVec new_dim_position(shape.dims(), -1);
  TransposeDimsVec combined_dims(shape.dims(), 0);
  int cur_head = perm[0];
  new_dim_position[cur_head] = 0;
  combined_dims[0] = shape.dim_size(cur_head);
  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < shape.dims(); ++perm_idx) {
    // If two indices in permutation are consecutive numbers, combine their
    // dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape.dim_size(cur_head);
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_position[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape.dim_size(cur_head);
    }
  }
  // Compact the new permutations and dimension sizes.
  new_perm->resize(dim_idx + 1);
  new_dims->resize(dim_idx + 1);
  dim_idx = 0;
  for (int i = 0; i < new_dim_position.size(); ++i) {
    if (new_dim_position[i] >= 0) {
      int new_perm_idx = new_dim_position[i];
      (*new_perm)[dim_idx] = new_perm_idx;
      (*new_dims)[dim_idx] = combined_dims[new_perm_idx];
      dim_idx++;
    }
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

}  // namespace internal

template <typename Device, typename T>
struct Transpose {
  static void run(const Device& d, const Tensor& in,
                  const gtl::ArraySlice<int32> perm, Tensor* out);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TRANSPOSE_FUNCTOR_H_
