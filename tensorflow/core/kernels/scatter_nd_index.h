/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_ND_INDEX_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_ND_INDEX_H_

#include <cstdint>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/overflow.h"

namespace tensorflow {
namespace scatter_nd_op {

// Computes prefix strides in int64. Returns false if a product overflows.
// Do not store strides in the index tensor dtype: int32 indices into a
// valid large shape would wrap and chip out of bounds.
template <int IXDIM>
bool ComputeScatterNdBatchStrides(
    const Eigen::array<Eigen::DenseIndex, IXDIM>& output_shape_prefix,
    Eigen::array<int64_t, IXDIM>* batch_strides) {
  if (IXDIM > 0) {
    (*batch_strides)[IXDIM - 1] = 1;
  }
  for (int dim = IXDIM - 2; dim >= 0; --dim) {
    const int64_t prod = MultiplyWithoutOverflow(
        (*batch_strides)[dim + 1],
        static_cast<int64_t>(output_shape_prefix[dim + 1]));
    if (TF_PREDICT_FALSE(prod < 0)) {
      return false;
    }
    (*batch_strides)[dim] = prod;
  }
  return true;
}

// Flattens `coords` using int64 strides. Returns false if any coordinate is
// out of range, arithmetic overflows, or the flat index is not in
// [0, num_slices).
template <typename Index, int IXDIM>
bool ComputeScatterNdFlatIndex(
    const Eigen::array<Eigen::DenseIndex, IXDIM>& output_shape_prefix,
    const Eigen::array<int64_t, IXDIM>& batch_strides,
    const Eigen::array<Index, IXDIM>& coords, int64_t num_slices,
    int64_t* flat_index) {
  int64_t i = 0;
  for (int dim = 0; dim < IXDIM; ++dim) {
    const Index ix_d = coords[dim];
    if (TF_PREDICT_FALSE(!FastBoundsCheck(ix_d, output_shape_prefix[dim]))) {
      return false;
    }
    const int64_t contrib = MultiplyWithoutOverflow(
        static_cast<int64_t>(ix_d), batch_strides[dim]);
    if (TF_PREDICT_FALSE(contrib < 0)) {
      return false;
    }
    i = AddWithoutOverflow(i, contrib);
    if (TF_PREDICT_FALSE(i < 0)) {
      return false;
    }
  }
  if (TF_PREDICT_FALSE(!FastBoundsCheck(i, num_slices))) {
    return false;
  }
  *flat_index = i;
  return true;
}

}  // namespace scatter_nd_op
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_ND_INDEX_H_
