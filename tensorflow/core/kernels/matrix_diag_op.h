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

#ifndef TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_
#define TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_

// Generator definition for MatrixDiagOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace functor {

// Reads the diagonal packing alignment.
void ReadAlignment(OpKernelConstruction* context,
                   bool* left_align_superdiagonal,
                   bool* left_align_subdiagonal);

// Calculates diagonal length and content offset (from aligning) of a diagonal.
// Returns a pair of integers {diag_len, content_offset}:
//   - diag_len: The length of the diag_index-th diagonal.
//   - content_offset: Each diagonal is stored as a row in the compact format.
//     If the diagonal is shorter than max_diag_len, its content is aligned
//     either to the left or right. content_offset is the index in the row
//     where the first element of the diag-index-th diagonal is stored. It is
//     always zero when the diagonal is left-aligned.
std::pair<int, int> ComputeDiagLenAndContentOffset(
    int diag_index, int max_diag_len, int num_rows, int num_cols,
    bool left_align_superdiagonal, bool left_align_subdiagonal);

template <typename Device, typename T>
struct MatrixDiagPart {
  EIGEN_ALWAYS_INLINE static void Compute(
      OpKernelContext* context, const Device& device,
      typename TTypes<T, 3>::ConstTensor& input,
      typename TTypes<T>::Tensor& output, const Eigen::Index lower_diag_index,
      const Eigen::Index upper_diag_index, const Eigen::Index max_diag_len,
      const T padding_value, const bool left_align_superdiagonal,
      const bool left_align_subdiagonal);
};

template <typename Device, typename T>
struct MatrixDiag {
  EIGEN_ALWAYS_INLINE static void Compute(
      OpKernelContext* context, const Device& device,
      typename TTypes<T>::ConstTensor& diag,
      typename TTypes<T, 3>::Tensor& output,
      const Eigen::Index lower_diag_index, const Eigen::Index upper_diag_index,
      const Eigen::Index max_diag_len, const T padding_value,
      const bool left_align_superdiagonal, const bool left_align_subdiagonal);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATRIX_DIAG_OP_H_
