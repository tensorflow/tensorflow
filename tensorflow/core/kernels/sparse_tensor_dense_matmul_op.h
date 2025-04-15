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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename Tindices, bool ADJ_A,
          bool ADJ_B>
struct SparseTensorDenseMatMulFunctor {
  static EIGEN_ALWAYS_INLINE absl::Status Compute(
      OpKernelContext* ctx, typename TTypes<T>::Matrix out,
      typename TTypes<Tindices>::ConstMatrix a_indices,
      typename TTypes<T>::ConstVec a_values, typename TTypes<T>::ConstMatrix b);
};

template <typename MATRIX, bool ADJ>
class MaybeAdjoint;

template <typename MATRIX>
class MaybeAdjoint<MATRIX, false> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return m_(i, j);
  }

 private:
  const MATRIX m_;
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T MaybeConj(T v) {
  return Eigen::numext::conj(v);
}

template <typename MATRIX>
class MaybeAdjoint<MATRIX, true> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(MATRIX m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename MATRIX::Scalar operator()(
      const typename MATRIX::Index i, const typename MATRIX::Index j) const {
    return Eigen::numext::conj(m_(j, i));
  }

 private:
  const MATRIX m_;
};

template <typename T>
struct SumType {
  using type = T;
};

template <>
struct SumType<Eigen::half> {
  using type = float;  // Use fp32 accumulator for fp16 input values
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_
