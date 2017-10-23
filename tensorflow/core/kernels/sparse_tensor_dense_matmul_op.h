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

#ifndef TENSORFLOW_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_
#define TENSORFLOW_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T, typename Tindices, bool ADJ_A,   \
          bool ADJ_B, int NDIM>
struct SparseTensorDenseMatMulFunctor {
  static EIGEN_ALWAYS_INLINE Status Compute(
      const Device& d,
      typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<Tindices>::ConstMatrix a_indices,
      typename TTypes<T>::ConstVec a_values,
      typename TTypes<T, NDIM>::ConstTensor b);
};

template <typename TENSOR, bool ADJ, int NDIM>
class MaybeAdjoint;

template <typename TENSOR, int NDIM>
class MaybeAdjoint<TENSOR, false, NDIM> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(TENSOR m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename TENSOR::Scalar operator()(
      const typename TENSOR::Index i, const typename TENSOR::Index j) const {
    return m_(i, j);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename TENSOR::Scalar operator()(
      const Eigen::array<Eigen::Index, NDIM> indices) const {
    return m_(indices);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename TENSOR::Scalar operator()(
      const Eigen::array<int, NDIM> indices) const {
    return m_(indices);
  }

 private:
  const TENSOR m_;
};

template <typename T>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T MaybeConj(T v) {
  return v;
}

template <typename TENSOR, int NDIM>
class MaybeAdjoint<TENSOR, true, NDIM> {
 public:
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE MaybeAdjoint(TENSOR m) : m_(m) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE typename TENSOR::Scalar operator()(
      const typename TENSOR::Index i, const typename TENSOR::Index j) const {
    return Eigen::numext::conj(m_(j, i));
  }
  typename TENSOR::Scalar operator()(
      const Eigen::array<Eigen::Index, NDIM> indices) const {
    Eigen::array<Eigen::Index, NDIM> swapped(indices);
    Eigen::Index tmp = swapped[NDIM - 2];
    swapped[NDIM - 2] = swapped[NDIM - 1];
    swapped[NDIM - 1] = tmp;
    return Eigen::numext::conj(m_(swapped));
  }
  typename TENSOR::Scalar operator()(
      const Eigen::array<int, NDIM> indices) const {
    Eigen::array<int, NDIM> swapped(indices);
    Eigen::Index tmp = swapped[NDIM - 2];
    swapped[NDIM - 2] = swapped[NDIM - 1];
    swapped[NDIM - 1] = tmp;
    return Eigen::numext::conj(m_(swapped));
  }

 private:
  const TENSOR m_;
};

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SPARSE_TENSOR_DENSE_MATMUL_OP_H_
