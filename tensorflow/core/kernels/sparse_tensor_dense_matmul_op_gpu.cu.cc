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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace generator {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B, int NDIM>
class SparseTensorDenseMatMulGPUGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SparseTensorDenseMatMulGPUGenerator(
      typename TTypes<T, NDIM>::Tensor32Bit out,
      typename TTypes<const Tindices, 2>::Tensor32Bit a_indices,
      typename TTypes<const T, 1>::Tensor32Bit a_values,
      typename TTypes<const T, NDIM>::Tensor32Bit b)
      : out_(out),
        lhs_index_a_(ADJ_A ? NDIM-1 : NDIM-2),
        rhs_index_a_(ADJ_A ? NDIM-2 : NDIM-1),
        a_indices_(a_indices),
        a_values_(a_values),
        lhs_right_size(ADJ_B ? b.dimension(NDIM-1) : b.dimension(NDIM-2)),
        maybe_adjoint_b_(
            functor::MaybeAdjoint<typename TTypes<const T, NDIM>::Tensor32Bit,
                                  ADJ_B, NDIM>(b)) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE T
  operator()(const Eigen::array<int, 2>& j_and_ix) const {
  
#ifdef __CUDA_ARCH__
    const int j = j_and_ix[0];
    const int ix = j_and_ix[1];
    int m = a_indices_(ix, lhs_index_a_);
    int k = a_indices_(ix, rhs_index_a_);
    assert(k < lhs_right_size);
    assert(m < out_.dimension(NDIM-2));
    // If asserts are disabled, the caller is violating the sparse
    // tensor index contract, and so we return invalid results.
    // Force returning NaNs to try to signal that something is amiss.
    Eigen::array<int, NDIM> indices;
      for (int k = 0; k < NDIM - 2; ++k) {
        indices[k] = a_indices_(ix, k);
      }
    T b_value;
    if (k >= lhs_right_size || m >= out_.dimension(NDIM-2)) {
      m = 0;
      k = 0;
      b_value = std::numeric_limits<T>::quiet_NaN();
    } else {
      indices[NDIM - 2] = k;
      indices[NDIM - 1] = j;
      b_value = maybe_adjoint_b_(indices);
    }
    indices[NDIM - 2] = m;
    indices[NDIM - 1] = j;
    atomicAdd(&out_(indices), a_values_(ix) * b_value);
#else
    assert(false && "This should only be run on the device");
#endif
    // Return something
    return T(0);
  }

 private:
  mutable typename TTypes<T, NDIM>::Tensor32Bit out_;
  const int lhs_index_a_;
  const int rhs_index_a_;
  typename TTypes<const Tindices, 2>::Tensor32Bit a_indices_;
  typename TTypes<const T, 1>::Tensor32Bit a_values_;
  const int lhs_right_size;
  functor::MaybeAdjoint<typename TTypes<const T, NDIM>::Tensor32Bit, ADJ_B, NDIM>
      maybe_adjoint_b_;
};

}  // namespace generator

namespace functor {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B, int NDIM>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B, NDIM> {
  static EIGEN_ALWAYS_INLINE Status Compute(
      const GPUDevice& d,
      typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<Tindices>::ConstMatrix a_indices,
      typename TTypes<T>::ConstVec a_values,
      typename TTypes<T, NDIM>::ConstTensor b,
      typename TTypes<T>::Vec scratch) {
    generator::SparseTensorDenseMatMulGPUGenerator<T, Tindices, ADJ_A, ADJ_B, NDIM>
        sparse_tensor_dense_matmul_generator(To32Bit(out), To32Bit(a_indices),
                                             To32Bit(a_values), To32Bit(b));
    // The following line gives Assertion Failure in both current and older version.
    To32Bit(out).device(d) = To32Bit(out).constant(T(0));
    int nnz = a_values.size();
    int n = (ADJ_B) ? b.dimension(NDIM-2) : b.dimension(NDIM-1);

#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::Tensor<int, 2>::Dimensions matrix_1_by_nnz{{ 1, nnz }};
    Eigen::array<int, 2> n_by_1{{ n, 1 }};
    Eigen::array<int, 1> reduce_on_rows{{ 0 }};
#else
    Eigen::IndexList<Eigen::type2index<1>, int> matrix_1_by_nnz;
    matrix_1_by_nnz.set(1, nnz);
    Eigen::IndexList<int, Eigen::type2index<1> > n_by_1;
    n_by_1.set(0, n);
    Eigen::IndexList<Eigen::type2index<0> > reduce_on_rows;
#endif

    // How this works: the generator iterates over (j, ix) where j
    // iterates from 0 .. n - 1 and ix iterates from
    // 0 .. nnz - 1.  A side effect of the generator is to accumulate
    // the products of values in A and B into the appropriate location
    // in the dense matrix out.  In order to run the iteration,
    // we take a smaller variable and broadcast to a size (n, nnz).
    // This is the scratch variable.  In order to enforce execution,
    // we have to perform assignment back into scratch (taking the sum).
    // We don't care what gets assigned to scratch - only the side effect
    // of the execution in the generator.
    //
    // Note it's not sufficient that scratch be a scalar, and to
    // broadcast it to a matrix.  Eigen splits the computation not
    // based on the largest intermediate shape (the size of the
    // broadcast of scratch) but based on the output shape.  So
    // scratch needs to be a vector at least.
    //
    // Note also that only float type is supported because the
    // atomicAdd operation is only supported for floats in hardware.
    To32Bit(scratch).device(d) =
        To32Bit(scratch)
            .reshape(matrix_1_by_nnz)
            .broadcast(n_by_1)
            .generate(sparse_tensor_dense_matmul_generator)
            .sum(reduce_on_rows);

    return Status::OK();
  }
};

}  // namespace functor

#define DEFINE(T, Tindices, NDIM)                                              \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, false, false, NDIM>;                             \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, false, true, NDIM>;                              \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, true, false, NDIM>;                              \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, true, true, NDIM>;

#define HANDLE_DIM(NDIM)       \
  DEFINE(float, int32, NDIM);  \
  DEFINE(float, int64, NDIM);
  
HANDLE_DIM(2);
HANDLE_DIM(3);
HANDLE_DIM(4);
HANDLE_DIM(5);

#undef HANDLE_DIM
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
