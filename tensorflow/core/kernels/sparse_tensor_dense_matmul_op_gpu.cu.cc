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
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B, int NDIM>
__global__ void SparseTensorDenseMatMulKernel(int nnz, int m, int b_rows,
                                              int b_cols, int p,
                                              const Tindices* a_indices,
                                              const T* a_values, const T* b,
                                              const int* skips, T* out) {
  // out_{ij} = sum_k {a_ik b_kj}
  // out = A * B', out_{ij} = sum_k {a_ik (b')_kj}; b'_{kj} = b_{jk}
  const int n = (ADJ_B) ? b_cols : b_rows;
  CUDA_1D_KERNEL_LOOP(index, nnz * p) {
    const int a_ix = index / p;
    const int j = index % p;
    const Tindices* a_index = a_indices + NDIM * a_ix;
    const int i = ldg(a_index + ((ADJ_A) ? NDIM - 1 : NDIM - 2));
    const int k = ldg(a_index + ((ADJ_A) ? NDIM - 2 : NDIM - 1));
    // matrices to skip
    int skip = 0;
    for (int ind = 0; ind < NDIM - 2; ++ind) {
      skip += ldg(a_index + ind) * skips[ind];
    }
    if (!FastBoundsCheck(i, m)) {
      continue;  // Nowhere to signal an error :(
    }
    // out[..., i, j]
    T* out_location = out + skip * m * p + i * p + j;
    if (!FastBoundsCheck(k, n)) {
      CudaAtomicAdd(out_location, std::numeric_limits<T>::quiet_NaN());
      continue;
    }

    // a_value == (ADJ_A) ? a[..., k, i] : a[..., i, k]
    const T a_value = ldg(a_values + a_ix);

    // b_value == (ADJ_B) ? b[..., j, k] : b[..., k, j]
    const T b_value = ldg(b
                          + skip * b_rows * b_cols
                          + ((ADJ_B) ? j * b_cols + k : k * b_cols + j));
    CudaAtomicAdd(out_location, a_value * b_value);
  }
}

namespace functor {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B, int NDIM>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B, NDIM> {
  static EIGEN_ALWAYS_INLINE Status Compute(
      const GPUDevice& d,
      typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<Tindices>::ConstMatrix a_indices,
      typename TTypes<T>::ConstVec a_values,
      typename TTypes<T, NDIM>::ConstTensor b) {
    out.device(d) = out.constant(T(0));
    int nnz = a_values.size();
    // out = A * B, A is [... x m x n] and B is [... x n x p], out is [... x m x p]
    int m = out.dimension(NDIM-2);
    int p = out.dimension(NDIM-1);
    int b_rows = b.dimension(NDIM-2);
    int b_cols = b.dimension(NDIM-1);
    
    // stores matrices to skip:
    // e.g.: if out.dims = {6, 5, 3, 4}, then skips = {5, 1, 1}
    //       if a_index = {4, 3, 2, 1}, then matrices to skip = 4*5 + 3*1
    int* skips = new int[NDIM-1];
    skips[NDIM-2] = 1;
    if (NDIM != 2) {
      skips[NDIM-3] = 1;
    }
    for (int i = NDIM - 4; i >= 0; --i) {
      skips[i] = skips[i+1] * out.dimension(i+1);
    }

    // TODO(ebrevdo): Should this be alpha * nnz instead of
    // out.size()?  Perhaps p * nnz ?
    CudaLaunchConfig config = GetCudaLaunchConfig(p * nnz, d);

    SparseTensorDenseMatMulKernel<T, Tindices, ADJ_A, ADJ_B, NDIM>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            nnz, m, b_rows, b_cols, p, a_indices.data(), a_values.data(),
            b.data(), skips, out.data());
            
    delete [] skips;

    return Status::OK();
  }
};

}  // namespace functor

#define INSTANTIATE_FUNCTORS(T, Tindices, NDIM)                                \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, false, false, NDIM>;                             \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, false, true, NDIM>;                              \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, true, false, NDIM>;                              \
  template struct functor::SparseTensorDenseMatMulFunctor<                     \
      GPUDevice, T, Tindices, true, true, NDIM>;

#define HANDLE_DIM(NDIM)                    \
  INSTANTIATE_FUNCTORS(float, int32, NDIM); \
  INSTANTIATE_FUNCTORS(float, int64, NDIM);
  
HANDLE_DIM(2);
HANDLE_DIM(3);
HANDLE_DIM(4);
HANDLE_DIM(5);

#undef HANDLE_DIM
#undef INSTANTIATE_FUNCTORS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
