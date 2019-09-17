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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
__global__ void SparseTensorDenseMatMulKernel(
    int nnz, int m, int b_rows, int b_cols, int p,
    const Tindices* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, T* __restrict__ out) {
  // out_{ij} = sum_k {a_ik b_kj}
  // out = A * B', out_{ij} = sum_k {a_ik (b')_kj}; b'_{kj} = b_{jk}
  const int n = (ADJ_B) ? b_cols : b_rows;
  GPU_1D_KERNEL_LOOP(index, nnz * p) {
    const int a_ix = index / p;
    const int j = index % p;
    const int i = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 1 : 0));
    const int k = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 0 : 1));
    if (!FastBoundsCheck(i, m)) {
      continue;  // Nowhere to signal an error :(
    }
    // out[i, j]
    T* out_location = out + i * p + j;
    if (!FastBoundsCheck(k, n)) {
      GpuAtomicAdd(out_location, std::numeric_limits<T>::quiet_NaN());
      continue;
    }

    // a_value == (ADJ_A) ? a[k, i] : a[i, k]
    const T a_value = ldg(a_values + a_ix);

    // b_value == (ADJ_B) ? b[j, k] : b[k, j]
    const T b_value = ldg(b + ((ADJ_B) ? j * b_cols + k : k * b_cols + j));
    GpuAtomicAdd(out_location, a_value * b_value);
  }
}

namespace functor {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B> {
  static EIGEN_ALWAYS_INLINE Status
  Compute(const GPUDevice& d, typename TTypes<T>::Matrix out,
          typename TTypes<Tindices>::ConstMatrix a_indices,
          typename TTypes<T>::ConstVec a_values,
          typename TTypes<T>::ConstMatrix b) {
    out.device(d) = out.constant(T(0));
    int nnz = a_values.size();
    // out = A * B, A is [m x n] and B is [n x p], out is [m x p]
    int m = out.dimension(0);
    int p = out.dimension(1);
    int b_rows = b.dimension(0);
    int b_cols = b.dimension(1);

    // TODO(ebrevdo): Should this be alpha * nnz instead of
    // out.size()?  Perhaps p * nnz ?
    GpuLaunchConfig config = GetGpuLaunchConfig(p * nnz, d);

    TF_CHECK_OK(GpuLaunchKernel(
        SparseTensorDenseMatMulKernel<T, Tindices, ADJ_A, ADJ_B>,
        config.block_count, config.thread_per_block, 0, d.stream(), nnz, m,
        b_rows, b_cols, p, a_indices.data(), a_values.data(), b.data(),
        out.data()));

    return Status::OK();
  }
};

}  // namespace functor

#define DEFINE(T, Tindices)                                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, false>;               \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, true>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, false>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, true>;

DEFINE(float, int32);
DEFINE(float, int64);
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
