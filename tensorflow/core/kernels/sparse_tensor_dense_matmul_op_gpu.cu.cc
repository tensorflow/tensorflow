/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct OutOfBoundsValue {
  __host__ __device__ static T value() {
    return Eigen::NumTraits<T>::quiet_NaN();
  }
};

template <typename T>
struct OutOfBoundsValue<std::complex<T>> {
  __host__ __device__ static std::complex<T> value() {
    return std::complex<T>(OutOfBoundsValue<T>::value(),
                           OutOfBoundsValue<T>::value());
  }
};

template <typename Tsrc, typename Tdst>
__global__ void CastCopy(
    const int size, const Tsrc* __restrict__  src, Tdst* __restrict__ dst) {

  GPU_1D_KERNEL_LOOP(index, size) {
    dst[index] = static_cast<Tdst>(src[index]);
  }
}

template<>
__global__ void CastCopy(
    const int size, const complex128* __restrict__  src,
    complex64* __restrict__ dst) {

  GPU_1D_KERNEL_LOOP(index, size) {
    dst[index] = complex64(src[index].real(), (src[index].imag()));
  }
}

template <typename Tout, typename Tin>
__device__ void GpuCastAtomicAdd(Tout* out_location, Tin val) {
  GpuAtomicAdd(out_location, static_cast<Tout>(val));
}

#ifndef TENSORFLOW_USE_ROCM
template<>
__device__ void GpuCastAtomicAdd(complex128* out_location, complex64 val) {
  GpuAtomicAdd(out_location, complex128(val.real(), val.imag()));
}
#endif

template <typename T, typename Tsum, typename Tindices, bool ADJ_A, bool ADJ_B>
__global__ void SparseTensorDenseMatMulKernel(
    int nnz, int m, int b_rows, int b_cols, int p,
    const Tindices* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, Tsum* __restrict__ out) {
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
    Tsum* out_location = out + i * p + j;
    if (!FastBoundsCheck(k, n)) {
      GpuAtomicAdd(out_location, OutOfBoundsValue<Tsum>::value());
      continue;
    }

    // a_value == (ADJ_A) ? conj(a[k, i]) : a[i, k]
    const T a_input = ldg(a_values + a_ix);
    const T a_value = ADJ_A ? Eigen::numext::conj(a_input) : a_input;

    // b_value == (ADJ_B) ? conj(b[j, k]) : b[k, j]
    const T b_input = ldg(b + ((ADJ_B) ? j * b_cols + k : k * b_cols + j));
    const T b_value = ADJ_B ? Eigen::numext::conj(b_input) : b_input;

    GpuCastAtomicAdd<Tsum, T>(out_location, a_value * b_value);
  }
}

namespace functor {

namespace {

bool RequireDeterminism() {
  static bool require_determinism = [] {
    bool deterministic_ops = false;
    TF_CHECK_OK(tensorflow::ReadBoolFromEnvVar("TF_DETERMINISTIC_OPS",
                                               /*default_val=*/false,
                                               &deterministic_ops));
    return deterministic_ops;
  }();
  return require_determinism;
}

} // namespace

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B> {

  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* ctx, typename TTypes<T>::Matrix out,
          typename TTypes<Tindices>::ConstMatrix a_indices,
          typename TTypes<T>::ConstVec a_values,
          typename TTypes<T>::ConstMatrix b) {

    if (RequireDeterminism()) {
      if (std::is_same<T, double>::value ||
          std::is_same<T, complex128>::value) {
        return errors::Unimplemented(
            "No deterministic GPU implementation of sparse_dense_matmul "
            "available for data of type tf.float64 or tf.complex128");
      }
      using Tsum = typename SumType<T>::type_for_determinism;
      ComputeWithPossiblyHigherPrecisionAccumulator<Tsum>(
          ctx, out, a_indices, a_values, b);
    } else {
      using Tsum = typename SumType<T>::type;
      ComputeWithPossiblyHigherPrecisionAccumulator<Tsum>(
          ctx, out, a_indices, a_values, b);
    }
    return Status::OK();
  }

  private:
    template <typename Tsum>
    static EIGEN_ALWAYS_INLINE Status
    ComputeWithPossiblyHigherPrecisionAccumulator(
        OpKernelContext* ctx, typename TTypes<T>::Matrix &out,
        typename TTypes<Tindices>::ConstMatrix a_indices,
        typename TTypes<T>::ConstVec a_values,
        typename TTypes<T>::ConstMatrix b) {

      int nnz = a_values.size();
      // out = A * B, A is [m x n] and B is [n x p], out is [m x p]
      int m = out.dimension(0);
      int p = out.dimension(1);
      int b_rows = b.dimension(0);
      int b_cols = b.dimension(1);
      const GPUDevice& d = ctx->eigen_device<GPUDevice>();
      GpuLaunchConfig config = GetGpuLaunchConfig(p * nnz, d);

      Tensor temp_out_t;
      Tsum* maybe_temp_out_data = nullptr;
      bool sum_type_is_different = !std::is_same<T, Tsum>::value;

      if (sum_type_is_different) {
        TF_RETURN_IF_ERROR(ctx->allocate_temp(
            DataTypeToEnum<Tsum>::value,
            TensorShape({out.dimension(0), out.dimension(1)}), &temp_out_t));

        auto temp_out = temp_out_t.matrix<Tsum>();
        maybe_temp_out_data = temp_out.data();
      } else {
        // Note: The reinterpret cast is only required to avoid a compilation
        // error; it is only used if Tsum == T.
        maybe_temp_out_data = reinterpret_cast<Tsum*>(out.data());
      }

      TF_CHECK_OK(GpuLaunchKernel(
          SetZero<Tsum>, config.block_count, config.thread_per_block, 0,
          d.stream(), m * p, maybe_temp_out_data));

      TF_CHECK_OK(GpuLaunchKernel(
          SparseTensorDenseMatMulKernel<T, Tsum, Tindices, ADJ_A, ADJ_B>,
          config.block_count, config.thread_per_block, 0, d.stream(), nnz, m,
          b_rows, b_cols, p, a_indices.data(), a_values.data(), b.data(),
          maybe_temp_out_data));

      if (sum_type_is_different) {
        TF_CHECK_OK(GpuLaunchKernel(
            CastCopy<Tsum, T>, config.block_count, config.thread_per_block,
            0, d.stream(), m * p, maybe_temp_out_data,
            out.data()));
      }
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

#define DEFINE_ALL_INDEX_TYPES(T) \
  DEFINE(T, int32);               \
  DEFINE(T, int64)

DEFINE_ALL_INDEX_TYPES(Eigen::half);
DEFINE_ALL_INDEX_TYPES(float);
DEFINE_ALL_INDEX_TYPES(double);

// ROCm's GpuAtomicAdd doesn't support std::complex yet.
#ifndef TENSORFLOW_USE_ROCM
DEFINE_ALL_INDEX_TYPES(complex64);
DEFINE_ALL_INDEX_TYPES(complex128);
#endif

#undef DEFINE_ALL_INDEX_TYPES
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
