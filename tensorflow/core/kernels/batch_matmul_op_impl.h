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

// See docs in ../ops/math_ops.cc.

#ifndef TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_

#define EIGEN_USE_THREADS

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace {

// Returns the pair of dimensions along which to perform Tensor contraction to
// emulate matrix multiplication.
// For matrix multiplication of 2D Tensors X and Y, X is contracted along
// second dimension and Y is contracted along the first dimension (if neither X
// nor Y is adjointed). The dimension to contract along is switched when any
// operand is adjointed.
// See http://en.wikipedia.org/wiki/Tensor_contraction
Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  return Eigen::IndexPair<Eigen::DenseIndex>(adj_x ? 0 : 1, adj_y ? 1 : 0);
}

// Parallel batch matmul kernel based on the multi-threaded tensor contraction
// in Eigen.
template <typename Scalar, bool IsComplex = true>
struct ParallelMatMulKernel {
  static void Conjugate(const OpKernelContext* context, Tensor* out) {
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    auto z = out->tensor<Scalar, 3>();
    z.device(d) = z.conjugate();
  }

  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor in_y, bool adj_x, bool adj_y,
                  const MatMulBCast& bcast, Tensor* out, int start, int limit) {
    static_assert(IsComplex, "Complex type expected.");
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    // We use the identities
    //   conj(a) * conj(b) = conj(a * b)
    //   conj(a) * b = conj(a * conj(b))
    // to halve the number of cases. The final conjugation of the result is
    // done at the end of LaunchBatchMatMul<CPUDevice, Scalar>::Launch().
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;

      auto x = Tx.template chip<0>(x_batch_index);
      auto z = Tz.template chip<0>(i);
      if (adj_x != adj_y) {
        auto y = Ty.template chip<0>(y_batch_index).conjugate();
        z.device(d) = x.contract(y, contract_pairs);
      } else {
        auto y = Ty.template chip<0>(y_batch_index);
        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

// The Eigen contraction kernel used here is very large and slow to compile,
// so we partially specialize ParallelMatMulKernel for real types to avoid all
// but one of the instantiations.
template <typename Scalar>
struct ParallelMatMulKernel<Scalar, false> {
  static void Conjugate(const OpKernelContext* context, Tensor* out) {}

  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y,
                  const MatMulBCast& bcast, Tensor* out, int start, int limit) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto x = Tx.template chip<0>(x_batch_index);
      auto y = Ty.template chip<0>(y_batch_index);
      auto z = Tz.template chip<0>(i);

      z.device(d) = x.contract(y, contract_pairs);
    }
  }
};

// Sequential batch matmul kernel that calls the regular Eigen matmul.
// We prefer this over the tensor contraction because it performs
// better on vector-matrix and matrix-vector products.
template <typename Scalar>
struct SequentialMatMulKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
    return MatrixMap(
        t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const Tensor& in_x, const Tensor& in_y, bool adj_x,
                  bool adj_y, const MatMulBCast& bcast, Tensor* out, int start,
                  int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto x = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto y = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto z = TensorSliceToEigenMatrix(out, i);
      if (!adj_x) {
        if (!adj_y) {
          z.noalias() = x * y;
        } else {
          z.noalias() = x * y.adjoint();
        }
      } else {
        if (!adj_y) {
          z.noalias() = x.adjoint() * y;
        } else {
          z.noalias() = x.adjoint() * y.adjoint();
        }
      }
    }
  }
};

}  // namespace

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

template <typename Scalar>
struct LaunchBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y,
                     const MatMulBCast& bcast, Tensor* out) {
    typedef ParallelMatMulKernel<Scalar, Eigen::NumTraits<Scalar>::IsComplex>
        ParallelMatMulKernel;
    bool conjugate_result = false;

    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = bcast.output_batch_size();
    const int64 cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    const int64 small_dim = std::min(
        std::min(in_x.dim_size(1), in_x.dim_size(2)), out->dim_size(2));
    // NOTE(nikhilsarda): This heuristic is optimal in benchmarks as of
    // Jan 21, 2020.
    const int64 kMaxCostOuterParallelism = 128 * 128;  // heuristic.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    if (small_dim > 1 &&
        (batch_size == 1 || cost_per_unit > kMaxCostOuterParallelism)) {
      // Parallelize over inner dims.
      // For large matrix products it is counter-productive to parallelize
      // over the batch dimension.
      ParallelMatMulKernel::Run(context, in_x, in_y, adj_x, adj_y, bcast, out,
                                0, batch_size);
      conjugate_result = adj_x;
    } else {
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, adj_x, adj_y, &bcast, out](int start, int limit) {
              SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y,
                                                  bcast, out, start, limit);
            });
    }
    if (conjugate_result) {
      // We used one of the identities
      //   conj(a) * conj(b) = conj(a * b)
      //   conj(a) * b = conj(a * conj(b))
      // above, we need to conjugate the final output. This is a
      // no-op for non-complex types.
      ParallelMatMulKernel::Conjugate(context, out);
    }
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(gpu_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

class BlasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  BlasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes() override { return -1; }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};
}  // namespace

template <typename Scalar>
struct LaunchBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y,
                     const MatMulBCast& bcast, Tensor* out) {
    constexpr se::blas::Transpose kTranspose =
        is_complex<Scalar>::value ? se::blas::Transpose::kConjugateTranspose
                                  : se::blas::Transpose::kTranspose;
    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const int64 batch_size = bcast.output_batch_size();
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef se::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(bcast.x_batch_size());
    b_device_memory.reserve(bcast.y_batch_size());
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    uint64 a_stride;
    uint64 b_stride;
    uint64 c_stride;

    bool is_full_broadcast =
        std::min(bcast.x_batch_size(), bcast.y_batch_size()) == 1;
    bool use_strided_batched =
        (!bcast.IsBroadcastingRequired() || is_full_broadcast) &&
        batch_size > 1;
    if (use_strided_batched) {
      a_stride = bcast.x_batch_size() != 1 ? m * k : 0;
      b_stride = bcast.y_batch_size() != 1 ? k * n : 0;
      c_stride = m * n;
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    } else if (!bcast.IsBroadcastingRequired()) {
      for (int64 i = 0; i < batch_size; ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory.back());
        b_ptrs.push_back(&b_device_memory.back());
        c_ptrs.push_back(&c_device_memory.back());
      }
    } else {
      const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
      const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
      for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      }
      for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      }
      for (int64 i = 0; i < batch_size; ++i) {
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
        b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
        c_ptrs.push_back(&c_device_memory.back());
      }
    }

    typedef Scalar Coefficient;

    // Blas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      if (n == 1 &&
          blas_transpose_b != se::blas::Transpose::kConjugateTranspose &&
          blas_transpose_a != se::blas::Transpose::kConjugateTranspose) {
        // This is a matrix*vector multiply so use GEMV to compute A * b.
        // Here we are multiplying in the natural order, so we have to flip
        // the transposition flag to compensate for the tensor being stored
        // row-major. Since GEMV doesn't provide a way to just conjugate an
        // argument, we have to defer those cases to GEMM below.
        auto gemv_trans_a = blas_transpose_a == se::blas::Transpose::kTranspose
                                ? se::blas::Transpose::kNoTranspose
                                : se::blas::Transpose::kTranspose;
        bool blas_launch_status =
            stream
                ->ThenBlasGemv(gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
                               static_cast<Coefficient>(1.0), *(a_ptrs[0]),
                               adj_x ? m : k, *(b_ptrs[0]), 1,
                               static_cast<Coefficient>(0.0), c_ptrs[0], 1)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMV launch failed : a.shape=", in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      } else {
        bool blas_launch_status =
            stream
                ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                               static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                               adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                               static_cast<Coefficient>(0.0), c_ptrs[0], n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMM launch failed : a.shape=", in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      }
    } else if (use_strided_batched) {
      bool blas_launch_status =
          stream
              ->ThenBlasGemmStridedBatched(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), *b_ptrs[0], adj_y ? k : n,
                  b_stride, *a_ptrs[0], adj_x ? m : k, a_stride,
                  static_cast<Coefficient>(0.0), c_ptrs[0], n, c_stride,
                  batch_size)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMStridedBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    } else {
      BlasScratchAllocator scratch_allocator(context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                  adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                  batch_size, &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
  }
};

template <>
struct LaunchBatchMatMul<GPUDevice, Eigen::half> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y,
                     const MatMulBCast& bcast, Tensor* out) {
    typedef Eigen::half Scalar;
    constexpr perftools::gputools::blas::Transpose kTranspose =
        is_complex<Scalar>::value
            ? perftools::gputools::blas::Transpose::kConjugateTranspose
            : perftools::gputools::blas::Transpose::kTranspose;
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose, kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = bcast.output_batch_size();
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef perftools::gputools::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(bcast.x_batch_size());
    b_device_memory.reserve(bcast.y_batch_size());
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();

    uint64 a_stride;
    uint64 b_stride;
    uint64 c_stride;

    bool is_full_broadcast =
        std::min(bcast.x_batch_size(), bcast.y_batch_size()) == 1;
    bool use_strided_batched =
        (!bcast.IsBroadcastingRequired() || is_full_broadcast) &&
        batch_size > 1;
    if (use_strided_batched) {
      a_stride = bcast.x_batch_size() != 1 ? m * k : 0;
      b_stride = bcast.y_batch_size() != 1 ? k * n : 0;
      c_stride = m * n;
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    } else if (!bcast.IsBroadcastingRequired()) {
      for (int64 i = 0; i < batch_size; ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory.back());
        b_ptrs.push_back(&b_device_memory.back());
        c_ptrs.push_back(&c_device_memory.back());
      }
    } else {
      const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
      const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
      for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      }
      for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      }
      for (int64 i = 0; i < batch_size; ++i) {
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
        b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
        c_ptrs.push_back(&c_device_memory.back());
      }
    }

    typedef float Coefficient;

    // Blas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      // TODO(benbarsdell): Use fp16 Gemv if it becomes supported by CUBLAS
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                             static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                             adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                             static_cast<Coefficient>(0.0), c_ptrs[0], n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMM launch failed : a.shape=", in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k));
      }
    } else if (use_strided_batched) {
      bool blas_launch_status =
          stream
              ->ThenBlasGemmStridedBatched(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), *b_ptrs[0], adj_y ? k : n,
                  b_stride, *a_ptrs[0], adj_x ? m : k, a_stride,
                  static_cast<Coefficient>(0.0), c_ptrs[0], n, c_stride,
                  batch_size)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMStridedBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    } else {
      BlasScratchAllocator scratch_allocator(context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                  adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                  batch_size, &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#ifdef TENSORFLOW_USE_SYCL
template <typename Scalar>
struct ParallelMatMulKernelSYCL {
  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y,
                  const MatMulBCast& bcast, Tensor* out, int start, int limit) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    auto d = context->eigen_sycl_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;

      auto x = Tx.template chip<0>(x_batch_index);
      auto y = Ty.template chip<0>(y_batch_index);
      auto z = Tz.template chip<0>(i);
      z.device(d) = x.contract(y, contract_pairs);
    }
  }
};

template <typename Scalar>
struct LaunchBatchMatMul<SYCLDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y,
                     const MatMulBCast& bcast, Tensor* out) {
    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = bcast.output_batch_size();
    ParallelMatMulKernelSYCL<Scalar>::Run(context, in_x, in_y, adj_x, adj_y,
                                          bcast, out, 0, batch_size);
  }
};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename Scalar>
class BaseBatchMatMulOp : public OpKernel {
 public:
  explicit BaseBatchMatMulOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  ~BaseBatchMatMulOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    ValidateInputTensors(ctx, in0, in1);

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", adj_x_, " ", adj_y_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }
    Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));
    LaunchBatchMatMul<Device, Scalar>::Launch(
        ctx, in0_reshaped, in1_reshaped, adj_x_, adj_y_, bcast, &out_reshaped);
  }

 protected:
  virtual void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) = 0;

 private:
  bool adj_x_;
  bool adj_y_;
};

// BatchMatMul Op implementation which disallows broadcasting.
template <typename Device, typename Scalar>
class BatchMatMulOp : public BaseBatchMatMulOp<Device, Scalar> {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Scalar>(context) {}

  ~BatchMatMulOp() override {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    // Disallow broadcasting support. Ensure that all batch dimensions of the
    // input tensors match.
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in1.dim_size(i),
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[1].dim(", i,
                      ") must be the same: ", in0.shape().DebugString(), " vs ",
                      in1.shape().DebugString()));
    }
  }
};

// BatchMatMul Op implementation with broadcasting support.
template <typename Device, typename Scalar>
class BatchMatMulV2Op : public BaseBatchMatMulOp<Device, Scalar> {
 public:
  explicit BatchMatMulV2Op(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Scalar>(context) {}

  ~BatchMatMulV2Op() override {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulOp.
    OP_REQUIRES(
        ctx, in0.dims() >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims()));
    OP_REQUIRES(
        ctx, in1.dims() >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims()));
  }
};

#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<CPUDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<CPUDevice, TYPE>)

#define REGISTER_BATCH_MATMUL_GPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<GPUDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<GPUDevice, TYPE>)

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_BATCH_MATMUL_SYCL(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BatchMatMul").Device(DEVICE_SYCL).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<SYCLDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("BatchMatMulV2").Device(DEVICE_SYCL).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<SYCLDevice, TYPE>)
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_
