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

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/host_or_device_scalar.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/kernels/matmul_util.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_blas_lt.h"
#endif  // GOOGLE_CUDA
#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if TF_HIPBLASLT
#include "xla/stream_executor/rocm/hip_blas_lt.h"
#endif
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

// Returns the pair of dimensions along which to perform Tensor contraction to
// emulate matrix multiplication.
// For matrix multiplication of 2D Tensors X and Y, X is contracted along
// second dimension and Y is contracted along the first dimension (if neither X
// nor Y is adjointed). The dimension to contract along is switched when any
// operand is adjointed.
// See http://en.wikipedia.org/wiki/Tensor_contraction
inline Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x,
                                                           bool adj_y) {
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
                  const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, const MatMulBCast& bcast, Tensor* out,
                  int batch_size) {
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
    contract_pairs[0] = ContractionDims(adj_x || trans_x, adj_y || trans_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    // TODO(rmlarsen): Consider launching these contractions asynchronously.
    for (int64_t i = 0; i < batch_size; ++i) {
      const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;

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
                  const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, const MatMulBCast& bcast, Tensor* out,
                  int batch_size) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x || trans_x, adj_y || trans_y);
    if (batch_size == 1 && !should_bcast) {
      auto Tx = in_x.flat_inner_dims<Scalar, 2>();
      auto Ty = in_y.flat_inner_dims<Scalar, 2>();
      auto Tz = out->flat_inner_dims<Scalar, 2>();
      Tz.device(d) = Tx.contract(Ty, contract_pairs);
    } else {
      auto Tx = in_x.tensor<Scalar, 3>();
      auto Ty = in_y.tensor<Scalar, 3>();
      auto Tz = out->tensor<Scalar, 3>();
      const auto& x_batch_indices = bcast.x_batch_indices();
      const auto& y_batch_indices = bcast.y_batch_indices();
      // TODO(rmlarsen): Consider launching these contractions asynchronously.
      for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
        const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
        auto x = Tx.template chip<0>(x_batch_index);
        auto y = Ty.template chip<0>(y_batch_index);
        auto z = Tz.template chip<0>(i);

        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

// Basic y-combinator implementation.
template <class Func>
struct YCombinatorImpl {
  Func func;
  template <class... Args>
  decltype(auto) operator()(Args&&... args) const {
    return func(*this, std::forward<Args>(args)...);
  }
};

template <class Func>
YCombinatorImpl<std::decay_t<Func>> YCombinator(Func&& func) {
  return YCombinatorImpl<std::decay_t<Func>>{std::forward<Func>(func)};
}

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
                  bool adj_y, bool trans_x, bool trans_y,
                  const MatMulBCast& bcast, Tensor* out, int start, int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64_t i = start; i < limit; ++i) {
      const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto x = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto y = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto z = TensorSliceToEigenMatrix(out, i);
      // Assume at most one of adj_x or trans_x is true. Similarly, for adj_y
      // and trans_y.
      if (!adj_x && !trans_x) {
        if (!adj_y && !trans_y) {
          z.noalias() = x * y;
        } else if (adj_y) {
          z.noalias() = x * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x * y.transpose();
        }
      } else if (adj_x) {
        if (!adj_y && !trans_y) {
          z.noalias() = x.adjoint() * y;
        } else if (adj_y) {
          z.noalias() = x.adjoint() * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x.adjoint() * y.transpose();
        }
      } else {  // trans_x == true
        if (!adj_y && !trans_y) {
          z.noalias() = x.transpose() * y;
        } else if (adj_y) {
          z.noalias() = x.transpose() * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x.transpose() * y.transpose();
        }
      }
    }
  }
};

// For single-batch multiplications, manually parallize by splitting the output
// matrix.
template <typename Scalar>
struct SingleBatchParallelMatMulKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorToEigenMatrix(const Tensor& t) {
    return ConstMatrixMap(t.flat<Scalar>().data(), t.dim_size(1),
                          t.dim_size(2));
  }

  static MatrixMap TensorToEigenMatrix(Tensor* t) {
    return MatrixMap(t->flat<Scalar>().data(), t->dim_size(1), t->dim_size(2));
  }

  static void Run(const CPUDevice& device, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, Tensor* out) {
    using Eigen::Index;
    Eigen::ThreadPoolInterface* pool = device.getPool();

    Index m = (trans_x || adj_x) ? in_x.dim_size(2) : in_x.dim_size(1);
    Index k = (trans_x || adj_x) ? in_x.dim_size(1) : in_x.dim_size(2);
    Index n = (trans_y || adj_y) ? in_y.dim_size(1) : in_y.dim_size(2);

    auto x_mat = ConstTensorToEigenMatrix(in_x);
    auto y_mat = ConstTensorToEigenMatrix(in_y);
    auto out_mat = TensorToEigenMatrix(out);

    // Computes a block of the output matrix.
    auto compute_matmul_block = [&x_mat, &y_mat, &out_mat, adj_x, trans_x,
                                 adj_y, trans_y](Index row, Index col,
                                                 Index nrows, Index ncols) {
      auto z = out_mat.block(row, col, nrows, ncols);

      // Assume at most one of adj_x or trans_x is true. Similarly, for adj_y
      // and trans_y.
      if (!adj_x && !trans_x) {
        auto x = x_mat.middleRows(row, nrows);
        if (!adj_y && !trans_y) {
          auto y = y_mat.middleCols(col, ncols);
          z = x * y;
        } else if (adj_y) {
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x * y.adjoint();
        } else {  // trans_y == true
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x * y.transpose();
        }
      } else if (adj_x) {
        auto x = x_mat.middleCols(row, nrows);
        if (!adj_y && !trans_y) {
          auto y = y_mat.middleCols(col, ncols);
          z.noalias() = x.adjoint() * y;
        } else if (adj_y) {
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x.adjoint() * y.adjoint();
        } else {  // trans_y == true
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x.adjoint() * y.transpose();
        }
      } else {  // trans_x == true
        auto x = x_mat.middleCols(row, nrows);
        if (!adj_y && !trans_y) {
          auto y = y_mat.middleCols(col, ncols);
          z.noalias() = x.transpose() * y;
        } else if (adj_y) {
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x.transpose() * y.adjoint();
        } else {  // trans_y == true
          auto y = y_mat.middleRows(col, ncols);
          z.noalias() = x.transpose() * y.transpose();
        }
      }
    };

    // Split the work across n threads, unless the total amount of work
    // is small (e.g. 128 * 128) - in which case use fewer threads.  This is
    // the same heuristic value used in LaunchBatchMatMul below.
    const int64_t kMaxCostOuterParallelism = 128 * 128;
    Index work_limit = std::max<Index>((m * k * n) / pool->NumThreads(),
                                       kMaxCostOuterParallelism);
    // Blocks should have a size no smaller than 8 * kPacketSize, except perhaps
    // for tail blocks.
    constexpr int kPacketSize = Eigen::internal::packet_traits<Scalar>::size;
    constexpr Index kBlockMin = 8 * kPacketSize;

    // Precompute how many blocks there will be.
    auto compute_blocks = YCombinator([k, work_limit, kBlockMin](
                                          auto& compute_blocks, Index row,
                                          Index col, Index nrows,
                                          Index ncols) -> Index {
      Index work = nrows * k * ncols;
      Index blocks = 0;
      while (work > work_limit && (nrows > kBlockMin || ncols > kBlockMin)) {
        if (nrows > ncols) {
          Index half = Eigen::divup(nrows / 2, kBlockMin) * kBlockMin;
          blocks += 1 + compute_blocks(row + half, col, nrows - half, ncols);
          nrows = half;
        } else {
          Index half = Eigen::divup(ncols / 2, kBlockMin) * kBlockMin;
          blocks += 1 + compute_blocks(row, col + half, nrows, ncols - half);
          ncols = half;
        }
        work = nrows * k * ncols;
      }
      return blocks;
    });
    Index total_blocks = 1 + compute_blocks(0, 0, m, n);

    // Recursively split work according to the exact same heuristic as above.
    Eigen::Barrier barrier(total_blocks);
    auto handle_range = YCombinator(
        [k, pool, &barrier, work_limit, kBlockMin, &compute_matmul_block](
            auto& handle_range, Index row, Index col, Index nrows,
            Index ncols) -> void {
          Index work = nrows * k * ncols;
          while (work > work_limit &&
                 (nrows > kBlockMin || ncols > kBlockMin)) {
            if (nrows > ncols) {
              Index half = Eigen::divup(nrows / 2, kBlockMin) * kBlockMin;
              pool->Schedule([&handle_range, row, half, col, nrows, ncols]() {
                handle_range(row + half, col, nrows - half, ncols);
              });
              nrows = half;
            } else {
              Index half = Eigen::divup(ncols / 2, kBlockMin) * kBlockMin;
              pool->Schedule([&handle_range, row, half, col, nrows, ncols]() {
                handle_range(row, col + half, nrows, ncols - half);
              });
              ncols = half;
            }
            work = nrows * k * ncols;
          }

          if (nrows > 0 && ncols > 0) {
            // Compute the output block.
            compute_matmul_block(row, col, nrows, ncols);
          }
          barrier.Notify();
        });
    handle_range(0, 0, m, n);
    barrier.Wait();
  }
};

}  // namespace

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

template <typename Scalar>
struct LaunchBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                     bool trans_y, bool grad_x, bool grad_y,
                     const MatMulBCast& bcast, Tensor* out) {
    typedef ParallelMatMulKernel<Scalar, Eigen::NumTraits<Scalar>::IsComplex>
        ParallelMatMulKernel;
    bool conjugate_result = false;

    // Number of matrix multiplies i.e. size of the batch.
    const int64_t batch_size = bcast.output_batch_size();
    const int64_t cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    const int64_t small_dim = std::min(
        std::min(in_x.dim_size(1), in_x.dim_size(2)), out->dim_size(2));
    // NOTE(nikhilsarda): This heuristic is optimal in benchmarks as of
    // Jan 21, 2020.
    const int64_t kMaxCostOuterParallelism = 128 * 128;  // heuristic.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // TODO(rmlarsen): Reconsider the heuristics now that we have asynchronous
    // evaluation in Eigen Tensor.
    if (small_dim > 1 &&
        (batch_size == 1 || cost_per_unit > kMaxCostOuterParallelism)) {
      // Parallelize over inner dims.
      // For large matrix products it is counter-productive to parallelize
      // over the batch dimension.
      ParallelMatMulKernel::Run(context, in_x, in_y, adj_x, adj_y, trans_x,
                                trans_y, bcast, out, batch_size);
      conjugate_result = adj_x;
    } else if (batch_size > 1) {
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, adj_x, adj_y, trans_x, trans_y, &bcast, out](
                int start, int limit) {
              SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y,
                                                  trans_x, trans_y, bcast, out,
                                                  start, limit);
            });
    } else if (cost_per_unit > kMaxCostOuterParallelism) {
      // Split along output blocks.
      SingleBatchParallelMatMulKernel<Scalar>::Run(context->eigen_cpu_device(),
                                                   in_x, in_y, adj_x, adj_y,
                                                   trans_x, trans_y, out);
    } else {
      // Single small multiplication.
      SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y, trans_x,
                                          trans_y, bcast, out, 0, batch_size);
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

#if GOOGLE_CUDA || TF_HIPBLASLT

namespace {
// A dummy type to group matmul autotune results together.
struct BlasLtMatmulAutoTuneGroup {
  static string name() { return "MatmulLt"; }
};

typedef AutotuneSingleton<BlasLtMatmulAutoTuneGroup, BlasLtMatmulPlanParams,
                          se::blas::AlgorithmConfig,
                          absl::Hash<BlasLtMatmulPlanParams>>
    AutoTuneBatchMatmul;

}  // namespace

#endif  // GOOGLE_CUDA
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class BlasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  BlasScratchAllocator(OpKernelContext* context)
      : memory_limit_(0), total_byte_size_(0), context_(context) {}

  BlasScratchAllocator(OpKernelContext* context, int64_t memory_limit)
      : memory_limit_(memory_limit), total_byte_size_(0), context_(context) {}

  int64_t GetMemoryLimitInBytes() override { return memory_limit_; }

  tsl::StatusOr<DeviceMemoryBytes> AllocateBytes(int64_t byte_size) override {
    Tensor temporary_memory;

    if (memory_limit_ > 0 && byte_size > memory_limit_) {
      return tsl::Status{
          absl::StatusCode::kUnavailable,
          absl::StrCat("Requested memory size (", byte_size,
                       ") exceeds the memory limit (", memory_limit_, ").")};
    }
    AllocationAttributes allocation_attr;
    allocation_attr.retry_on_failure = false;
    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return tsl::Status{
          absl::StatusCode::kUnavailable,
          absl::StrCat("Failed to allocate requested memory of (", byte_size,
                       ").")};
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return tsl::StatusOr<DeviceMemoryBytes>(DeviceMemoryBytes::MakeFromByteSize(
        temporary_memory.flat<uint8>().data(),
        temporary_memory.flat<uint8>().size()));
  }
  int64 TotalByteSize() { return total_byte_size_; }

 private:
  int64_t memory_limit_;
  int64_t total_byte_size_;
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};

template <typename Scalar>
struct LaunchBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                     bool trans_y, bool grad_x, bool grad_y,
                     const MatMulBCast& bcast, Tensor* out) {
    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   se::blas::Transpose::kTranspose,
                                   se::blas::Transpose::kConjugateTranspose};
    const uint64 m = in_x.dim_size(adj_x || trans_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x || trans_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y || trans_y ? 1 : 2);
    const int64_t batch_size = bcast.output_batch_size();
    auto blas_transpose_a = trans[adj_x ? 2 : (trans_x ? 1 : 0)];
    auto blas_transpose_b = trans[adj_y ? 2 : (trans_y ? 1 : 0)];

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

    // Use float as coefficient type for half and bfloat16 precision inputs,
    // otherwise use the input type.
    constexpr bool is_16bit_input = std::is_same_v<Scalar, Eigen::half> ||
                                    std::is_same_v<Scalar, Eigen::bfloat16>;
    using Coefficient = std::conditional_t<is_16bit_input, float, Scalar>;

    se::blas::CallContext call_context = se::blas::CallContext::kNone;
    OP_REQUIRES(context, grad_x == false || grad_y == false,
                errors::InvalidArgument(
                    "At least 1 of grad_x and grad_y shall be false"));
    if (grad_x) {
      call_context = se::blas::CallContext::kBackpropInput1;
    }
    if (grad_y) {
      call_context = se::blas::CallContext::kBackpropInput2;
    }
#if GOOGLE_CUDA || TF_HIPBLASLT
    static const bool use_autotune = MatmulAutotuneEnable();
    bool bCublasLtSupport = true;

    const auto& cc =
        stream->parent()->GetDeviceDescription().gpu_compute_capability();
    if (auto* procm = std::get_if<se::RocmComputeCapability>(&cc)) {
      bCublasLtSupport = procm->gfx9_mi200_or_later();
    }

    if (EnableCublasLtGemm() && bCublasLtSupport) {
      static const int64_t max_scratch_size =
          GetWorkspaceLimit(1LL << 32);  // 4GB by default

      bool requires_mixed_broadcasting =
          bcast.IsBroadcastingRequired() && !is_full_broadcast;

      if (!requires_mixed_broadcasting) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr));
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr));
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr));
        a_ptrs.push_back(&a_device_memory.back());
        b_ptrs.push_back(&b_device_memory.back());
        c_ptrs.push_back(&c_device_memory.back());

        BlasLtMatmulPlanParams matmul_params{
            se::blas::ToDataType<Scalar>::value,
            static_cast<size_t>(m),
            static_cast<size_t>(n),
            static_cast<size_t>(k),
            blas_transpose_a,
            blas_transpose_b,
            static_cast<size_t>(batch_size),
            /*broadcast_a=*/bcast.x_batch_size() == 1,
            /*broadcast_b=*/bcast.y_batch_size() == 1};

        std::optional<int> max_algorithm_count;
        if (!use_autotune) max_algorithm_count = 1;
        absl::Mutex* pmu = nullptr;
        auto plan_and_algorithms_or = PlanAndAlgorithms::GetOrCreate(
            stream, matmul_params, &pmu, max_algorithm_count);
        OP_REQUIRES_OK(context, plan_and_algorithms_or.status());
        absl::MutexLock lock(pmu);
        const auto* plan_and_algorithms =
            std::move(plan_and_algorithms_or).value();
        auto n_algorithms = plan_and_algorithms->algorithms.size();

        se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);
        if (!use_autotune) {
          algorithm_config.set_algorithm(0);
        } else if (!AutoTuneBatchMatmul::GetInstance()->Find(
                       matmul_params, &algorithm_config)) {
          VLOG(4) << "Autotuning BlasLtMatmul over " << n_algorithms
                  << " algorithms.";
          se::blas::ProfileResult best_result;
          se::blas::ProfileResult profile_result;

          for (size_t i = 0; i != n_algorithms; ++i) {
            // Create a new scratch allocator with every autotuning run so that
            // scratch space is deallocated between runs.
            BlasScratchAllocator scratch_allocator(context, max_scratch_size);
            Status cublas_launch_status = plan_and_algorithms->ExecuteOnStream(
                stream, *a_ptrs[0], *b_ptrs[0], *c_ptrs[0], i,
                scratch_allocator, se::DeviceMemoryBase{}, &profile_result);

            VLOG(4) << "  Autotune algorithm " << i
                    << " result: " << profile_result.elapsed_time_in_ms()
                    << " ms, valid=" << profile_result.is_valid()
                    << ", workspace_size="
                    << plan_and_algorithms->algorithms[i].workspace_size;

            if (cublas_launch_status.ok() && profile_result.is_valid() &&
                profile_result.elapsed_time_in_ms() <
                    best_result.elapsed_time_in_ms()) {
              best_result = profile_result;
              // Use index into algorithms array, instead of cublas internal ID.
              best_result.set_algorithm(i);
            }
          }

          if (best_result.is_valid()) {
            algorithm_config.set_algorithm(best_result.algorithm());
          }
          // Each matmul parameter set gets one pass of
          // autotune. If no algorithms works, kNoAlgorithm is added to the
          // autotune map.
          AutoTuneBatchMatmul::GetInstance()->Insert(matmul_params,
                                                     algorithm_config);
        }
        se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
        OP_REQUIRES(context, 0 <= algorithm_idx && algorithm_idx < n_algorithms,
                    errors::Internal("Missing/invalid BatchMatmul algorithm"));
        BlasScratchAllocator scratch_allocator(context, max_scratch_size);
        VLOG(4) << "Calling BlasLtMatMul: a.shape=(" << bcast.x_batch_size()
                << ", " << in_x.dim_size(1) << ", " << in_x.dim_size(2)
                << "), b.shape=(" << bcast.y_batch_size() << ", "
                << in_y.dim_size(1) << ", " << in_y.dim_size(2) << "), m=" << m
                << ", n=" << n << ", k=" << k << ", batch_size=" << batch_size
                << "trans_x = " << trans_x << "trans_y = " << trans_y
                << "adj_x = " << adj_x << "adj_y = " << adj_y;

        OP_REQUIRES_OK(context, plan_and_algorithms->ExecuteOnStream(
                                    stream, *a_ptrs[0], *b_ptrs[0], *c_ptrs[0],
                                    algorithm_idx, scratch_allocator));
      } else {  // requires mixed broadcasting
        const std::vector<int64_t>& a_batch_indices = bcast.x_batch_indices();
        const std::vector<int64_t>& b_batch_indices = bcast.y_batch_indices();
        for (int64_t i = 0; i < bcast.x_batch_size(); ++i) {
          a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        }
        for (int64_t i = 0; i < bcast.y_batch_size(); ++i) {
          b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
        }
        for (int64_t i = 0; i < batch_size; ++i) {
          c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
          a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
          b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
          c_ptrs.push_back(&c_device_memory.back());
        }

        BlasScratchAllocator scratch_allocator(context, max_scratch_size);
        auto blas = stream->parent()->AsBlas();
        OP_REQUIRES(context, blas != nullptr,
                    absl::InternalError("No blas support for stream"));
        bool blas_launch_status = blas->DoBlasGemmBatched(
            stream, blas_transpose_b, blas_transpose_a, n, m, k,
            static_cast<Coefficient>(1.0), b_ptrs, adj_y || trans_y ? k : n,
            a_ptrs, adj_x || trans_x ? m : k, static_cast<Coefficient>(0.0),
            c_ptrs, n, batch_size, GetNumericOptions(), &scratch_allocator,
            call_context);
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMMBatched launch failed: a.shape=",
              in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k, ", batch_size=", batch_size));
        }
      }
    } else {
#endif  // GOOGLE_CUDA
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
        for (int64_t i = 0; i < batch_size; ++i) {
          a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
          b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
          c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
          a_ptrs.push_back(&a_device_memory.back());
          b_ptrs.push_back(&b_device_memory.back());
          c_ptrs.push_back(&c_device_memory.back());
        }
      } else {
        const std::vector<int64_t>& a_batch_indices = bcast.x_batch_indices();
        const std::vector<int64_t>& b_batch_indices = bcast.y_batch_indices();
        for (int64_t i = 0; i < bcast.x_batch_size(); ++i) {
          a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        }
        for (int64_t i = 0; i < bcast.y_batch_size(); ++i) {
          b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
        }
        for (int64_t i = 0; i < batch_size; ++i) {
          c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
          a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
          b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
          c_ptrs.push_back(&c_device_memory.back());
        }
      }

      // Blas does
      // C = A x B
      // where A, B and C are assumed to be in column major.
      // We want the output to be in row-major, so we can compute
      // C' = B' x A', where ' stands for transpose (not adjoint).
      // TODO(yangzihao): Choose the best of the three strategies using
      // autotune.
      auto blas = stream->parent()->AsBlas();
      OP_REQUIRES(context, blas != nullptr,
                  absl::InternalError("No blas support for stream"));
      if (batch_size == 1) {
        // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
        // overhead of the scratch allocator and the batch interface.
        // TODO(benbarsdell): Use fp16 Gemv if it becomes supported by CUBLAS
        if constexpr (!std::is_same_v<Scalar, Eigen::half> &&
                      !std::is_same_v<Scalar, Eigen::bfloat16>) {
          if (n == 1 &&
              blas_transpose_b != se::blas::Transpose::kConjugateTranspose &&
              blas_transpose_a != se::blas::Transpose::kConjugateTranspose) {
            // This is a matrix*vector multiply so use GEMV to compute A * b.
            // Here we are multiplying in the natural order, so we have to flip
            // the transposition flag to compensate for the tensor being stored
            // row-major. Since GEMV doesn't provide a way to just conjugate an
            // argument, we have to defer those cases to GEMM below.
            auto gemv_trans_a =
                blas_transpose_a == se::blas::Transpose::kTranspose
                    ? se::blas::Transpose::kNoTranspose
                    : se::blas::Transpose::kTranspose;
            bool blas_launch_status = blas->DoBlasGemv(
                stream, gemv_trans_a, adj_x || trans_x ? m : k,
                adj_x || trans_x ? k : m, static_cast<Coefficient>(1.0),
                *(a_ptrs[0]), adj_x || trans_x ? m : k, *(b_ptrs[0]), 1,
                static_cast<Coefficient>(0.0), c_ptrs[0], 1);
            if (!blas_launch_status) {
              context->SetStatus(errors::Internal(
                  "Blas xGEMV launch failed : a.shape=",
                  in_x.shape().DebugString(), ", b.shape=",
                  in_y.shape().DebugString(), ", m=", m, ", n=", n, ", k=", k));
            }
            return;
          }
        }

        OP_REQUIRES_OK(
            context,
            blas->BlasGemm(stream, blas_transpose_b, blas_transpose_a, n, m, k,
                           *(b_ptrs[0]), adj_y || trans_y ? k : n, *(a_ptrs[0]),
                           adj_x || trans_x ? m : k, c_ptrs[0], n,
                           GetNumericOptions(), call_context));
      } else if (use_strided_batched) {
        OP_REQUIRES_OK(
            context, blas->BlasGemmStridedBatched(
                         stream, blas_transpose_b, blas_transpose_a, n, m, k,
                         static_cast<Coefficient>(1.0), *b_ptrs[0],
                         adj_y || trans_y ? k : n, b_stride, *a_ptrs[0],
                         adj_x || trans_x ? m : k, a_stride,
                         static_cast<Coefficient>(0.0), c_ptrs[0], n, c_stride,
                         batch_size, GetNumericOptions(), call_context));
      } else {
        BlasScratchAllocator scratch_allocator(context);
        bool blas_launch_status = blas->DoBlasGemmBatched(
            stream, blas_transpose_b, blas_transpose_a, n, m, k,
            static_cast<Coefficient>(1.0), b_ptrs, adj_y || trans_y ? k : n,
            a_ptrs, adj_x || trans_x ? m : k, static_cast<Coefficient>(0.0),
            c_ptrs, n, batch_size, GetNumericOptions(), &scratch_allocator,
            call_context);
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMMBatched launch failed : a.shape=",
              in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k, ", batch_size=", batch_size));
        }
      }
#if GOOGLE_CUDA || TF_HIPBLASLT
    }
#endif  // GOOGLE_CUDA
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename T>
inline void FastConvertToFloat(const T* src, float* dst, int64_t size) {
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> src_eigen(src, size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.template cast<float>();
}

template <typename T>
inline void FastConvertFromFloat(const float* src, T* dst, int64_t size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.template cast<T>();
}

template <>
inline void FastConvertToFloat<bfloat16>(const bfloat16* src, float* dst,
                                         int64_t size) {
  BFloat16ToFloat(src, dst, size);
}

template <>
inline void FastConvertFromFloat<bfloat16>(const float* src, bfloat16* dst,
                                           int64_t size) {
  FloatToBFloat16(src, dst, size);
}

template <typename Device, typename Ta, typename Tb, typename Tout>
class BaseBatchMatMulOp : public OpKernel {
 public:
  explicit BaseBatchMatMulOp(OpKernelConstruction* context,
                             bool is_legacy_matmul)
      : OpKernel(context) {
    if (is_legacy_matmul) {
      // The old MatMul kernel has "transpose_a/transpose_b" attributes.
      OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &trans_x_));
      OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &trans_y_));
      adj_x_ = false;
      adj_y_ = false;
      OP_REQUIRES_OK(context, context->GetAttr("grad_a", &grad_input_1_));
      OP_REQUIRES_OK(context, context->GetAttr("grad_b", &grad_input_2_));
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
      OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
      trans_x_ = false;
      trans_y_ = false;
      OP_REQUIRES_OK(context, context->GetAttr("grad_x", &grad_input_1_));
      OP_REQUIRES_OK(context, context->GetAttr("grad_y", &grad_input_2_));
    }
  }

  ~BaseBatchMatMulOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    const absl::Status s = ValidateInputTensors(ctx, in0, in1);
    if (!s.ok()) {
      ctx->SetStatus(s);
      return;
    }

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
    if (adj_x_ || trans_x_) std::swap(d0, d1);
    if (adj_y_ || trans_y_) std::swap(d2, d3);
    OP_REQUIRES(
        ctx, d1 == d2,
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", in0.shape().DebugString(),
            ", In[1]: ", in1.shape().DebugString()));
    OP_REQUIRES_OK(ctx, out_shape.AddDimWithStatus(d0));
    OP_REQUIRES_OK(ctx, out_shape.AddDimWithStatus(d3));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Tout> f;
      f(ctx->eigen_device<Device>(), out->flat<Tout>());
      return;
    }
    Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));

    // b/307285203: There seems to be an overly aggressive compiler optimization
    // that optimizes away these data pointers unless we explicitly check them.
    OP_REQUIRES(ctx,
                in0_reshaped.data() != nullptr &&
                    in1_reshaped.data() != nullptr &&
                    out_reshaped.data() != nullptr,
                absl::InternalError("Null data pointer encountered."));
    if constexpr (std::is_same_v<Device, CPUDevice> && std::is_same_v<Ta, Tb> &&
                  (std::is_same_v<Ta, bfloat16> ||
                   std::is_same_v<Ta, Eigen::half>)) {
      Tensor in0_reshaped_float, in1_reshaped_float, out_reshaped_float;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, in0_reshaped.shape(),
                                             &in0_reshaped_float));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, in1_reshaped.shape(),
                                             &in1_reshaped_float));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, out_reshaped.shape(),
                                             &out_reshaped_float));

      // TODO: Avoid extra copy to make (b)float16 matmul efficient on CPU.
      FastConvertToFloat(in0_reshaped.flat<Ta>().data(),
                         in0_reshaped_float.flat<float>().data(),
                         in0_reshaped.NumElements());
      FastConvertToFloat(in1_reshaped.flat<Tb>().data(),
                         in1_reshaped_float.flat<float>().data(),
                         in1_reshaped.NumElements());

      LaunchBatchMatMul<Device, float>::Launch(
          ctx, in0_reshaped_float, in1_reshaped_float, adj_x_, adj_y_, trans_x_,
          trans_y_, grad_input_1_, grad_input_2_, bcast, &out_reshaped_float);
      FastConvertFromFloat<Tout>(out_reshaped_float.flat<float>().data(),
                                 out_reshaped.flat<Tout>().data(),
                                 out->NumElements());
    } else {
      // Cast tensor to desired type to reuse Eigen.
      // TODO(b/178749687): remove this cast if Eigen supports this natively.
      if constexpr (!std::is_same<Ta, Tout>::value) {
        in0_reshaped = CastTensor<Ta, Tout>(in0_reshaped);
      }
      if constexpr (!std::is_same<Tb, Tout>::value) {
        in1_reshaped = CastTensor<Tb, Tout>(in1_reshaped);
      }
      LaunchBatchMatMul<Device, Tout>::Launch(
          ctx, in0_reshaped, in1_reshaped, adj_x_, adj_y_, trans_x_, trans_y_,
          grad_input_1_, grad_input_2_, bcast, &out_reshaped);
    }
  }

 protected:
  virtual absl::Status ValidateInputTensors(OpKernelContext* ctx,
                                            const Tensor& in0,
                                            const Tensor& in1) = 0;

 private:
  // TODO(171979567) Make the ops take both adj and transpose attributes.
  bool adj_x_ = false;
  bool adj_y_ = false;
  bool trans_x_ = false;
  bool trans_y_ = false;
  bool grad_input_1_ = false;
  bool grad_input_2_ = false;

  // Cast `t` from `SrcT` to `DstT`.
  template <typename SrcT, typename DstT>
  Tensor CastTensor(const Tensor& t) {
    Tensor res = Tensor(DataTypeToEnum<DstT>::v(), t.shape());
    res.flat<DstT>() = t.flat<SrcT>().template cast<DstT>();
    return res;
  }
};

// BatchMatMul Op implementation which disallows broadcasting.
template <typename Device, typename Ta, typename Tb, typename Tout,
          bool is_legacy_matmul = false>
class BatchMatMulOp : public BaseBatchMatMulOp<Device, Ta, Tb, Tout> {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Ta, Tb, Tout>(context, is_legacy_matmul) {}

  ~BatchMatMulOp() override {}

 private:
  absl::Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) override {
    // Disallow broadcasting support. Ensure that all batch dimensions of the
    // input tensors match.
    if (in0.dims() != in1.dims()) {
      return errors::InvalidArgument(
          "In[0] and In[1] has different ndims: ", in0.shape().DebugString(),
          " vs. ", in1.shape().DebugString());
    }
    const int ndims = in0.dims();
    if (is_legacy_matmul) {
      if (ndims != 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be == 2: ",
                                       ndims);
      }
    } else {
      if (ndims < 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ",
                                       ndims);
      }
      for (int i = 0; i < ndims - 2; ++i) {
        if (in0.dim_size(i) != in1.dim_size(i)) {
          return errors::InvalidArgument(
              "In[0].dim(", i, ") and In[1].dim(", i,
              ") must be the same: ", in0.shape().DebugString(), " vs ",
              in1.shape().DebugString());
        }
      }
    }
    return absl::OkStatus();
  }
};

// BatchMatMul Op implementation with broadcasting support.
template <typename Device, typename Ta, typename Tb, typename Tout>
class BatchMatMulV2Op : public BaseBatchMatMulOp<Device, Ta, Tb, Tout> {
 public:
  explicit BatchMatMulV2Op(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Ta, Tb, Tout>(context,
                                                /* is_legacy_matmul= */ false) {
  }

  ~BatchMatMulV2Op() override {}

 private:
  absl::Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) override {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulOp.
    if (in0.dims() < 2) {
      return errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims());
    }
    if (in1.dims() < 2) {
      return errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims());
    }
    return absl::OkStatus();
  }
};

// Register for MatMul, BatchMatMul, BatchMatMulv2 where Tin = Tout.
#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE>);                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<CPUDevice, TYPE, TYPE, TYPE>);                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),        \
      BatchMatMulOp<CPUDevice, TYPE, TYPE, TYPE, /* is_legacy_matmul=*/true>)

#define REGISTER_BATCH_MATMUL_GPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE>);                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<GPUDevice, TYPE, TYPE, TYPE>);                      \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),        \
      BatchMatMulOp<GPUDevice, TYPE, TYPE, TYPE, /* is_legacy_matmul=*/true>)

// Register for BatchMatMulv3 where Ta, Tb and Tout are not the same.
#define REGISTER_BATCH_MATMUL_TOUT_CPU(Ta, Tb, Tout)         \
  REGISTER_KERNEL_BUILDER(Name("BatchMatMulV3")              \
                              .Device(DEVICE_CPU)            \
                              .TypeConstraint<Ta>("Ta")      \
                              .TypeConstraint<Tb>("Tb")      \
                              .TypeConstraint<Tout>("Tout"), \
                          BatchMatMulV2Op<CPUDevice, Ta, Tb, Tout>)

#define REGISTER_BATCH_MATMUL_TOUT_GPU(Ta, Tb, Tout)         \
  REGISTER_KERNEL_BUILDER(Name("BatchMatMulV3")              \
                              .Device(DEVICE_GPU)            \
                              .TypeConstraint<Ta>("Ta")      \
                              .TypeConstraint<Tb>("Tb")      \
                              .TypeConstraint<Tout>("Tout"), \
                          BatchMatMulV2Op<GPUDevice, Ta, Tb, Tout>)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_
