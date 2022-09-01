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

// See docs in ../ops/linalg_ops.cc.
//
#ifndef TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename Scalar>
se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* gpu_memory) {
  se::DeviceMemoryBase wrapped(const_cast<Scalar*>(gpu_memory));
  se::DeviceMemory<Scalar> typed(wrapped);
  return typed;
}

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Sequential batch matrix triangular solve kernel that calls Eigen's
// matrix triangular solve.
template <typename Scalar>
struct SequentialMatrixTriangularSolveKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

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

  static void Run(const Tensor& in_x, const Tensor& in_y, bool lower,
                  bool adjoint, const MatMulBCast& bcast, Tensor* out,
                  int start, int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64_t i = start; i < limit; ++i) {
      const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto matrix = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto rhs = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto output = TensorSliceToEigenMatrix(out, i);
      if (lower) {
        auto triangle = matrix.template triangularView<Eigen::Lower>();
        if (adjoint) {
          output.noalias() = triangle.adjoint().solve(rhs);
        } else {
          output.noalias() = triangle.solve(rhs);
        }
      } else {
        auto triangle = matrix.template triangularView<Eigen::Upper>();
        if (adjoint) {
          output.noalias() = triangle.adjoint().solve(rhs);
        } else {
          output.noalias() = triangle.solve(rhs);
        }
      }
    }
  }
};

template <typename Device, typename Scalar>
struct LaunchBatchMatrixTriangularSolve;

template <typename Scalar>
struct LaunchBatchMatrixTriangularSolve<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adjoint, bool lower,
                     const MatMulBCast& bcast, Tensor* out) {
    // Number of matrix triangular solves i.e. size of the batch.
    const int64_t batch_size = bcast.output_batch_size();
    const int64_t cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(1) * in_y.dim_size(2) / 2;
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          cost_per_unit,
          [&in_x, &in_y, adjoint, lower, &bcast, out](int start, int limit) {
            SequentialMatrixTriangularSolveKernel<Scalar>::Run(
                in_x, in_y, lower, adjoint, bcast, out, start, limit);
          });
  }
};

template <typename Device, typename Scalar>
class BaseMatrixTriangularSolveOp : public OpKernel {
 public:
  explicit BaseMatrixTriangularSolveOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  ~BaseMatrixTriangularSolveOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    ValidateInputTensors(ctx, in0, in1);
    if (!ctx->status().ok()) {
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
    if (adjoint_) std::swap(d0, d1);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", lower_, " ", adjoint_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));
    LaunchBatchMatrixTriangularSolve<Device, Scalar>::Launch(
        ctx, in0_reshaped, in1_reshaped, adjoint_, lower_, bcast,
        &out_reshaped);
  }

 private:
  virtual void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                    const Tensor& in1) = 0;
  bool lower_;
  bool adjoint_;
};

template <class Device, class Scalar>
class MatrixTriangularSolveOp
    : public BaseMatrixTriangularSolveOp<Device, Scalar> {
 public:
  explicit MatrixTriangularSolveOp(OpKernelConstruction* context)
      : BaseMatrixTriangularSolveOp<Device, Scalar>(context) {}

  ~MatrixTriangularSolveOp() override {}

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) override {
    const auto in0_num_dims = in0.dims();
    OP_REQUIRES(
        ctx, in0_num_dims >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0_num_dims));

    const auto in1_num_dims = in1.dims();
    OP_REQUIRES(
        ctx, in1_num_dims >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1_num_dims));

    const auto in0_last_dim = in0.dim_size(in0_num_dims - 1);
    const auto in0_prev_dim = in0.dim_size(in0_num_dims - 2);
    OP_REQUIRES(ctx, in0_last_dim == in0_prev_dim,
                errors::InvalidArgument(
                    "In[0] matrices in the last dimensions must be square (",
                    in0_last_dim, " =/= ", in0_prev_dim, ")"));
  }
};

#define REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_CPU(TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("MatrixTriangularSolve")              \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<CPUDevice, TYPE>); \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixTriangularSolve")         \
                              .Device(DEVICE_CPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<CPUDevice, TYPE>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Scalar>
struct LaunchBatchMatrixTriangularSolve<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adjoint, bool lower,
                     const MatMulBCast& bcast, Tensor* out) {
    auto* stream = context->op_device_context()->stream();

    const uint64 m = in_x.dim_size(1);
    const uint64 n = out->dim_size(2);

    //  Do a memcpy when we don't need to broadcast.
    if (!bcast.IsBroadcastingRequired() || out->shape() == in_y.shape()) {
      auto src_device_mem = AsDeviceMemory(in_y.template flat<Scalar>().data());
      auto dst_device_mem = AsDeviceMemory(out->template flat<Scalar>().data());
      OP_REQUIRES(
          context,
          stream
              ->ThenMemcpyD2D(&dst_device_mem, src_device_mem,
                              bcast.y_batch_size() * m * n * sizeof(Scalar))
              .ok(),
          errors::Internal("MatrixTriangularSolveOp: failed to copy rhs "
                           "from device"));
    } else {
      std::vector<Scalar*> out_ptrs;
      std::vector<const Scalar*> b_tmp_ptrs;
      auto* b_base_ptr = in_y.template flat<Scalar>().data();
      const std::vector<int64_t>& b_batch_indices = bcast.y_batch_indices();
      for (int64_t i = 0; i < bcast.y_batch_size(); ++i) {
        b_tmp_ptrs.push_back(b_base_ptr + i * m * n);
      }
      for (int64_t i = 0; i < bcast.output_batch_size(); ++i) {
        auto src_device_mem = AsDeviceMemory(b_tmp_ptrs[b_batch_indices[i]]);
        auto dst_device_mem =
            AsDeviceMemory(out->template flat<Scalar>().data() + i * m * n);
        OP_REQUIRES(
            context,
            stream
                ->ThenMemcpyD2D(&dst_device_mem, src_device_mem,
                                m * n * sizeof(Scalar))
                .ok(),
            errors::Internal("MatrixTriangularSolveOp: failed to copy rhs "
                             "from device"));
      }
    }

    if (out->NumElements() == 0) {
      return;
    }

#if GOOGLE_CUDA

    cublasSideMode_t side = CUBLAS_SIDE_RIGHT;
    cublasFillMode_t uplo;
    cublasOperation_t trans;
    cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

    // Cublas does
    // output = matrix \ rhs
    // where matrix, rhs and output are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // output' = rhs' / matrix' (' stands for transpose)
    // Upper/lower needs to be swapped for this.

    uplo = lower ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    trans = adjoint ? CUBLAS_OP_C : CUBLAS_OP_N;

#elif TENSORFLOW_USE_ROCM
    rocblas_side side = rocblas_side_right;
    rocblas_fill uplo;
    rocblas_operation trans;
    rocblas_diagonal diag = rocblas_diagonal_non_unit;

    // rocblas does
    // output = matrix \ rhs
    // where matrix, rhs and output are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // output' = rhs' / matrix' (' stands for transpose)
    // Upper/lower needs to be swapped for this.

    uplo = lower ? rocblas_fill_upper : rocblas_fill_lower;
    trans = adjoint ? rocblas_operation_conjugate_transpose
                    : rocblas_operation_none;

#endif

    auto solver = absl::make_unique<GpuSolver>(context);
    const uint64 leading_dim_matrix = m;
    const uint64 leading_dim_output = n;
    const uint64 colmajor_rows = n;
    const uint64 colmajor_cols = m;

    const int64_t batch_size = bcast.output_batch_size();
    std::vector<const Scalar*> a_ptrs;
    std::vector<Scalar*> out_ptrs;
    std::vector<const Scalar*> a_tmp_ptrs;
    a_ptrs.reserve(batch_size);
    out_ptrs.reserve(batch_size);
    a_tmp_ptrs.reserve(bcast.x_batch_size());
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* out_base_ptr = out->template flat<Scalar>().data();

    if (!bcast.IsBroadcastingRequired()) {
      for (int64_t i = 0; i < batch_size; ++i) {
        a_ptrs.push_back(a_base_ptr + i * m * m);
        out_ptrs.push_back(out_base_ptr + i * m * n);
      }
    } else {
      const std::vector<int64_t>& a_batch_indices = bcast.x_batch_indices();
      for (int64_t i = 0; i < bcast.x_batch_size(); ++i) {
        a_tmp_ptrs.push_back(a_base_ptr + i * m * m);
      }
      for (int64_t i = 0; i < batch_size; ++i) {
        a_ptrs.push_back(a_tmp_ptrs[a_batch_indices[i]]);
        out_ptrs.push_back(out_base_ptr + i * m * n);
      }
    }

    typedef Scalar Coefficient;
    const Scalar alpha = Scalar(1.0);

#if GOOGLE_CUDA

    // TODO(b/146763573): Consider using Trsv here when the right hand side is
    // a vector. This will require an explicit transpose since Trsv assumes
    // CUBLAS_SIDE_LEFT.
    if (batch_size == 1) {
      OP_REQUIRES_OK(
          context,
          solver->Trsm(side, uplo, trans, diag, colmajor_rows, colmajor_cols,
                       &alpha, a_ptrs[0], leading_dim_matrix /*lda*/,
                       out_ptrs[0], leading_dim_output /*ldb*/));
    } else {
      // Heuristic for choosing between batched interface vs. non-batched
      // interface. This is inspired by matrix_solve_op and can probably be
      // tuned.
      // TODO(b/146763573): Tune this heuristic.
      const int kMaxMatrixSizeToBatchSizeRatio = 128;
      const bool use_batched_solver =
          m <= kMaxMatrixSizeToBatchSizeRatio * batch_size;
      if (use_batched_solver) {
        OP_REQUIRES_OK(
            context, solver->TrsmBatched(
                         side, uplo, trans, diag, colmajor_rows, colmajor_cols,
                         &alpha, &a_ptrs[0], leading_dim_matrix /*lda*/,
                         &out_ptrs[0], leading_dim_output /*ldb*/, batch_size));
      } else {
        for (int batch = 0; batch < batch_size; ++batch) {
          OP_REQUIRES_OK(
              context, solver->Trsm(side, uplo, trans, diag, colmajor_rows,
                                    colmajor_cols, &alpha, a_ptrs[batch],
                                    leading_dim_matrix /*lda*/, out_ptrs[batch],
                                    leading_dim_output /*ldb*/));
        }
      }
    }
#elif TENSORFLOW_USE_ROCM
    for (int batch = 0; batch < batch_size; ++batch) {
      OP_REQUIRES_OK(
          context,
          solver->Trsm(side, uplo, trans, diag, colmajor_rows, colmajor_cols,
                       &alpha, a_ptrs[batch], leading_dim_matrix /*lda*/,
                       out_ptrs[batch], leading_dim_output /*ldb*/));
    }
#endif
  }
};

#define REGISTER_BATCH_MATRIX_TRIANGULAR_SOLVE_GPU(TYPE)             \
  REGISTER_KERNEL_BUILDER(Name("MatrixTriangularSolve")              \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<GPUDevice, TYPE>); \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixTriangularSolve")         \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<TYPE>("T"),            \
                          MatrixTriangularSolveOp<GPUDevice, TYPE>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_MATRIX_TRIANGULAR_SOLVE_OP_IMPL_H_
