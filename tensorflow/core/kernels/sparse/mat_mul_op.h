/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_MAT_MUL_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_MAT_MUL_OP_H_

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/SparseCore"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/kernels/sparse/transpose_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/threadpool.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

#include "tensorflow/core/kernels/sparse/mat_mul_op.h"

namespace tensorflow {

// TODO(anudhyan): These constants may be tuned based on the performance of
// 'benchmark_sparse_matrix_mat_vec_mul'. We would like to find constants
// which work across hardware platforms for typical matrix sizes. It should be
// possible to observe at least 30-50% improvement as we increase the number
// of threads by 1. If not, then it may we worth increasing kMaxShards and
// kNumShardsPerThread. However, once we have too many shards, latency may be
// dominated by per-shard overhead.
//
// Maximum number of shards into which to divide the computation for each CSR
// Sparse Matrix instance.
static constexpr int32_t kMaxShards = 20;
// Number of shards allocated to each thread.
static constexpr int32_t kNumShardsPerThread = 3;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Abstract OpKernel to compute sparse-dense matrix multiplication.
//
// Implements a kernel which, given a SparseMatrix `a` and dense Tensor `b`,
// computes a dense Tensor `c` satisfying `c = a * b` where * denotes matrix
// multiplication.
//
// The boolean attributes `transpose_a` and `adjoint_a` will transpose or
// adjoint `a` before multiplication, respectively. At most one of these
// attributes must be set to True. Corresponding attributes will transpose or
// adjoint `b` or the output (after multiplication).
//
// The rank of both `a` and `b` must be equal and their shapes must be
// compatible for matrix multiplication. Otherwise, InvalidArgument runtime
// errors will be thrown. Only rank 2 or rank 3 inputs are supported.
//
template <typename Device, typename T>
class CSRMatMulOp : public OpKernel {
 public:
  explicit CSRMatMulOp(OpKernelConstruction* c);

  ~CSRMatMulOp() override {}

  Status ValidateInputs(const CSRSparseMatrix& sparse_matrix_a,
                        const Tensor& dense_tensor_b, int* rank,
                        int64_t* batch_size);

 public:
  bool transpose_a_;
  bool transpose_b_;
  bool conjugate_a_;
  bool conjugate_b_;
  bool transpose_output_;
  bool conjugate_output_;
};

// CPU Kernel to compute sparse-dense matrix multiplication.
//
// Uses Eigen SparseMatrix to compute the sparse-dense multiplication between
// a CSR SparseMatrix `a` and dense Tensor `b`. If intra-op parallelism is
// available, the implementation parallelizes the computation across each row
// of the sparse matrix.
template <typename T>
class CSRMatMulCPUOp : public CSRMatMulOp<CPUDevice, T> {
  using SparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

 public:
  explicit CSRMatMulCPUOp(OpKernelConstruction* c)
      : CSRMatMulOp<CPUDevice, T>(c) {}

  ~CSRMatMulCPUOp() override{};

  void Compute(OpKernelContext* ctx) final;

 private:
  Status AllocateOutput(OpKernelContext* ctx, const int32_t rank,
                        const int64_t batch_size, const int64_t num_rows,
                        const int64_t num_cols, const bool transpose_output,
                        Tensor** output, Tensor* output_transposed,
                        Tensor** matmul_result);

  Eigen::Ref<const SparseMatrix> GetSparseMatrixRef(
      const CSRSparseMatrix& csr_matrix, const int batch_index,
      const int64_t row_begin, const int64_t num_shard_rows,
      std::vector<int32>* row_ptrs);

  void SparseDenseMatMulWithoutTransposedLHS(OpKernelContext* ctx,
                                             const int64_t batch_size,
                                             const int64_t num_lhs_rows,
                                             const CSRSparseMatrix& lhs,
                                             const Tensor& rhs, Tensor* output);

  void SparseDenseMatMulWithTransposedLHS(OpKernelContext* ctx,
                                          const int64_t batch_size,
                                          const int64_t num_lhs_rows,
                                          const int64_t num_lhs_cols,
                                          const CSRSparseMatrix& lhs,
                                          const Tensor& rhs, Tensor* output);

  void HandleBatchAndRowRange(
      const int64_t num_rows, const int64_t batch_and_row_begin,
      const int64_t batch_and_row_end,
      const std::function<void(int64_t, int64_t, int64_t)>& fn);

  Status TransposeAndConjugateTensor(OpKernelContext* ctx, const Tensor& input,
                                     bool conjugate, Tensor* output);

  Status TransposeAndConjugateAllocatedTensor(OpKernelContext* ctx,
                                              const Tensor& input,
                                              bool conjugate, Tensor* output);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_MAT_MUL_OP_H_
