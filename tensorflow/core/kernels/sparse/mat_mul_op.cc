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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
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
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

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
static constexpr int32 kMaxShards = 20;
// Number of shards allocated to each thread.
static constexpr int32 kNumShardsPerThread = 3;

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
  explicit CSRMatMulOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(c, c->GetAttr("transpose_b", &transpose_b_));
    bool adjoint_a;
    OP_REQUIRES_OK(c, c->GetAttr("adjoint_a", &adjoint_a));
    OP_REQUIRES(c, !(adjoint_a && transpose_a_),
                errors::InvalidArgument(
                    "Only one of adjoint_a and transpose_a may be true."));
    bool adjoint_b;
    OP_REQUIRES_OK(c, c->GetAttr("adjoint_b", &adjoint_b));
    OP_REQUIRES(c, !(adjoint_b && transpose_b_),
                errors::InvalidArgument(
                    "Only one of adjoint_b and transpose_b may be true."));
    OP_REQUIRES_OK(c, c->GetAttr("transpose_output", &transpose_output_));
    OP_REQUIRES_OK(c, c->GetAttr("conjugate_output", &conjugate_output_));
    conjugate_a_ = adjoint_a;
    conjugate_b_ = adjoint_b;
    transpose_a_ |= adjoint_a;
    transpose_b_ |= adjoint_b;
  }

  ~CSRMatMulOp() override {}

  Status ValidateInputs(const CSRSparseMatrix& sparse_matrix_a,
                        const Tensor& dense_tensor_b, int* rank,
                        int64* batch_size) {
    if (sparse_matrix_a.dtype() != dense_tensor_b.dtype()) {
      return errors::InvalidArgument(
          "Input types don't match.  a.dtype == ",
          DataTypeString(sparse_matrix_a.dtype()),
          " vs. b.dtype == ", DataTypeString(dense_tensor_b.dtype()));
    }
    *rank = sparse_matrix_a.dims();
    // TODO(ebrevdo): Add support for broadcasting matmul.
    if (*rank != dense_tensor_b.dims()) {
      return errors::InvalidArgument("Ranks of a and b must match, saw: ", rank,
                                     " vs. ", dense_tensor_b.dims(), ".");
    }
    // A valid CSR SparseMatrix has rank 2 or rank 3.
    *batch_size = (*rank == 2) ? 1 : dense_tensor_b.dim_size(0);
    if (sparse_matrix_a.batch_size() != *batch_size) {
      return errors::InvalidArgument("Batch sizes of a and b must match, saw: ",
                                     sparse_matrix_a.batch_size(), " vs. ",
                                     batch_size, ".");
    }
    const auto& a_dense_shape = sparse_matrix_a.dense_shape().vec<int64>();
    const int64 a_inner_dim =
        a_dense_shape(this->transpose_a_ ? *rank - 2 : *rank - 1);
    const int64 b_inner_dim =
        dense_tensor_b.dim_size(this->transpose_b_ ? *rank - 1 : *rank - 2);
    if (a_inner_dim != b_inner_dim) {
      return errors::InvalidArgument(
          "Inner product dimensions of A and B do not agree.  Shapes are: ",
          TensorShape(a_dense_shape), " vs. ",
          dense_tensor_b.shape().DebugString());
    }
    return Status::OK();
  }

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

  ~CSRMatMulCPUOp() override {}

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* sparse_matrix_a;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &sparse_matrix_a));
    const Tensor& matrix_b = ctx->input(1);

    int rank;
    int64 batch_size;
    OP_REQUIRES_OK(ctx, this->ValidateInputs(*sparse_matrix_a, matrix_b, &rank,
                                             &batch_size));

    const auto dense_shape = sparse_matrix_a->dense_shape().vec<int64>();
    int64 num_lhs_rows = dense_shape(rank - 2);
    int64 num_lhs_cols = dense_shape(rank - 1);
    int64 num_rhs_rows = matrix_b.dim_size(rank - 2);
    int64 num_rhs_cols = matrix_b.dim_size(rank - 1);

    if (this->transpose_a_) {
      std::swap(num_lhs_rows, num_lhs_cols);
    }

    // Possibly transpose the dense Tensor b.
    const Tensor* rhs = &matrix_b;
    Tensor b_transposed;
    if (this->transpose_b_) {
      OP_REQUIRES_OK(
          ctx, TransposeAndConjugateTensor(ctx, matrix_b, this->conjugate_b_,
                                           &b_transposed));
      rhs = &b_transposed;
      std::swap(num_rhs_rows, num_rhs_cols);
    }

    // If we're transposing the output, then allocate a temporary buffer to
    // store the output. Otherwise allocate the output directly.
    Tensor* output = nullptr;
    Tensor* matmul_result = nullptr;
    Tensor output_transposed;
    OP_REQUIRES_OK(
        ctx, AllocateOutput(ctx, rank, batch_size, num_lhs_rows, num_rhs_cols,
                            this->transpose_output_, &output,
                            &output_transposed, &matmul_result));

    if (!this->transpose_a_) {
      SparseDenseMatMulWithoutTransposedLHS(
          ctx, batch_size, num_lhs_rows, *sparse_matrix_a, *rhs, matmul_result);
    } else {  // transpose_a_ == true
      SparseDenseMatMulWithTransposedLHS(ctx, batch_size, num_lhs_rows,
                                         num_lhs_cols, *sparse_matrix_a, *rhs,
                                         matmul_result);
    }

    // Transpose (and conjugate) the output if necessary.
    // Note that conjugate is only true if transpose is also true.
    if (this->transpose_output_) {
      OP_REQUIRES_OK(
          ctx, TransposeAndConjugateAllocatedTensor(
                   ctx, output_transposed, this->conjugate_output_, output));
    } else if (this->conjugate_output_) {
      functor::maybe_conj_inplace<CPUDevice, T>::run(
          ctx->eigen_device<CPUDevice>(), output);
    }
  }

 private:
  // Allocates the output with the appropriate shape. Additionally, if
  // transpose_output is True, allocates a temporary buffer with the transposed
  // output. 'matmul_result' points to either output or output_transposed, based
  // on whether transpose_output is True.
  Status AllocateOutput(OpKernelContext* ctx, const int32 rank,
                        const int64 batch_size, const int64 num_rows,
                        const int64 num_cols, const bool transpose_output,
                        Tensor** output, Tensor* output_transposed,
                        Tensor** matmul_result) {
    TensorShape output_shape;
    if (rank == 3) output_shape.AddDim(batch_size);

    if (!transpose_output) {
      output_shape.AppendShape({num_rows, num_cols});
      TF_RETURN_IF_ERROR(ctx->allocate_output(0, output_shape, output));
      *matmul_result = *output;
    } else {
      TensorShape output_transposed_shape = output_shape;
      output_transposed_shape.AppendShape({num_rows, num_cols});
      output_shape.AppendShape({num_cols, num_rows});
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<T>::value,
                                            output_transposed_shape,
                                            output_transposed));
      TF_RETURN_IF_ERROR(ctx->allocate_output(0, output_shape, output));
      *matmul_result = output_transposed;
    }
    return Status::OK();
  }

  // Returns an Eigen::Ref expression of a sparse sub-matrix from the given
  // contiguous segment of rows of the CSR Sparse Matrix.
  Eigen::Ref<const SparseMatrix> GetSparseMatrixRef(
      const CSRSparseMatrix& csr_matrix, const int batch_index,
      const int64 row_begin, const int64 num_shard_rows,
      std::vector<int32>* row_ptrs) {
    // Compute the row pointers of the sparse sub-matrix.
    row_ptrs->resize(num_shard_rows + 1);
    const int64 row_offset =
        csr_matrix.row_pointers_vec(batch_index)(row_begin);
    for (int64 row_idx = 0; row_idx <= num_shard_rows; ++row_idx) {
      row_ptrs->at(row_idx) =
          csr_matrix.row_pointers_vec(batch_index)(row_begin + row_idx) -
          row_offset;
    }
    const int64 num_cols =
        csr_matrix.dense_shape().vec<int64>()(csr_matrix.dims() - 1);
    return Eigen::Map<const SparseMatrix>(
        num_shard_rows /* num_rows */, num_cols /* num_cols */,
        row_ptrs->at(num_shard_rows) /* total_nnz */, row_ptrs->data(),
        csr_matrix.col_indices_vec(batch_index).data() + row_offset,
        csr_matrix.values_vec<T>(batch_index).data() + row_offset);
  }

  // Sparse-Dense Matrix Multiplication between a CSRSparseMatrix (LHS) and a
  // dense Tensor (RHS).
  void SparseDenseMatMulWithoutTransposedLHS(
      OpKernelContext* ctx, const int64 batch_size, const int64 num_lhs_rows,
      const CSRSparseMatrix& lhs, const Tensor& rhs, Tensor* output) {
    // Parallelize matrix multiplication across batch dimensions and across
    // rows in each batch.
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int32 num_threads = worker_threads.num_threads;
    const int64 block_size =
        num_lhs_rows / std::max(kMaxShards, kNumShardsPerThread * num_threads);
    const int64 num_rhs_rows = rhs.dim_size(rhs.dims() - 2);
    const int64 num_rhs_cols = rhs.dim_size(rhs.dims() - 1);
    worker_threads.workers->ParallelFor(
        batch_size * num_lhs_rows /* total */,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::
                kFixedBlockSize /* strategy */,
            absl::nullopt /* cost_per_unit */, block_size),
        [&](int64 batch_and_row_begin, int64 batch_and_row_end) {
          HandleBatchAndRowRange(
              num_lhs_rows, batch_and_row_begin, batch_and_row_end,
              [&](int64 batch_idx, int64 row_begin, int64 row_end) {
                const int64 num_shard_rows = row_end - row_begin;

                // Define an Eigen::SparseMatrix over the row range:
                // [row_begin, row_end) of the CSR SparseMatrix A.
                std::vector<int32> row_ptrs;
                auto sparse_matrix = GetSparseMatrixRef(
                    lhs, batch_idx, row_begin, num_shard_rows, &row_ptrs);

                // Map the corresponding rows of the rhs.
                ConstMatrixMap rhs_map(rhs.flat<T>().data() + batch_idx *
                                                                  num_rhs_rows *
                                                                  num_rhs_cols,
                                       num_rhs_rows, num_rhs_cols);

                // Write to the corresponding rows of the output matrix.
                MatrixMap output_map(
                    output->flat<T>().data() +
                        batch_idx * num_lhs_rows * num_rhs_cols +
                        row_begin * num_rhs_cols,
                    num_shard_rows, num_rhs_cols);
                output_map.noalias() = sparse_matrix * rhs_map;
              });
        });
  }

  // Sparse-Dense Matrix Multiplication assuming the CSRSparseMatrix (LHS) is
  // to be transposed before the operation.
  void SparseDenseMatMulWithTransposedLHS(OpKernelContext* ctx,
                                          const int64 batch_size,
                                          const int64 num_lhs_rows,
                                          const int64 num_lhs_cols,
                                          const CSRSparseMatrix& lhs,
                                          const Tensor& rhs, Tensor* output) {
    auto device = ctx->eigen_device<CPUDevice>();
    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int32 num_threads = worker_threads.num_threads;
    const int64 num_rhs_rows = rhs.dim_size(rhs.dims() - 2);
    const int64 num_rhs_cols = rhs.dim_size(rhs.dims() - 1);
    // Usually, we want to avoid transposing the sparse matrix A since it may be
    // an expensive operation. Instead, we use the identity (A^T B) = (B^T A)^T.
    // We don't actually transpose B or the output because it is more convenient
    // to have them in column major form.
    //
    // However, if A is hypersparse and B and C are huge, transposing A will be
    // cheaper. In the future, we should have a cost model estimating the cost
    // of transposing all matrices (A, B, C) to decide which variant to use.

    // Each thread writes to its own copy of the matrix product. These
    // `num_threads` copies are summed together to obtain the final result.
    Tensor matmul_result_buffer;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                           TensorShape({num_threads + 1,
                                                        output->NumElements()}),
                                           &matmul_result_buffer));
    functor::SetZeroFunctor<CPUDevice, T> set_zero;
    set_zero(device, matmul_result_buffer.flat<T>());

    // Parallelize matrix multiplication across batch dimensions and across
    // columns of A^T in each batch. These correspond to rows of A.
    const int64 block_size =
        num_lhs_cols / std::max(kMaxShards, kNumShardsPerThread * num_threads);
    worker_threads.workers->ParallelForWithWorkerId(
        batch_size * num_lhs_cols /* total */,
        thread::ThreadPool::SchedulingParams(
            thread::ThreadPool::SchedulingStrategy::
                kFixedBlockSize /* strategy */,
            absl::nullopt /* cost_per_unit */, block_size),
        [&](int64 batch_and_row_begin, int64 batch_and_row_end, int tid) {
          HandleBatchAndRowRange(
              num_lhs_cols, batch_and_row_begin, batch_and_row_end,
              [&](int64 batch_idx, int64 row_begin, int64 row_end) {
                const int64 num_shard_rows = row_end - row_begin;

                // Define a new sparse sub-matrix from the row range
                // [row_begin, row_end) of the sparse matrix A.
                std::vector<int32> row_ptrs;
                auto sparse_matrix = GetSparseMatrixRef(
                    lhs, batch_idx, row_begin, num_shard_rows, &row_ptrs);

                // Map the corresponding `num_shard_rows` columns of B^T.
                // This is the same as taking the `num_shard_rows` rows of B.
                ConstMatrixMap b_dense_map(
                    rhs.flat<T>().data() +
                        batch_idx * num_rhs_rows * num_rhs_cols +
                        row_begin * num_rhs_cols,
                    num_shard_rows, num_rhs_cols);

                // Map to the corresponding rows of the output.
                MatrixMap output_map(
                    matmul_result_buffer.flat<T>().data() +
                        tid * batch_size * num_lhs_rows * num_rhs_cols +
                        batch_idx * num_lhs_rows * num_rhs_cols,
                    num_lhs_rows, num_rhs_cols);

                // Compute the product C^T = B^T * A; restricted to the row
                // range in the current shard.
                if (this->conjugate_a_) {
                  output_map.transpose().noalias() +=
                      b_dense_map.transpose() * sparse_matrix.conjugate();
                } else {
                  output_map.transpose().noalias() +=
                      b_dense_map.transpose() * sparse_matrix;
                }
              });
        });

    // Sum across each thread's matmul result.
    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;
    output->flat<T>().device(device) = matmul_result_buffer.matrix<T>().reduce(
        Eigen::array<Index, 1>({0}), Reducer());
  }

  // Given a range [batch_and_row_begin, batch_and_row_end) which is a
  // contiguous subset of [0, num_rows * batch_size), calls the function
  // fn(batch_idx, row_begin, row_end) for each batch index
  // and the row range [row_begin, row_end) contained in the batch.
  void HandleBatchAndRowRange(
      const int64 num_rows, const int64 batch_and_row_begin,
      const int64 batch_and_row_end,
      const std::function<void(int64, int64, int64)>& fn) {
    // Obtain the batch indices overlapping with the current shard.
    const int64 batch_begin = batch_and_row_begin / num_rows;
    const int64 batch_end_inclusive = batch_and_row_end / num_rows;

    for (int64 batch_idx = batch_begin; batch_idx <= batch_end_inclusive;
         ++batch_idx) {
      // Find the contiguous set of rows which are contained in this shard as
      // well as the current batch. We intersect with interval [batch_idx *
      // num_rows, (batch_idx + 1) * num_rows) which denotes the current batch.
      const int64 current_batch_row_begin =
          std::max(batch_and_row_begin, batch_idx * num_rows);
      const int64 current_batch_row_end =
          std::min(batch_and_row_end, (batch_idx + 1) * num_rows);

      const int64 row_begin = current_batch_row_begin % num_rows;
      const int64 num_shard_rows =
          current_batch_row_end - current_batch_row_begin;
      // Edge case for when current_batch_row_end is the first index of a new
      // row.
      if (num_shard_rows == 0) continue;

      fn(batch_idx, row_begin, row_begin + num_shard_rows);
    }
  }

  // Transposes (and optionally, conjugates) a given Tensor. Also allocates the
  // required memory for the output Tensor.
  Status TransposeAndConjugateTensor(OpKernelContext* ctx, const Tensor& input,
                                     bool conjugate, Tensor* output) {
    TensorShape transposed_shape = input.shape();
    transposed_shape.set_dim(input.dims() - 1,
                             input.dim_size(input.dims() - 2));
    transposed_shape.set_dim(input.dims() - 2,
                             input.dim_size(input.dims() - 1));
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, transposed_shape, output));
    return TransposeAndConjugateAllocatedTensor(ctx, input, conjugate, output);
  }

  // Transposes (and optionally, conjugates) a given Tensor. The output should
  // be already allocated.
  Status TransposeAndConjugateAllocatedTensor(OpKernelContext* ctx,
                                              const Tensor& input,
                                              bool conjugate, Tensor* output) {
    if (conjugate) {
      TF_RETURN_IF_ERROR(DoConjugateMatrixTranspose(
          ctx->eigen_device<CPUDevice>(), input, output));
    } else {
      TF_RETURN_IF_ERROR(
          DoMatrixTranspose(ctx->eigen_device<CPUDevice>(), input, output));
    }
    return Status::OK();
  }
};

// GPU Kernel to compute sparse-dense matrix multiplication.
template <typename T>
class CSRMatMulGPUOp : public CSRMatMulOp<GPUDevice, T> {
  using SparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

 public:
  explicit CSRMatMulGPUOp(OpKernelConstruction* c)
      : CSRMatMulOp<GPUDevice, T>(c) {}

  ~CSRMatMulGPUOp() override {}

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* a_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    const Tensor& b_t = ctx->input(1);

    int rank;
    int64 batch_size;
    OP_REQUIRES_OK(ctx,
                   this->ValidateInputs(*a_matrix, b_t, &rank, &batch_size));

    const Tensor& a_dense_shape_t = a_matrix->dense_shape();
    TensorShape a_dense_tensor_shape;
    auto a_dense_shape = a_dense_shape_t.vec<int64>();
    OP_REQUIRES_OK(
        ctx, TensorShapeUtils::MakeShape(a_dense_shape, &a_dense_tensor_shape));

    const int row_dim = (rank == 2) ? 0 : 1;
    const int64 a_outer_dim = a_dense_tensor_shape.dim_size(
        this->transpose_a_ ? row_dim + 1 : row_dim);
    const int64 b_inner_dim =
        b_t.shape().dim_size(this->transpose_b_ ? row_dim + 1 : row_dim);
    const int64 b_outer_dim =
        b_t.dim_size(this->transpose_b_ ? row_dim : row_dim + 1);
    const int64 b_slice_size = b_inner_dim * b_outer_dim;

    TensorShape c_shape;
    if (rank == 3) c_shape.AddDim(batch_size);
    if (this->transpose_output_) {
      c_shape.AddDim(b_outer_dim);
      c_shape.AddDim(a_outer_dim);
    } else {
      c_shape.AddDim(a_outer_dim);
      c_shape.AddDim(b_outer_dim);
    }

    const int64 c_matrix_lhs = c_shape.dim_size(row_dim);
    const int64 c_matrix_rhs = c_shape.dim_size(row_dim + 1);
    const int64 c_slice_size = c_matrix_lhs * c_matrix_rhs;
    Tensor* c_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &c_t));

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    bool shortcut_ok = (b_outer_dim == 1);
#if TENSORFLOW_USE_ROCM
    // ROCm hipsparse does not implement csrmv with transposed input a
    shortcut_ok = shortcut_ok && !this->transpose_a_;
#endif      
    if (shortcut_ok) {
      // Call matrix-vector multiply if b is a vector.
      TTypes<int64>::ConstVec a_dense_shape_comp(a_dense_shape.data() + row_dim,
                                                 2);
      Tensor b_conj_t;
      const T* b_base_ptr = b_t.template flat<T>().data();
      bool conjugate_a = this->conjugate_a_;
      bool conjugate_output = this->conjugate_output_;
      if (this->conjugate_b_) {
        if (conjugate_a) {
          // In this case we can use the identity
          //   conj(a) * conj(b) = conj(a * b)
          // instead of creating a conjugated copy of b.
          conjugate_a = false;
          conjugate_output = !conjugate_output;
        } else {
          OP_REQUIRES_OK(
              ctx, ctx->forward_input_or_allocate_temp(
                       {1}, DataTypeToEnum<T>::value, b_t.shape(), &b_conj_t));
          functor::maybe_conj<GPUDevice, T>::run(d, b_t, &b_conj_t);
          b_base_ptr = b_conj_t.template flat<T>().data();
        }
      }

      functor::CSRSparseMatrixMatVec<GPUDevice, T> csr_spmv(this->transpose_a_,
                                                            conjugate_a);
      for (int i = 0; i < batch_size; ++i) {
        auto a_row_ptr = a_matrix->row_pointers_vec(i);
        auto a_col_ind = a_matrix->col_indices_vec(i);
        auto a_values = a_matrix->values_vec<T>(i);
        ConstCSRComponent<T> a_comp{a_row_ptr, a_col_ind, a_values,
                                    a_dense_shape_comp};
        const T* b_i = b_base_ptr + i * b_slice_size;
        T* c_i = &c_t->template flat<T>()(i * c_slice_size);
        Status s = csr_spmv.Compute(ctx, a_comp, b_i, c_i);
        OP_REQUIRES_OK(ctx, s);
      }
      if (conjugate_output) {
        functor::maybe_conj_inplace<GPUDevice, T>::run(d, c_t);
      }
      return;
    }

    functor::CSRSparseMatrixMatMul<GPUDevice, T> csr_spmmadd(
        this->transpose_output_);

    Tensor c_mat_col_major_t;
    if (!this->transpose_output_) {
      // If transpose_output is false, we'll need to transpose the (col
      // major) output of the csrgemm call to get proper (row-major)
      // output.  Which means we need to keep a temporary buffer to
      // store the intermediate gemm output.
      TensorShape c_mat_col_major_shape;
      if (rank == 2) {
        c_mat_col_major_shape = TensorShape({c_matrix_rhs, c_matrix_lhs});
      } else {
        c_mat_col_major_shape =
            TensorShape({batch_size, c_matrix_rhs, c_matrix_lhs});
      }
      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                  c_mat_col_major_shape, &c_mat_col_major_t));
    }

    // If transpose_output is true, return the direct (column-major i.e.,
    // transposed) output of the csrgemm call.  Otherwise we'll need
    // to transpose it to row major format.
    auto c_mat_col_major = (this->transpose_output_)
                               ? c_t->flat<T>()
                               : c_mat_col_major_t.flat<T>();

    // Possibly transpose a.
    const CSRSparseMatrix* a_input_matrix;
    // If we need to transpose a, we will store the result temporarily
    // in the object below.
    CSRSparseMatrix a_matrix_transposed;
    if (!this->transpose_a_) {
      a_input_matrix = a_matrix;
    } else {
      functor::CSRSparseMatrixTranspose<GPUDevice, T> transpose;
      OP_REQUIRES_OK(ctx, transpose(ctx, this->conjugate_a_, *a_matrix,
                                    &a_matrix_transposed));
      a_input_matrix = &a_matrix_transposed;
    }

    auto a_input_dense_shape = a_input_matrix->dense_shape().vec<int64>();

    // Possibly transpose b.
    Tensor b_t_input;
    if (!this->transpose_b_) {
      b_t_input = b_t;
    } else {
      TensorShape b_t_transposed_shape;
      if (rank == 3) {
        b_t_transposed_shape.AddDim(batch_size);
      }
      b_t_transposed_shape.AddDim(b_t.dim_size(row_dim + 1));
      b_t_transposed_shape.AddDim(b_t.dim_size(row_dim));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             b_t_transposed_shape, &b_t_input));
      const GPUDevice& d = ctx->eigen_device<GPUDevice>();
      if (this->conjugate_b_) {
        OP_REQUIRES_OK(ctx, DoConjugateMatrixTranspose(d, b_t /*input*/,
                                                       &b_t_input /*output*/));
      } else {
        OP_REQUIRES_OK(
            ctx, DoMatrixTranspose(d, b_t /*input*/, &b_t_input /*output*/));
      }
    }

    // Dense shape of a batch component of A.
    TTypes<int64>::ConstVec a_input_dense_shape_comp(
        a_input_dense_shape.data() + row_dim, 2);

    auto b = b_t_input.flat<T>();

    for (int i = 0; i < batch_size; ++i) {
      auto a_row_ptr = a_input_matrix->row_pointers_vec(i);
      auto a_col_ind = a_input_matrix->col_indices_vec(i);
      auto a_values = a_input_matrix->values_vec<T>(i);
      typename TTypes<T>::UnalignedConstMatrix b_i(b.data() + i * b_slice_size,
                                                   {b_inner_dim, b_outer_dim});
      typename TTypes<T>::UnalignedMatrix c_mat_col_major_i(
          c_mat_col_major.data() + i * c_slice_size,
          {c_matrix_lhs, c_matrix_rhs});
      ConstCSRComponent<T> a_comp{a_row_ptr, a_col_ind, a_values,
                                  a_input_dense_shape_comp};
      Status s = csr_spmmadd.Compute(ctx, a_comp, b_i, c_mat_col_major_i);
      OP_REQUIRES_OK(ctx, s);
    }

    if (!this->transpose_output_) {
      // We need to return values in row major format, so transpose
      // the column-major values in c_mat_col_major_t to row-major output c_t.
      OP_REQUIRES_OK(ctx, DoMatrixTranspose(d, /*input=*/c_mat_col_major_t,
                                            /*output=*/c_t));
    }
    if (this->conjugate_output_) {
      functor::maybe_conj_inplace<GPUDevice, T>::run(d, c_t);
    }
  }
};

#define REGISTER_CPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseMatrixMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CSRMatMulCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("SparseMatrixMatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CSRMatMulGPUOp<T>);

REGISTER_GPU(float)
REGISTER_GPU(double)
#if GOOGLE_CUDA
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)
#endif

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T>
class CSRSparseMatrixMatMul<GPUDevice, T> {
 public:
  explicit CSRSparseMatrixMatMul(const bool transpose_output)
      : transpose_output_(transpose_output) {}

  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 typename TTypes<T>::UnalignedConstMatrix b,
                 typename TTypes<T>::UnalignedMatrix c) {
    GpuSparse cuda_sparse(ctx);
    TF_RETURN_IF_ERROR(cuda_sparse.Initialize());
    {
      // Use Csrmm to calculate:
      //   C = alpha * op(A) * op(B) + beta * C
      // where alpha = 1.0, beta = 0.0, A is sparse and B and C are dense.
      // Note that Csrmm assumes B and C are in column-major form; so we
      // use transB == true, and manually transpose the output in place
      // using blas<t>geam.
      // TODO(ebrevdo,rmlarsen): Add support for transposition and adjoint.

      // Create alpha and beta scalars; alpha = 1.0, beta = 0.0
      // TODO(ebrevdo,rmlarsen): Add support for non-trivial alpha and beta.
      const T alpha = 1;
      const T beta = 0;

      // transA must be non-transpose if transB is transpose (cusparse
      // limitation).
#if GOOGLE_CUDA
      const gpusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
#elif TENSORFLOW_USE_ROCM
      const gpusparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
#endif

      // transB: b is row-major, and cusparse requires col-major b (or
      // equivalently transB == transpose).  this version is actually more
      // efficient.
#if GOOGLE_CUDA
      const gpusparseOperation_t transB = CUSPARSE_OPERATION_TRANSPOSE;

      gpusparseMatDescr_t descrA;
      TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
#elif TENSORFLOW_USE_ROCM
      const gpusparseOperation_t transB = HIPSPARSE_OPERATION_TRANSPOSE;

      gpusparseMatDescr_t descrA;
      TF_RETURN_IF_GPUSPARSE_ERROR(hipsparseCreateMatDescr(&descrA));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO));
#endif

      // A is (m, k), Bt is (ldb, k) and Ct is (ldc, n)
      const int k = b.dimension(0);
      DCHECK_EQ(k, a.dense_shape_host(1));

      // If transpose_output_ is true, then the c matrix we receive
      // here is the direct row major output (into which we will store
      // csrgemm's col major output).  Otherwise it's a
      // temporary tensor that will store the column major output that
      // will eventually be transposed.
      const int m = c.dimension(transpose_output_ ? 1 : 0);
      const int n = c.dimension(transpose_output_ ? 0 : 1);
      DCHECK_EQ(m, a.dense_shape_host(0));
      DCHECK_EQ(n, b.dimension(1));
      const int nnz = a.values.size();
      DCHECK_EQ(nnz, a.col_ind.size());

      // ldb: leading dimension of B. If op(B)=B, it must be at least max(1, k)
      // if op(A) = A and at least max (1, m) otherwise. If op(B) != B, it must
      // be at least max(1, n).
      const int ldb = n;
      // ldc: leading dimension of C. It must be at least max(1, m) if
      // op(A) = A and at least max(1, k) otherwise.
      const int ldc = m;

      TF_RETURN_IF_ERROR(
          cuda_sparse.Csrmm(transA, transB, m, n, k, nnz, &alpha, descrA,
                            a.values.data(), a.row_ptr.data(), a.col_ind.data(),
                            b.data(), ldb, &beta, c.data(), ldc));
    }

    return Status::OK();
  }

 private:
  bool transpose_output_;
};

template <typename T>
class CSRSparseMatrixMatVec<GPUDevice, T> {
 public:
  CSRSparseMatrixMatVec(bool transpose_a, bool conjugate_a)
      : transA_(TransposeAndConjugateToGpuSparseOp(transpose_a, conjugate_a,
                                                   &status_)) {}

  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 const T* x, T* y) {
    TF_RETURN_IF_ERROR(status_);
    GpuSparse cuda_sparse(ctx);
    TF_RETURN_IF_ERROR(cuda_sparse.Initialize());
    {
      // Use Csrmv to calculate:
      //   y = alpha * op(A) * x + beta * y
      // where alpha = 1.0, beta = 0.0, A is a sparse matrix and x and y are
      // dense vectors.

      // Create alpha and beta scalars; alpha = 1.0, beta = 0.0
      // TODO(rmlarsen,ebrevdo): Add support for general alpha, beta.
      const T alpha = 1;
      const T beta = 0;

      gpusparseMatDescr_t descrA;
#if GOOGLE_CUDA
      TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
#elif TENSORFLOW_USE_ROCM
      TF_RETURN_IF_GPUSPARSE_ERROR(hipsparseCreateMatDescr(&descrA));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_GPUSPARSE_ERROR(
          hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

      const int m = a.dense_shape_host(0);
      const int n = a.dense_shape_host(1);
      const int nnz = a.values.size();
      DCHECK_EQ(nnz, a.col_ind.size());
      TF_RETURN_IF_ERROR(cuda_sparse.Csrmv(transA_, m, n, nnz, &alpha, descrA,
                                           a.values.data(), a.row_ptr.data(),
                                           a.col_ind.data(), x, &beta, y));
    }

    return Status::OK();
  }

 private:
  Status status_;
  const gpusparseOperation_t transA_;
};

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
