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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include <memory>
#include <numeric>

#include "third_party/eigen3/Eigen/SparseCore"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

// Swaps the dim sizes at two given dimensions of a TensorShape.
// Callers are responsible for making sure the given dimensions are within the
// valid dimension range of the TensorShape.
void SwapDimSizes(const int dim_a, const int dim_b, TensorShape* shape) {
  const int64 size_a = shape->dim_size(dim_a);
  const int64 size_b = shape->dim_size(dim_b);
  shape->set_dim(dim_a, size_b);
  shape->set_dim(dim_b, size_a);
}

}  // namespace

// Op to compute the matrix multiplication of two CSR Sparse Matrices.
//
// Implements a CPU kernel to perform matrix multiplication using Eigen
// SparseMatrix and its Sparse-Sparse matmul. Supports transposing and
// adjointing on the fly for both the inputs without actually constructing the
// transpose or adjoint.
//
// This implementation does not support broadcasting. Hence both the input
// CSRSparseMatrices must have the same rank. (Either rank 2 or rank 3).
//
// The output sparse have numeric (non-structural) zeros.
// TODO(anudhyan): Consider exposing whether to prune zeros as an attribute in
// the op's interface.
//
// If multiple threads are available, we parallelize across multiple batches
// using Eigen ThreadPool. Within a single batch, we run in single threaded mode
// because Eigen's Sparse-Sparse matmul doesn't support multithreading.
//
// TODO(b/126472741): Due to the multiple batches of a 3D CSRSparseMatrix being
// laid out in contiguous memory, this implementation allocates memory to store
// a temporary copy of the matrix product. Consequently, it uses roughly twice
// the amount of memory that it needs to. This may cause a memory blowup for
// sparse matrices with a high number of non-zero elements.
template <typename T>
class CSRSparseMatMulCPUOp : public OpKernel {
  using SparseMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor>;

 public:
  explicit CSRSparseMatMulCPUOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(c, c->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(c, c->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES(c, !(adjoint_a_ && transpose_a_),
                errors::InvalidArgument(
                    "Only one of adjoint_a and transpose_a may be true."));
    OP_REQUIRES_OK(c, c->GetAttr("adjoint_b", &adjoint_b_));
    OP_REQUIRES(c, !(adjoint_b_ && transpose_b_),
                errors::InvalidArgument(
                    "Only one of adjoint_b and transpose_b may be true."));
  }

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* input_matrix_a;
    const CSRSparseMatrix* input_matrix_b;
    // TODO(anudhyan): Factor out common validation logic in CPU and GPU Ops
    // into a common base class.
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix_a));
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 1, &input_matrix_b));
    OP_REQUIRES(ctx, input_matrix_a->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of a is not equal to 'type': ",
                    DataTypeString(input_matrix_a->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(ctx, input_matrix_b->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of b is not equal to 'type': ",
                    DataTypeString(input_matrix_b->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(ctx,
                input_matrix_a->batch_size() == input_matrix_b->batch_size(),
                errors::InvalidArgument(
                    "Batch sizes of A and B do not agree.  Batch sizes are: ",
                    input_matrix_a->batch_size(), " vs. ",
                    input_matrix_b->batch_size()));

    // Validate input_matrix_a's and input_matrix_b's shapes
    TensorShape a_shape;
    TensorShape b_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       input_matrix_a->dense_shape().vec<int64>(), &a_shape));
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(
                       input_matrix_b->dense_shape().vec<int64>(), &b_shape));

    const int rank = a_shape.dims();
    const int row_dim = (rank == 2) ? 0 : 1;
    if (transpose_a_ || adjoint_a_)
      SwapDimSizes(row_dim, row_dim + 1, &a_shape);
    if (transpose_b_ || adjoint_b_)
      SwapDimSizes(row_dim, row_dim + 1, &b_shape);

    OP_REQUIRES(
        ctx, a_shape.dim_size(row_dim + 1) == b_shape.dim_size(row_dim),
        errors::InvalidArgument(
            "Inner product dimensions of A and B do not agree.  Shapes are: ",
            a_shape.DebugString(), " vs. ", b_shape.DebugString()));

    // Infer the output shape of the matrix product.
    // TODO(ebrevdo): MatMul support for broadcasting at least in the
    // batch dimension.
    const int batch_size = input_matrix_a->batch_size();
    Tensor output_shape(cpu_allocator(), DT_INT64, TensorShape({rank}));
    auto output_shape_vec = output_shape.vec<int64>();
    if (rank == 3) output_shape_vec(0) = batch_size;
    output_shape_vec(row_dim) = a_shape.dim_size(row_dim);
    output_shape_vec(row_dim + 1) = b_shape.dim_size(row_dim + 1);

    // Set batch pointers.
    Tensor batch_ptr(cpu_allocator(), DT_INT32, TensorShape({batch_size + 1}));
    auto batch_ptr_vec = batch_ptr.vec<int32>();
    batch_ptr_vec(0) = 0;

    // Store intermediate matrix products for each batch.
    // TODO(b/126472741): For a single batch, consider reusing the
    // SparseMatrices' buffers to construct the CSRSparseMatrix to prevent 2x
    // memory usage.
    std::vector<SparseMatrix> output_matrices(batch_size);

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    // Estimate the cost per batch per as num_output_rows times the product of
    // average number of nonzeros per row.
    const int64 num_output_rows = output_shape_vec(row_dim);
    const double avg_nnz_per_row_a =
        input_matrix_a->total_nnz() /
        static_cast<double>(a_shape.dim_size(row_dim) * batch_size);
    const double avg_nnz_per_row_b =
        input_matrix_b->total_nnz() /
        static_cast<double>(b_shape.dim_size(row_dim) * batch_size);
    const int64 matmul_cost_per_batch =
        num_output_rows * (avg_nnz_per_row_a * avg_nnz_per_row_b);

    // Parallelize matrix multiplication across batches.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          matmul_cost_per_batch, [&](int64 batch_begin, int64 batch_end) {
            for (int64 batch_idx = batch_begin; batch_idx < batch_end;
                 ++batch_idx) {
              // For each batch, map the CSRSparseMatrix as Eigen SparseMatrix
              // without copying the underlying data.
              auto a_ref = GetSparseMatrixRef(*input_matrix_a, rank, batch_idx,
                                              transpose_a_, adjoint_a_);
              auto b_ref = GetSparseMatrixRef(*input_matrix_b, rank, batch_idx,
                                              transpose_b_, adjoint_b_);

              // Matrix multiply while *not* pruning numerical zeros on the fly.
              // Allocates output SparseMatrix and moves it to our list of
              // output_matrices.
              output_matrices[batch_idx] = a_ref * b_ref;

              // For now, batch_ptr contains the number of nonzeros in each
              // batch.
              batch_ptr_vec(batch_idx + 1) =
                  output_matrices[batch_idx].nonZeros();
            }
          });

    // Compute the cumulative sum to obtain the batch pointers.
    std::partial_sum(batch_ptr_vec.data(),
                     batch_ptr_vec.data() + batch_size + 1,
                     batch_ptr_vec.data());
    const int64 total_nnz = batch_ptr_vec(batch_size);

    // Allocate output tensors.
    Tensor output_row_ptr(cpu_allocator(), DT_INT32,
                          TensorShape({(num_output_rows + 1) * batch_size}));
    Tensor output_col_ind(cpu_allocator(), DT_INT32, TensorShape({total_nnz}));
    Tensor output_values(cpu_allocator(), DataTypeToEnum<T>::value,
                         TensorShape({total_nnz}));
    auto output_row_ptr_ptr = output_row_ptr.flat<int32>().data();
    auto output_col_ind_ptr = output_col_ind.flat<int32>().data();
    auto output_values_ptr = output_values.flat<T>().data();

    // Copy the output matrices from each batch into the CSRSparseMatrix
    // tensors.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          (3 * total_nnz) / batch_size /* cost per unit */,
          [&](int64 batch_begin, int64 batch_end) {
            for (int64 batch_idx = batch_begin; batch_idx < batch_end;
                 ++batch_idx) {
              const SparseMatrix& output_matrix = output_matrices[batch_idx];
              const int64 nnz = output_matrix.nonZeros();
              std::copy(output_matrix.outerIndexPtr(),
                        output_matrix.outerIndexPtr() + num_output_rows + 1,
                        output_row_ptr_ptr + batch_idx * (num_output_rows + 1));
              std::copy(output_matrix.innerIndexPtr(),
                        output_matrix.innerIndexPtr() + nnz,
                        output_col_ind_ptr + batch_ptr_vec(batch_idx));
              std::copy(output_matrix.valuePtr(),
                        output_matrix.valuePtr() + nnz,
                        output_values_ptr + batch_ptr_vec(batch_idx));
            }
          });

    // Create the CSRSparseMatrix object from its component Tensors and prepare
    // the Variant output Tensor.
    CSRSparseMatrix output_csr_matrix;
    OP_REQUIRES_OK(ctx, CSRSparseMatrix::CreateCSRSparseMatrix(
                            DataTypeToEnum<T>::value, output_shape, batch_ptr,
                            output_row_ptr, output_col_ind, output_values,
                            &output_csr_matrix));
    Tensor* output_csr_matrix_tensor;
    AllocatorAttributes cpu_alloc;
    cpu_alloc.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_csr_matrix_tensor,
                                  cpu_alloc));
    output_csr_matrix_tensor->scalar<Variant>()() =
        std::move(output_csr_matrix);
  }

 private:
  // Returns an Eigen::Ref expression of a SparseMatrix; which points to the
  // underlying memory of the given CSRSparseMatrix.
  Eigen::Ref<const SparseMatrix> GetSparseMatrixRef(
      const CSRSparseMatrix& csr_matrix, const int rank, const int batch_index,
      const bool transpose, const bool adjoint) {
    const auto dense_shape = csr_matrix.dense_shape().vec<int64>();
    const int64 num_rows = dense_shape(rank == 2 ? 0 : 1);
    const int64 num_cols = dense_shape(rank == 2 ? 1 : 2);

    Eigen::Map<const SparseMatrix> sparse_matrix(
        num_rows, num_cols, csr_matrix.nnz(batch_index),
        csr_matrix.row_pointers_vec(batch_index).data(),
        csr_matrix.col_indices_vec(batch_index).data(),
        csr_matrix.values_vec<T>(batch_index).data());

    // The transpose/adjoint expressions are not actually evaluated until
    // necessary. Hence we don't create copies or modify the input matrix
    // inplace.
    if (transpose) return sparse_matrix.transpose();
    if (adjoint) return sparse_matrix.adjoint();
    return sparse_matrix;
  }

  bool transpose_a_;
  bool transpose_b_;
  bool adjoint_a_;
  bool adjoint_b_;
};

template <typename Device, typename T>
class CSRSparseMatMulGPUOp : public OpKernel {
 public:
  explicit CSRSparseMatMulGPUOp(OpKernelConstruction* c) : OpKernel(c) {
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
    conjugate_a_ = adjoint_a;
    conjugate_b_ = adjoint_b;
    transpose_a_ = transpose_a_ || adjoint_a;
    transpose_b_ = transpose_b_ || adjoint_b;
  }

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* a_matrix;
    const CSRSparseMatrix* b_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 1, &b_matrix));
    OP_REQUIRES(
        ctx, a_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of a is not equal to 'type': ",
                                DataTypeString(a_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(
        ctx, b_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of b is not equal to 'type': ",
                                DataTypeString(b_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    // TODO(ebrevdo): MatMul support for broadcasting at least in the
    // batch dimension.
    auto a_dense_shape = a_matrix->dense_shape().vec<int64>();
    auto b_dense_shape = b_matrix->dense_shape().vec<int64>();

    TensorShape a_tensor_shape;
    TensorShape b_tensor_shape;
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(a_dense_shape, &a_tensor_shape));
    OP_REQUIRES_OK(ctx,
                   TensorShapeUtils::MakeShape(b_dense_shape, &b_tensor_shape));

    const int rank = a_tensor_shape.dims();
    const int row_dim = (rank == 2) ? 0 : 1;

    const int64 a_inner_dim =
        a_tensor_shape.dim_size(transpose_a_ ? row_dim : row_dim + 1);
    const int64 b_inner_dim =
        b_tensor_shape.dim_size(transpose_b_ ? row_dim + 1 : row_dim);

    const int batch_size = a_matrix->batch_size();

    OP_REQUIRES(
        ctx, a_inner_dim == b_inner_dim,
        errors::InvalidArgument(
            "Inner product dimensions of A and B do not agree.  Shapes are: ",
            a_tensor_shape.DebugString(), " vs. ",
            b_tensor_shape.DebugString()));

    Tensor c_dense_shape_t(cpu_allocator(), DT_INT64, TensorShape({rank}));
    auto c_dense_shape = c_dense_shape_t.vec<int64>();

    if (rank == 3) c_dense_shape(0) = batch_size;
    c_dense_shape(row_dim) =
        a_tensor_shape.dim_size(transpose_a_ ? row_dim + 1 : row_dim);
    c_dense_shape(row_dim + 1) =
        b_tensor_shape.dim_size(transpose_b_ ? row_dim : row_dim + 1);

    const int64 rows = c_dense_shape((rank == 2) ? 0 : 1);

    CSRSparseMatrix c;
    Tensor c_row_ptrs;
    Tensor c_col_inds;
    Tensor c_values;

    // TODO(ebrevdo): Re-enable transposing within the GEMM kernel when cuSparse
    // stops spitting out CUSPARSE_STATUS_INTERNAL_ERROR values for transposes.
    functor::CSRSparseSparseMatrixMatMul<Device, T> csr_gemm(
        ctx, /*transpose_a=*/false, /*adjoint_a=*/false, /*transpose_b=*/false);
    OP_REQUIRES_OK(ctx, csr_gemm.Initialize());

    Tensor c_batch_ptr_t(cpu_allocator(), DT_INT32,
                         TensorShape({batch_size + 1}));
    auto c_batch_ptr = c_batch_ptr_t.vec<int32>();
    c_batch_ptr(0) = 0;

    Tensor c_row_ptr_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                            DT_INT32, TensorShape({batch_size * (rows + 1)}),
                            &c_row_ptr_t));
    auto c_row_ptr = c_row_ptr_t.vec<int32>();

    // Possibly transpose a.
    const CSRSparseMatrix* a_input_matrix;
    // If we need to transpose a, we will store the result temporarily
    // in the object below.
    CSRSparseMatrix a_matrix_transposed;
    if (!transpose_a_) {
      a_input_matrix = a_matrix;
    } else {
      functor::CSRSparseMatrixTranspose<Device, T> transpose;
      OP_REQUIRES_OK(
          ctx, transpose(ctx, conjugate_a_, *a_matrix, &a_matrix_transposed));
      a_input_matrix = &a_matrix_transposed;
    }
    auto a_input_dense_shape = a_input_matrix->dense_shape().vec<int64>();

    // Possibly transpose b.
    const CSRSparseMatrix* b_input_matrix;
    // If we need to transpose a, we will store the result temporarily
    // in the object below.
    CSRSparseMatrix b_matrix_transposed;
    if (!transpose_b_) {
      b_input_matrix = b_matrix;
    } else {
      functor::CSRSparseMatrixTranspose<Device, T> transpose;
      OP_REQUIRES_OK(
          ctx, transpose(ctx, conjugate_b_, *b_matrix, &b_matrix_transposed));
      b_input_matrix = &b_matrix_transposed;
    }
    auto b_input_dense_shape = b_input_matrix->dense_shape().vec<int64>();

    for (int i = 0; i < batch_size; ++i) {
      // Calculate output sizes for all minibatch entries.
      // Store in c_batch_ptr and update c_row_ptrs.
      ConstCSRComponent<T> a_comp{a_input_matrix->row_pointers_vec(i),
                                  a_input_matrix->col_indices_vec(i),
                                  a_input_matrix->values_vec<T>(i),
                                  a_input_dense_shape};
      ConstCSRComponent<T> b_comp{b_input_matrix->row_pointers_vec(i),
                                  b_input_matrix->col_indices_vec(i),
                                  b_input_matrix->values_vec<T>(i),
                                  b_input_dense_shape};

      TTypes<int32>::UnalignedVec c_row_ptr_i(&c_row_ptr(i * (rows + 1)),
                                              rows + 1);

      int c_nnz_i;
      OP_REQUIRES_OK(ctx, csr_gemm.GetOutputStructure(a_comp, b_comp,
                                                      c_row_ptr_i, &c_nnz_i));
      c_batch_ptr(i + 1) = c_batch_ptr(i) + c_nnz_i;
    }

    Tensor c_col_ind_t;
    Tensor c_values_t;

    const int total_nnz = c_batch_ptr(batch_size);

    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                           &c_col_ind_t));
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_temp(DataTypeToEnum<T>::value,
                                      TensorShape({total_nnz}), &c_values_t));
    OP_REQUIRES_OK(ctx,
                   CSRSparseMatrix::CreateCSRSparseMatrix(
                       DataTypeToEnum<T>::value, c_dense_shape_t, c_batch_ptr_t,
                       c_row_ptr_t, c_col_ind_t, c_values_t, &c));

    for (int i = 0; i < batch_size; ++i) {
      ConstCSRComponent<T> a_comp{a_input_matrix->row_pointers_vec(i),
                                  a_input_matrix->col_indices_vec(i),
                                  a_input_matrix->values_vec<T>(i),
                                  a_input_dense_shape};
      ConstCSRComponent<T> b_comp{b_input_matrix->row_pointers_vec(i),
                                  b_input_matrix->col_indices_vec(i),
                                  b_input_matrix->values_vec<T>(i),
                                  b_input_dense_shape};
      CSRComponent<T> c_comp{c.row_pointers_vec(i), c.col_indices_vec(i),
                             c.values_vec<T>(i), c_dense_shape};
      OP_REQUIRES_OK(ctx, csr_gemm.Compute(a_comp, b_comp, &c_comp));
    }

    Tensor c_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    c_t.scalar<Variant>()() = std::move(c);
    ctx->set_output(0, c_t);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool conjugate_a_;
  bool conjugate_b_;
};

#define REGISTER_CPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSparseMatMul") \
                              .Device(DEVICE_CPU)          \
                              .TypeConstraint<T>("type"),  \
                          CSRSparseMatMulCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

#define REGISTER(DEV, T)                                   \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSparseMatMul") \
                              .Device(DEVICE_##DEV)        \
                              .TypeConstraint<T>("type"),  \
                          CSRSparseMatMulGPUOp<DEV##Device, T>);

#if GOOGLE_CUDA

#define REGISTER_GPU(T) REGISTER(GPU, T)

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA

#undef REGISTER

#if GOOGLE_CUDA
namespace functor {
template <typename T>
struct CSRSparseSparseMatrixMatMul<GPUDevice, T>
    : public CSRStructureModifyingFunctor<GPUDevice, T> {
  explicit CSRSparseSparseMatrixMatMul(OpKernelContext* ctx, bool transpose_a,
                                       bool adjoint_a, bool transpose_b)
      : ctx_(ctx),
        cuda_sparse_(ctx),
        initialized_(false),
        transpose_a_(transpose_a),
        adjoint_a_(adjoint_a),
        transpose_b_(transpose_b) {
    // TODO(ebrevdo): Figure out why transposed implementations crash cuSparse.
    transA_ = transpose_a ? (adjoint_a ? CUSPARSE_OPERATION_TRANSPOSE
                                       : CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                          : CUSPARSE_OPERATION_NON_TRANSPOSE;
    transB_ = transpose_b ? CUSPARSE_OPERATION_TRANSPOSE
                          : CUSPARSE_OPERATION_NON_TRANSPOSE;
  }

  Status Initialize() {
    if (adjoint_a_ && transpose_a_) {
      return errors::InvalidArgument(
          "Only one of adjoint_a and transpose_a may be true.");
    }

    TF_RETURN_IF_ERROR(cuda_sparse_.Initialize());
    TF_RETURN_IF_ERROR(descrA_.Initialize());
    TF_RETURN_IF_ERROR(descrB_.Initialize());
    TF_RETURN_IF_ERROR(descrC_.Initialize());
    initialized_ = true;
    return Status::OK();
  }

  Status GetOutputStructure(const ConstCSRComponent<T>& a,
                            const ConstCSRComponent<T>& b,
                            TTypes<int32>::UnalignedVec c_row_ptr,
                            int* output_nnz) {
    DCHECK(initialized_);

    const int m =
        a.dense_shape_host(a.dense_shape_host.size() - (transpose_a_ ? 1 : 2));
    if (!transpose_a_) {
      DCHECK_EQ(m, a.row_ptr.size() - 1);
    }
    DCHECK_EQ(m, c_row_ptr.size() - 1);
    const int k =
        a.dense_shape_host(a.dense_shape_host.size() - (transpose_a_ ? 2 : 1));
    if (!transpose_b_) {
      DCHECK_EQ(k, b.row_ptr.size() - 1);
    }
    const int nnzA = a.col_ind.size();
    const int nnzB = b.col_ind.size();

    const int n =
        b.dense_shape_host(b.dense_shape_host.size() - (transpose_b_ ? 2 : 1));

    *output_nnz = -1;

    TF_RETURN_IF_ERROR(cuda_sparse_.CsrgemmNnz(
        transA_, transB_, m, n, k, descrA_.descr(), nnzA, a.row_ptr.data(),
        a.col_ind.data(), descrB_.descr(), nnzB, b.row_ptr.data(),
        b.col_ind.data(), descrC_.descr(), c_row_ptr.data(), output_nnz));

    if (*output_nnz < 0) {
      return errors::Internal(
          "CSRMatMul: CsrgemmNnz returned nnzTotalDevHostPtr < 0: ",
          *output_nnz);
    }
    return Status::OK();
  }

  Status Compute(const ConstCSRComponent<T>& a, const ConstCSRComponent<T>& b,
                 CSRComponent<T>* c) {
    DCHECK(initialized_);

    const int m =
        a.dense_shape_host(a.dense_shape_host.size() - (transpose_a_ ? 1 : 2));
    if (!transpose_a_) {
      DCHECK_EQ(m, a.row_ptr.size() - 1);
    }
    DCHECK_EQ(m, c->dense_shape_host(c->dense_shape_host.size() - 2));
    DCHECK_EQ(m, c->row_ptr.size() - 1);
    const int k =
        a.dense_shape_host(a.dense_shape_host.size() - (transpose_a_ ? 2 : 1));
    if (!transpose_b_) {
      DCHECK_EQ(k, b.row_ptr.size() - 1);
    }
    const int nnzA = a.col_ind.size();
    const int nnzB = b.col_ind.size();

    const int n =
        b.dense_shape_host(b.dense_shape_host.size() - (transpose_b_ ? 2 : 1));
    DCHECK_EQ(n, c->dense_shape_host(c->dense_shape_host.size() - 1));

    TF_RETURN_IF_ERROR(cuda_sparse_.Csrgemm(
        transA_, transB_, m, k, n, descrA_.descr(), nnzA, a.values.data(),
        a.row_ptr.data(), a.col_ind.data(), descrB_.descr(), nnzB,
        b.values.data(), b.row_ptr.data(), b.col_ind.data(), descrC_.descr(),
        c->values.data(), c->row_ptr.data(), c->col_ind.data()));

    // TODO(ebrevdo): Add a flag to CSRSparseMatrix whether matrix
    // columns are sorted?  Above operation leads to unsorted columns.
    // For now, here is an example of how to ensure the output columns
    // are sorted.  Leaving here in case we realize we need to ensure
    // sorted columns in the future.
    //
    // TF_RETURN_IF_ERROR(cuda_sparse.Csru2csr(
    //     m, n, nnzTotalDevHostPtr, descrA_.descr(), c->values.data(),
    //     c->row_ptr.data(), c->col_ind.data()));

    return Status::OK();
  }

 private:
  OpKernelContext* ctx_;
  GpuSparse cuda_sparse_;
  bool initialized_;
  bool transpose_a_;
  bool adjoint_a_;
  bool transpose_b_;
  GpuSparseMatrixDescriptor descrA_;
  GpuSparseMatrixDescriptor descrB_;
  GpuSparseMatrixDescriptor descrC_;
  gpusparseOperation_t transA_;
  gpusparseOperation_t transB_;
};

}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
