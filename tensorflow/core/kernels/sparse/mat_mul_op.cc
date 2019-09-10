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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/kernels/sparse/transpose_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

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
    transpose_a_ = transpose_a_ || adjoint_a;
    transpose_b_ = transpose_b_ || adjoint_b;
  }

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* a_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    const Tensor& b_t = ctx->input(1);

    OP_REQUIRES(ctx, a_matrix->dtype() == b_t.dtype(),
                errors::InvalidArgument(
                    "Input types don't match.  a.dtype == ",
                    DataTypeString(a_matrix->dtype()),
                    " vs. b.dtype == ", DataTypeString(b_t.dtype())));

    const int a_rank = a_matrix->dims();
    const int b_rank = b_t.dims();
    const int64 batch_size = (b_rank == 2) ? 1 : b_t.dim_size(0);

    // TODO(ebrevdo): Add support for broadcasting matmul.
    OP_REQUIRES(ctx, a_rank == b_rank,
                errors::InvalidArgument("Ranks of a and b must match, saw: ",
                                        a_rank, " vs. ", b_rank, "."));
    OP_REQUIRES(ctx, a_matrix->batch_size() == batch_size,
                errors::InvalidArgument(
                    "Batch sizes of a and b must match, saw: ",
                    a_matrix->batch_size(), " vs. ", batch_size, "."));

    const Tensor& a_dense_shape_t = a_matrix->dense_shape();
    TensorShape a_dense_tensor_shape;
    auto a_dense_shape = a_dense_shape_t.vec<int64>();
    OP_REQUIRES_OK(
        ctx, TensorShapeUtils::MakeShape(a_dense_shape, &a_dense_tensor_shape));

    const int row_dim = (a_rank == 2) ? 0 : 1;
    const int64 a_inner_dim =
        a_dense_tensor_shape.dim_size(transpose_a_ ? row_dim : row_dim + 1);
    const int64 b_inner_dim =
        b_t.shape().dim_size(transpose_b_ ? row_dim + 1 : row_dim);
    const int64 b_outer_dim =
        b_t.shape().dim_size(transpose_b_ ? row_dim : row_dim + 1);
    const int64 b_slice_size = b_inner_dim * b_outer_dim;

    OP_REQUIRES(
        ctx, a_inner_dim == b_inner_dim,
        errors::InvalidArgument(
            "Inner product dimensions of A and B do not agree.  Shapes are: ",
            a_dense_tensor_shape.DebugString(), " vs. ",
            b_t.shape().DebugString()));

    TensorShape c_shape;
    if (a_rank == 3) c_shape.AddDim(batch_size);
    if (transpose_output_) {
      c_shape.AddDim(b_t.dim_size(transpose_b_ ? row_dim : row_dim + 1));
      c_shape.AddDim(
          a_dense_tensor_shape.dim_size(transpose_a_ ? row_dim + 1 : row_dim));
    } else {
      c_shape.AddDim(
          a_dense_tensor_shape.dim_size(transpose_a_ ? row_dim + 1 : row_dim));
      c_shape.AddDim(b_t.dim_size(transpose_b_ ? row_dim : row_dim + 1));
    }

    const int64 c_matrix_lhs = c_shape.dim_size(row_dim);
    const int64 c_matrix_rhs = c_shape.dim_size(row_dim + 1);
    const int64 c_slice_size = c_matrix_lhs * c_matrix_rhs;
    Tensor* c_t;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &c_t));

    const Device& d = ctx->eigen_device<Device>();

    if (b_outer_dim == 1) {
      // Call matrix-vector multiply if b is a vector.
      TTypes<int64>::ConstVec a_dense_shape_comp(a_dense_shape.data() + row_dim,
                                                 2);
      Tensor b_conj_t;
      const T* b_base_ptr = b_t.template flat<T>().data();
      bool conjugate_a = conjugate_a_;
      bool conjugate_output = conjugate_output_;
      if (conjugate_b_) {
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
          functor::maybe_conj<Device, T>::run(d, b_t, &b_conj_t);
          b_base_ptr = b_conj_t.template flat<T>().data();
        }
      }

      functor::CSRSparseMatrixMatVec<Device, T> csr_spmv(transpose_a_,
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
        functor::maybe_conj_inplace<Device, T>::run(d, c_t);
      }
      return;
    }

    functor::CSRSparseMatrixMatMul<Device, T> csr_spmmadd(transpose_output_);

    Tensor c_mat_col_major_t;
    if (!transpose_output_) {
      // If transpose_output is false, we'll need to transpose the (col
      // major) output of the csrgemm call to get proper (row-major)
      // output.  Which means we need to keep a temporary buffer to
      // store the intermediate gemm output.
      TensorShape c_mat_col_major_shape;
      if (a_rank == 2) {
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
    auto c_mat_col_major =
        (transpose_output_) ? c_t->flat<T>() : c_mat_col_major_t.flat<T>();

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
    Tensor b_t_input;
    if (!transpose_b_) {
      b_t_input = b_t;
    } else {
      TensorShape b_t_transposed_shape;
      if (a_rank == 3) {
        b_t_transposed_shape.AddDim(batch_size);
      }
      b_t_transposed_shape.AddDim(b_t.dim_size(row_dim + 1));
      b_t_transposed_shape.AddDim(b_t.dim_size(row_dim));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                             b_t_transposed_shape, &b_t_input));
      const Device& d = ctx->eigen_device<Device>();
      if (conjugate_b_) {
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

    if (!transpose_output_) {
      // We need to return values in row major format, so transpose
      // the column-major values in c_mat_col_major_t to row-major output c_t.
      OP_REQUIRES_OK(ctx, DoMatrixTranspose(d, /*input=*/c_mat_col_major_t,
                                            /*output=*/c_t));
    }
    if (conjugate_output_) {
      functor::maybe_conj_inplace<Device, T>::run(d, c_t);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool conjugate_a_;
  bool conjugate_b_;
  bool transpose_output_;
  bool conjugate_output_;
};

#define REGISTER(DEV, T)                                                      \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("SparseMatrixMatMul").Device(DEVICE_##DEV).TypeConstraint<T>("T"), \
      CSRMatMulOp<DEV##Device, T>);

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
class CSRSparseMatrixMatMul<GPUDevice, T> {
 public:
  explicit CSRSparseMatrixMatMul(const bool transpose_output)
      : transpose_output_(transpose_output) {}

  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 typename TTypes<T>::UnalignedConstMatrix b,
                 typename TTypes<T>::UnalignedMatrix c) {
    CudaSparse cuda_sparse(ctx);
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
      const cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

      // transB: b is row-major, and cusparse requires col-major b (or
      // equivalently transB == transpose).  this version is actually more
      // efficient.
      const cusparseOperation_t transB = CUSPARSE_OPERATION_TRANSPOSE;

      cusparseMatDescr_t descrA;
      TF_RETURN_IF_CUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
      TF_RETURN_IF_CUSPARSE_ERROR(
          cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_CUSPARSE_ERROR(
          cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

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
      : transA_(TransposeAndConjugateToCuSparseOp(transpose_a, conjugate_a,
                                                  &status_)) {}

  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 const T* x, T* y) {
    TF_RETURN_IF_ERROR(status_);
    CudaSparse cuda_sparse(ctx);
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

      cusparseMatDescr_t descrA;
      TF_RETURN_IF_CUSPARSE_ERROR(cusparseCreateMatDescr(&descrA));
      TF_RETURN_IF_CUSPARSE_ERROR(
          cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
      TF_RETURN_IF_CUSPARSE_ERROR(
          cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

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
  const cusparseOperation_t transA_;
};

}  // namespace functor

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
