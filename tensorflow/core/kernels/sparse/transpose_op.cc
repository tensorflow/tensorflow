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

// Implements the kernel for the CSRTranspose op, which transposes the
// two innermost dimensions of a CSRSparseMatrix object stored in a
// DT_VARIANT.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_sparse.h"
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/kernels/sparse/transpose_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRTransposeOp : public OpKernel {
 public:
  explicit CSRTransposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("conjugate", &conjugate_));
  }

  void Compute(OpKernelContext* ctx) override {
    const CSRSparseMatrix* input_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &input_matrix));
    OP_REQUIRES(
        ctx, input_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of input is not equal to 'type': ",
                                DataTypeString(input_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    // Allocate output shapes
    functor::CSRSparseMatrixTranspose<Device, T> transpose;
    CSRSparseMatrix output_matrix;
    OP_REQUIRES_OK(ctx,
                   transpose(ctx, conjugate_, *input_matrix, &output_matrix));
    Tensor output_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    output_t.scalar<Variant>()() = std::move(output_matrix);
    ctx->set_output(0, output_t);
  }

 private:
  bool conjugate_;
};

#ifdef GOOGLE_CUDA
#define REGISTER(DEV, T)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixTranspose")   \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRTransposeOp<DEV##Device, T>);

REGISTER(GPU, float)
REGISTER(GPU, double)
REGISTER(GPU, complex64)
REGISTER(GPU, complex128)

#undef REGISTER
#endif  // GOOGLE_CUDA

namespace functor {

template <typename Device, typename T>
Status CSRSparseMatrixTranspose<Device, T>::operator()(
    OpKernelContext* ctx, bool conjugate, const CSRSparseMatrix& input_matrix,
    CSRSparseMatrix* output_matrix) {
  const int rank = input_matrix.dims();
  Tensor output_dense_shape_t(cpu_allocator(), DT_INT64, TensorShape({rank}));
  const Tensor& input_dense_shape_t = input_matrix.dense_shape();
  auto input_dense_shape = input_dense_shape_t.vec<int64>();
  auto output_dense_shape = output_dense_shape_t.vec<int64>();
  const int64 batch_size = input_matrix.batch_size();
  if (rank == 3) {
    output_dense_shape(0) = batch_size;
  }
  output_dense_shape(rank - 2) = input_dense_shape(rank - 1);
  output_dense_shape(rank - 1) = input_dense_shape(rank - 2);
  const int64 output_rows = output_dense_shape(rank - 2);

  // nnzs per batch do not change with matrix transposition.
  Tensor batch_ptr_t = input_matrix.batch_pointers();
  const int total_nnz = input_matrix.total_nnz();

  Tensor output_row_ptr_t;
  Tensor output_col_ind_t;
  Tensor output_values_t;

  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DT_INT32, TensorShape({batch_size * (output_rows + 1)}),
      &output_row_ptr_t));
  TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                        &output_col_ind_t));
  TF_RETURN_IF_ERROR(ctx->allocate_temp(
      DataTypeToEnum<T>::value, TensorShape({total_nnz}), &output_values_t));

  TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
      DataTypeToEnum<T>::value, output_dense_shape_t, batch_ptr_t,
      output_row_ptr_t, output_col_ind_t, output_values_t, output_matrix));

  // Set the output row pointers to zero, in case we hit any empty
  // input batches.
  functor::SetZeroFunctor<Device, int32> set_zero;
  const Device& d = ctx->eigen_device<Device>();
  set_zero(d, output_row_ptr_t.flat<int32>());

  functor::CSRSparseMatrixTransposeComponent<Device, T> transpose_component;
  for (int i = 0; i < batch_size; ++i) {
    if (output_matrix->nnz(i) == 0) {
      continue;
    }
    ConstCSRComponent<T> input_comp{
        input_matrix.row_pointers_vec(i), input_matrix.col_indices_vec(i),
        input_matrix.values_vec<T>(i), input_dense_shape};
    CSRComponent<T> output_comp{
        output_matrix->row_pointers_vec(i), output_matrix->col_indices_vec(i),
        output_matrix->values_vec<T>(i), output_dense_shape};

    TF_RETURN_IF_ERROR(transpose_component(ctx, input_comp, &output_comp));
  }
  if (conjugate) {
    // conjugate all values with a single kernel launch.
    maybe_conj_inplace<Device, T>::run(d, &output_values_t);
  }

  return Status::OK();
}

#ifdef GOOGLE_CUDA

template <typename T>
struct CSRSparseMatrixTransposeComponent<GPUDevice, T> {
  Status operator()(OpKernelContext* ctx, const ConstCSRComponent<T>& x,
                    CSRComponent<T>* y) {
    CudaSparse cuda_sparse(ctx);
    TF_RETURN_IF_ERROR(cuda_sparse.Initialize());
    const cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
    const int rank = x.dense_shape_host.size();
    const int m = x.row_ptr.size() - 1;
    const int n = x.dense_shape_host(rank - 1);
    const int nnz = x.col_ind.size();
    DCHECK_EQ(nnz, x.values.size());
    DCHECK_EQ(n, y->row_ptr.size() - 1);
    DCHECK_EQ(rank, y->dense_shape_host.size());
    DCHECK_EQ(m, y->dense_shape_host(rank - 1));
    DCHECK_EQ(nnz, y->col_ind.size());
    DCHECK_EQ(nnz, y->values.size());

    return cuda_sparse.Csr2csc(
        m, n, nnz, x.values.data() /*csrVal*/, x.row_ptr.data() /*csrRowPtr*/,
        x.col_ind.data() /*csrColInd*/, y->values.data() /*cscVal*/,
        y->col_ind.data() /*cscRowInd*/, y->row_ptr.data() /*cscColPtr*/,
        copyValues);
    return Status::OK();
  }
};
#endif  // GOOGLE_CUDA
}  // namespace functor

}  // namespace tensorflow
