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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_solvers.h"
#include "tensorflow/core/util/cuda_sparse.h"
#endif

namespace tensorflow {
namespace {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// Validate that CSR SparseMatrix has the expected dtype and rank 2 or 3.
Status ValidateCSRSparseMatrix(const CSRSparseMatrix& csr_sparse_matrix,
                               DataType expected_dtype) {
  if (csr_sparse_matrix.dtype() != expected_dtype) {
    return errors::InvalidArgument(
        "Expected a CSRSparseMatrix of type ", DataTypeString(expected_dtype),
        " but saw type: ", DataTypeString(csr_sparse_matrix.dtype()));
  }
  const int rank = csr_sparse_matrix.dense_shape().dim_size(0);
  if (rank != 2 && rank != 3) {
    return errors::InvalidArgument("CSR SparseMatrix must have rank 2 or 3; ",
                                   "but dense_shape has size ", rank);
  }
  return Status::OK();
}
}  // namespace

// Op to convert a (batched) CSR SparseMatrix to SparseTensors on the CPU.
// The resulting SparseTensor will have the same dense shape and non-zero values
// as the CSR SparseMatrix. rank 2 or (if batched) 3. Moreover, the resulting
// SparseTensor's indices will be present in the canonical, row-major ordering.
template <typename T>
class CSRSparseMatrixToSparseTensorCPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToSparseTensorCPUOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));
    OP_REQUIRES_OK(c, ValidateCSRSparseMatrix(*csr_sparse_matrix,
                                              DataTypeToEnum<T>::value));

    // Copy the SparseTensor's dense_shape and values from the CSRSparseMatrix.
    c->set_output(1, csr_sparse_matrix->values());
    const Tensor& dense_shape = csr_sparse_matrix->dense_shape();
    c->set_output(2, dense_shape);

    const int batch_size = csr_sparse_matrix->batch_size();
    const int64 total_nnz = csr_sparse_matrix->total_nnz();
    const int rank = csr_sparse_matrix->dense_shape().dim_size(0);
    auto dense_shape_vec = dense_shape.vec<int64>();
    const int64 num_rows = dense_shape_vec((rank == 2) ? 0 : 1);

    Tensor* indices;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, TensorShape({total_nnz, rank}), &indices));
    auto indices_flat = indices->template flat<int64>();

    auto csr_row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto csr_col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();

    // Process the individual batches in parallel using a threadpool.
    auto shard = [&](int64 batch_begin, int64 batch_end) {
      for (int64 batch_idx = batch_begin; batch_idx < batch_end; ++batch_idx) {
        const int64 csr_batch_offset = batch_ptrs(batch_idx);

        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
          const int64 row_offset = batch_idx * (num_rows + 1) + row_idx;

          // The column indices of the current row lie in the range:
          //  [csr_row_ptr[row_offset], csr_row_ptr[row_offset + 1])
          const int64 col_begin = csr_row_ptr(row_offset);
          const int64 col_end = csr_row_ptr(row_offset + 1);
          for (int64 i = col_begin; i < col_end; ++i) {
            const int64 col_idx = csr_col_ind(csr_batch_offset + i);
            const int64 indices_offset = rank * (csr_batch_offset + i);

            if (rank == 2) {
              indices_flat(indices_offset) = row_idx;
              indices_flat(indices_offset + 1) = col_idx;
            } else {  // rank == 3
              indices_flat(indices_offset) = batch_idx;
              indices_flat(indices_offset + 1) = row_idx;
              indices_flat(indices_offset + 2) = col_idx;
            }
          }
        }
      }
    };
    auto worker_threads = *(c->device()->tensorflow_cpu_worker_threads());
    // TODO(anudhyan): Estimate the cost per unit based on Eigen::TensorOpCost
    // units and scale based on benchmarks.
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          csr_sparse_matrix->total_nnz() / batch_size /* cost per unit */,
          shard);
  }
};

template <typename Device, typename T>
class CSRSparseMatrixToSparseTensorGPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToSparseTensorGPUOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));
    OP_REQUIRES_OK(c, ValidateCSRSparseMatrix(*csr_sparse_matrix,
                                              DataTypeToEnum<T>::value));

    const Tensor& dense_shape_t = csr_sparse_matrix->dense_shape();
    c->set_output(2, dense_shape_t);
    const int rank = dense_shape_t.dim_size(0);
    const int batch_size = csr_sparse_matrix->batch_size();
    const int64 total_nnz = csr_sparse_matrix->total_nnz();

    auto dense_shape = dense_shape_t.vec<int64>();
    const int64 rows = dense_shape((rank == 2) ? 0 : 1);

    Tensor* indices_t;
    OP_REQUIRES_OK(
        c, c->allocate_output(0, TensorShape({total_nnz, rank}), &indices_t));

    Tensor* values_t;
    OP_REQUIRES_OK(c,
                   c->allocate_output(1, TensorShape({total_nnz}), &values_t));

    functor::CSRSparseMatrixToCOOSparseMatrix<Device> csr_to_coo;
    auto indices = indices_t->matrix<int64>();

    auto csr_row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto coo_col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();

    Tensor coo_row_ind_t;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                       &coo_row_ind_t));
    auto coo_row_ind = coo_row_ind_t.vec<int32>();

    // TODO(ebrevdo): Convert to one or two single kernel calls,
    // where the kernels are batch-friendly.
    for (int i = 0; i < batch_size; ++i) {
      const int nnz_i = csr_sparse_matrix->nnz(i);
      if (nnz_i == 0) {
        // No copying required.  Avoid failure case below.
        continue;
      }
      const TTypes<int32>::UnalignedConstVec csr_row_ptr_i(
          &csr_row_ptr((rows + 1) * i), rows + 1);
      const TTypes<int32>::UnalignedVec coo_row_ind_i(
          &coo_row_ind(csr_sparse_matrix->batch_offset(i)), nnz_i);
      OP_REQUIRES_OK(c, csr_to_coo(c, csr_row_ptr_i, coo_row_ind_i));
    }

    if (total_nnz > 0) {
      functor::COOSparseMatrixToSparseTensor<Device> coo_to_st;
      OP_REQUIRES_OK(c, coo_to_st(c, dense_shape, batch_ptrs, coo_row_ind,
                                  coo_col_ind, indices));
    }

    *values_t = csr_sparse_matrix->values();
  }
};

#define REGISTER_GPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToSparseTensor") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("type")        \
                              .HostMemory("dense_shape"),       \
                          CSRSparseMatrixToSparseTensorGPUOp<GPUDevice, T>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {
template <>
struct COOSparseMatrixToSparseTensor<GPUDevice> {
  Status operator()(OpKernelContext* ctx,
                    TTypes<int64>::ConstVec host_dense_shape,
                    TTypes<int>::ConstVec host_batch_ptrs,
                    TTypes<int>::Vec coo_row_ind,
                    TTypes<int>::ConstVec coo_col_ind,
                    TTypes<int64>::Matrix indices);
};
extern template struct COOSparseMatrixToSparseTensor<GPUDevice>;

template <>
struct CSRSparseMatrixToCOOSparseMatrix<GPUDevice> {
  Status operator()(OpKernelContext* c,
                    TTypes<const int>::UnalignedVec csr_row_ptr,
                    TTypes<int>::UnalignedVec coo_row_ind);
};
extern template struct CSRSparseMatrixToCOOSparseMatrix<GPUDevice>;

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToSparseTensor") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<T>("type"),       \
                          CSRSparseMatrixToSparseTensorCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

}  // namespace tensorflow
