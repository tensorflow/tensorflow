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
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/scatter_nd_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_solvers.h"
#include "tensorflow/core/util/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Op to convert a (batched) CSR SparseMatrix to dense Tensors on the CPU.
// The resulting Tensor will have rank 2 or (if batched) 3. Missing values in
// the CSR SparseMatrix are interpreted as zeros in the dense Tensor.
template <typename Device, typename T>
class CSRSparseMatrixToDenseCPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToDenseCPUOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* context) override {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(context,
                   ExtractVariantFromInput(context, 0, &csr_sparse_matrix));

    OP_REQUIRES(
        context, csr_sparse_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument(
            "Asked for a CSRSparseMatrix of type ",
            DataTypeString(DataTypeToEnum<T>::value),
            " but saw dtype: ", DataTypeString(csr_sparse_matrix->dtype())));

    const Tensor& dense_shape_t = csr_sparse_matrix->dense_shape();
    const int rank = dense_shape_t.dim_size(0);
    OP_REQUIRES(context, rank == 2 || rank == 3,
                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                        "but dense_shape has size ", rank));

    auto dense_shape = dense_shape_t.vec<int64>();
    const int64 num_rows = dense_shape((rank == 2) ? 0 : 1);
    const int64 num_cols = dense_shape((rank == 2) ? 1 : 2);

    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();
    auto row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto values = csr_sparse_matrix->values().vec<T>();

    TensorShape dense_tensor_shape;
    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(dense_shape.data(),
                                                        dense_shape.size(),
                                                        &dense_tensor_shape));
    Tensor dense_t(cpu_allocator(), DataTypeToEnum<T>::value,
                   dense_tensor_shape);

    // Fill the dense tensor with zeros.
    functor::SetZeroFunctor<Device, T> set_zero;
    set_zero(context->eigen_device<Device>(), dense_t.flat<T>());

    auto dense_ptr = dense_t.flat<T>().data();

    // Process the individual batches in parallel using a threadpool.
    auto shard = [&](int64_t batch_begin, int64_t batch_end) {
      for (int64_t batch_idx = batch_begin; batch_idx < batch_end;
           ++batch_idx) {
        const int64 csr_batch_offset = batch_ptrs(batch_idx);
        const int64 dense_batch_offset = batch_idx * num_rows * num_cols;

        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
          const int64 row_offset = batch_idx * (num_rows + 1) + row_idx;
          const int64 col_begin = row_ptr(row_offset);
          const int64 col_end = row_ptr(row_offset + 1);
          for (int64_t i = col_begin; i < col_end; ++i) {
            const int64 col_idx = col_ind(csr_batch_offset + i);
            dense_ptr[dense_batch_offset + (row_idx * num_cols) + col_idx] =
                values(csr_batch_offset + i);
          }
        }
      }
    };
    const int batch_size = csr_sparse_matrix->batch_size();
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          csr_sparse_matrix->total_nnz() / batch_size /* cost per unit */,
          shard);

    context->set_output(0, dense_t);
  }
};

template <typename Device, typename T>
class CSRSparseMatrixToDenseGPUOp : public OpKernel {
 public:
  explicit CSRSparseMatrixToDenseGPUOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));

    OP_REQUIRES(
        c, csr_sparse_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument(
            "Asked for a CSRSparseMatrix of type ",
            DataTypeString(DataTypeToEnum<T>::value),
            " but saw dtype: ", DataTypeString(csr_sparse_matrix->dtype())));

    const Tensor& dense_shape_t = csr_sparse_matrix->dense_shape();
    const int rank = dense_shape_t.dim_size(0);
    OP_REQUIRES(c, rank == 2 || rank == 3,
                errors::InvalidArgument("sparse matrix must have rank 2 or 3; ",
                                        "but dense_shape has size ", rank));

    const int batch_size = csr_sparse_matrix->batch_size();
    const int64 total_nnz = csr_sparse_matrix->total_nnz();

    auto dense_shape = dense_shape_t.vec<int64>();
    const int64 rows = dense_shape((rank == 2) ? 0 : 1);

    Tensor indices_t;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT64, TensorShape({total_nnz, rank}),
                                       &indices_t));

    Tensor values_t;
    OP_REQUIRES_OK(c, c->allocate_temp(DataTypeToEnum<T>::value,
                                       TensorShape({total_nnz}), &values_t));

    functor::CSRSparseMatrixToCOOSparseMatrix<Device> csr_to_coo;
    auto indices = indices_t.matrix<int64>();

    auto csr_row_ptr = csr_sparse_matrix->row_pointers().vec<int32>();
    auto coo_col_ind = csr_sparse_matrix->col_indices().vec<int32>();
    auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();

    Tensor coo_row_ind_t;
    OP_REQUIRES_OK(c, c->allocate_temp(DT_INT32, TensorShape({total_nnz}),
                                       &coo_row_ind_t));
    auto coo_row_ind = coo_row_ind_t.vec<int32>();

    // TODO(ebrevdo): just write a custom kernel that converts from
    // csr to dense.
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

    values_t = csr_sparse_matrix->values();

    Tensor dense_t;
    TensorShape dense_tensor_shape;
    OP_REQUIRES_OK(
        c, TensorShapeUtils::MakeShape(dense_shape.data(), dense_shape.size(),
                                       &dense_tensor_shape));
    OP_REQUIRES_OK(
        c,
        functor::DoScatterNd<Device, T, int64, scatter_nd_op::UpdateOp::ASSIGN>(
            c, indices_t, values_t, dense_tensor_shape, &dense_t,
            true /*allocate*/));
    c->set_output(0, dense_t);
  }
};

#define REGISTER_GPU(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToDense")  \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<T>("type"), \
                          CSRSparseMatrixToDenseGPUOp<GPUDevice, T>);

#define REGISTER_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixToDense")  \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("type"), \
                          CSRSparseMatrixToDenseCPUOp<CPUDevice, T>);
REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_CPU
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

}  // namespace tensorflow
