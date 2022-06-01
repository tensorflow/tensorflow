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
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
using ::perftools::gputools::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_activation.h"
using ::perftools::gputools::rocm::ScopedActivateExecutorContext;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Op to convert SparseTensors to CSR SparseMatrices on the CPU.
// Takes a SparseTensor of rank 2 or (if batched) 3 as the input. The
// SparseTensor's indices must be present in the canonical, row-major ordering.
//
// Returns a (batched) CSR SparseMatrix with the same dense shape and non-zero
// values.
template <typename T>
class SparseTensorToCSRSparseMatrixCPUOp : public OpKernel {
 public:
  explicit SparseTensorToCSRSparseMatrixCPUOp(OpKernelConstruction* c)
      : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) final {
    const Tensor& indices = ctx->input(0);
    const Tensor& values = ctx->input(1);
    const Tensor& dense_shape = ctx->input(2);
    const int rank = dense_shape.NumElements();
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(dense_shape.shape()),
        errors::InvalidArgument("dense_shape must be rank 1 but got rank",
                                dense_shape.shape().dims()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(indices.shape()),
                errors::InvalidArgument("indices must be rank 2 but got rank",
                                        indices.shape().dims()));
    OP_REQUIRES(ctx, rank == 2 || rank == 3,
                errors::InvalidArgument("SparseTensor must have rank 2 or 3; ",
                                        "but indices has rank: ", rank));
    auto dense_shape_vec = dense_shape.vec<int64_t>();
    const int64_t batch_size = (rank == 2) ? 1 : dense_shape_vec(0);
    const int64_t num_rows = dense_shape_vec((rank == 2) ? 0 : 1);
    const int64_t total_nnz = values.NumElements();

    // Allocate output Tensors.
    TensorShape batch_ptr_shape;
    OP_REQUIRES_OK(
        ctx, TensorShape::BuildTensorShape({batch_size + 1}, &batch_ptr_shape));
    Tensor batch_ptr(cpu_allocator(), DT_INT32, batch_ptr_shape);
    TensorShape csr_col_ind_shape;
    OP_REQUIRES_OK(
        ctx, TensorShape::BuildTensorShape({total_nnz}, &csr_col_ind_shape));
    Tensor csr_col_ind(cpu_allocator(), DT_INT32, csr_col_ind_shape);
    TensorShape csr_row_ind_shape;
    OP_REQUIRES_OK(ctx, TensorShape::BuildTensorShape(
                            {(num_rows + 1) * batch_size}, &csr_row_ind_shape));
    Tensor csr_row_ptr(cpu_allocator(), DT_INT32, csr_row_ind_shape);

    // Fill the row pointers with zeros.
    functor::SetZeroFunctor<CPUDevice, int32> set_zero;
    set_zero(ctx->eigen_device<CPUDevice>(), csr_row_ptr.flat<int32>());

    // Convert from COO to CSR format.
    functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
    OP_REQUIRES_OK(
        ctx,
        coo_to_csr(batch_size, num_rows, indices.template matrix<int64_t>(),
                   batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                   csr_col_ind.vec<int32>()));

    // Create the CSRSparseMatrix object from its component Tensors and prepare
    // the Variant output Tensor.
    CSRSparseMatrix output_csr_matrix;
    OP_REQUIRES_OK(
        ctx, CSRSparseMatrix::CreateCSRSparseMatrix(
                 DataTypeToEnum<T>::value, dense_shape, batch_ptr, csr_row_ptr,
                 csr_col_ind, values, &output_csr_matrix));
    Tensor* output_csr_matrix_tensor;
    AllocatorAttributes cpu_alloc;
    cpu_alloc.set_on_host(true);
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &output_csr_matrix_tensor,
                                  cpu_alloc));
    output_csr_matrix_tensor->scalar<Variant>()() =
        std::move(output_csr_matrix);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class SparseTensorToCSRSparseMatrixGPUOp : public AsyncOpKernel {
 public:
  explicit SparseTensorToCSRSparseMatrixGPUOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    auto stream = c->op_device_context()->stream();
    const Device& d = c->eigen_device<Device>();

    const Tensor& indices_t = c->input(0);
    const Tensor& values_t = c->input(1);
    const Tensor& dense_shape_t = c->input(2);
    const int rank = dense_shape_t.NumElements();
    OP_REQUIRES_ASYNC(
        c, rank == 2 || rank == 3,
        errors::InvalidArgument("sparse tensor must have rank == 2 or 3; ",
                                "but indices has ", rank, " columns"),
        done);
    auto dense_shape = dense_shape_t.vec<int64_t>();
    const int64_t batch_size = (rank == 2) ? 1 : dense_shape(0);
    const int64_t rows = dense_shape((rank == 2) ? 0 : 1);
    const int64_t cols = dense_shape((rank == 2) ? 1 : 2);

    ScratchSpace<int32> nnz_per_batch_host(c, batch_size, /*on_host*/ true);

    Tensor nnz_per_batch_device_t;
    if (rank == 2) {
      // Simple case.
      nnz_per_batch_host.mutable_data()[0] = indices_t.dim_size(0);
    } else {
      OP_REQUIRES_OK_ASYNC(c,
                           c->allocate_temp(DT_INT32, TensorShape({batch_size}),
                                            &nnz_per_batch_device_t),
                           done);
      auto nnz_per_batch_device = nnz_per_batch_device_t.vec<int32>();

      functor::CalculateNNZPerBatchMatrixFromIndices<Device>
          calculate_nnz_from_indices;
      auto indices = indices_t.matrix<int64_t>();
      OP_REQUIRES_OK_ASYNC(
          c, calculate_nnz_from_indices(c, indices, nnz_per_batch_device),
          done);

      perftools::gputools::DeviceMemoryBase nnz_per_batch_device_ptr(
          static_cast<void*>(nnz_per_batch_device.data()));

      OP_REQUIRES_ASYNC(
          c,
          stream
              ->ThenMemcpy(nnz_per_batch_host.mutable_data() /*host_dst*/,
                           nnz_per_batch_device_ptr /*gpu_src*/,
                           batch_size * sizeof(int32) /*size*/)
              .ok(),
          errors::Internal("SparseTensorToSparseMatrixGPUOp: failed to copy "
                           "nnz_per_batch from device"),
          done);
    }

    TensorReference nnz_per_batch_device_ref(nnz_per_batch_device_t);
    auto convert_to_csr = [this, c, batch_size, nnz_per_batch_host,
                           nnz_per_batch_device_ref, stream, &d, &values_t,
                           &indices_t, &dense_shape_t, dense_shape, rows, cols,
                           rank, done]() {
      // The data has been copied out of the nnz_per_batch_device
      // tensor by the time we get here; we can unreference it.
      nnz_per_batch_device_ref.Unref();

      auto nnz_per_batch = nnz_per_batch_host.tensor().vec<int32>();

      // Ensure that within the callback, the proper GPU settings are
      // configured.
      ScopedActivateExecutorContext scoped_activation{stream->parent()};
      Tensor batch_ptr_t(cpu_allocator(), DT_INT32,
                         TensorShape({batch_size + 1}));

      auto batch_ptr = batch_ptr_t.vec<int32>();
      auto indices = indices_t.matrix<int64_t>();

      batch_ptr(0) = 0;
      for (int i = 0; i < batch_size; ++i) {
        batch_ptr(i + 1) = batch_ptr(i) + nnz_per_batch(i);
      }
      int total_nnz = batch_ptr(batch_size);
      OP_REQUIRES_ASYNC(
          c, total_nnz == values_t.NumElements(),
          errors::Internal("nnz returned by "
                           "CalculateNNZPerBatchMatrixFromInd"
                           "ices != len(values): ",
                           total_nnz, " vs. ", values_t.NumElements()),
          done);

      Tensor coo_col_ind_t;
      Tensor csr_row_ptr_t;
      Tensor csr_values_t = values_t;

      Tensor coo_row_ind_t;
      OP_REQUIRES_OK_ASYNC(
          c,
          c->allocate_temp(DT_INT32, TensorShape({total_nnz}), &coo_row_ind_t),
          done);
      OP_REQUIRES_OK_ASYNC(
          c,
          c->allocate_temp(DT_INT32, TensorShape({total_nnz}), &coo_col_ind_t),
          done);
      OP_REQUIRES_OK_ASYNC(
          c,
          c->allocate_temp(DT_INT32, TensorShape({batch_size * (rows + 1)}),
                           &csr_row_ptr_t),
          done);

      auto coo_row_ind = coo_row_ind_t.vec<int32>();
      auto coo_col_ind = coo_col_ind_t.vec<int32>();
      auto csr_row_ptr = csr_row_ptr_t.vec<int32>();

      // Convert SparseTensor rep to coo row ind, coo col ind.
      if (total_nnz > 0) {
        functor::SparseTensorToCOOSparseMatrix<Device> st_to_coo;
        st_to_coo(d, dense_shape, indices, coo_row_ind, coo_col_ind);
      }

      // Set all csr row pointers to zero, so that when iterating over
      // batches converting coo to csr, we do not have to perform an
      // unaligned SetZero for any nnz == 0 minibatches.  coo2csr has
      // a bug if you have empty coo rows.
      // TODO(ebrevdo): File bug w/ nvidia so coo2csr can handle
      // zero-element input coo rows.
      functor::SetZeroFunctor<Device, int32> set_zero;
      set_zero(d, csr_row_ptr_t.flat<int32>());

      functor::COOSparseMatrixToCSRSparseMatrix<Device> coo_to_csr;
      for (int i = 0; i < batch_size; ++i) {
        int nnz_i = batch_ptr(i + 1) - batch_ptr(i);
        if (nnz_i == 0) {
          // This is an empty minibatch; no call to coo2csr: it's
          // handled by the SetZero above.
        } else {
          // Convert coo to csr.
          auto coo_row_ind_i =
              TTypes<int32>::UnalignedVec(&coo_row_ind(batch_ptr(i)), nnz_i);
          auto csr_row_ptr_i = TTypes<int32>::UnalignedVec(
              &csr_row_ptr((rows + 1) * i), rows + 1);
          OP_REQUIRES_OK_ASYNC(
              c, coo_to_csr(c, rows, cols, coo_row_ind_i, csr_row_ptr_i), done);
        }
      }

      CSRSparseMatrix matrix;
      OP_REQUIRES_OK_ASYNC(
          c,
          CSRSparseMatrix::CreateCSRSparseMatrix(
              values_t.dtype(), dense_shape_t, batch_ptr_t, csr_row_ptr_t,
              coo_col_ind_t, csr_values_t, &matrix),
          done);
      Tensor* matrix_t;
      AllocatorAttributes cpu_alloc;
      cpu_alloc.set_on_host(true);
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, TensorShape({}), &matrix_t, cpu_alloc),
          done);
      matrix_t->scalar<Variant>()() = std::move(matrix);

      done();
    };

    if (rank == 2) {
      convert_to_csr();
    } else {
      // Launch the GPU kernel to count nnz entries, then call convert_to_csr.
      c->device()->tensorflow_accelerator_device_info()->event_mgr->ThenExecute(
          stream, convert_to_csr);
    }
  }
};

namespace functor {

template <>
Status CalculateNNZPerBatchMatrixFromIndices<GPUDevice>::operator()(
    OpKernelContext* c, TTypes<int64_t>::ConstMatrix indices,
    TTypes<int32>::Vec nnz_per_batch);
extern template struct CalculateNNZPerBatchMatrixFromIndices<GPUDevice>;

template <>
struct SparseTensorToCOOSparseMatrix<GPUDevice> {
  void operator()(const GPUDevice& d,
                  TTypes<int64_t>::ConstVec host_dense_shape,
                  TTypes<int64_t>::ConstMatrix indices,
                  TTypes<int>::Vec coo_row_ind, TTypes<int>::Vec coo_col_ind);
};
extern template struct SparseTensorToCOOSparseMatrix<GPUDevice>;

template <>
struct COOSparseMatrixToCSRSparseMatrix<GPUDevice> {
  Status operator()(OpKernelContext* c, const int rows, const int cols,
                    TTypes<int>::UnalignedVec coo_row_ind,
                    TTypes<int>::UnalignedVec csr_row_ptr) {
    GpuSparse cuda_sparse(c);
    TF_RETURN_IF_ERROR(cuda_sparse.Initialize());
    return cuda_sparse.Coo2csr(coo_row_ind.data(),
                               /*nnz*/ coo_row_ind.size(),
                               /*m == rows of A*/ rows, csr_row_ptr.data());
  }
};
extern template struct COOSparseMatrixToCSRSparseMatrix<GPUDevice>;

}  // namespace functor

#define REGISTER_GPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorToCSRSparseMatrix") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<T>("T")           \
                              .HostMemory("dense_shape"),       \
                          SparseTensorToCSRSparseMatrixGPUOp<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_CPU(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("SparseTensorToCSRSparseMatrix") \
                              .Device(DEVICE_CPU)               \
                              .TypeConstraint<T>("T"),          \
                          SparseTensorToCSRSparseMatrixCPUOp<T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

}  // namespace tensorflow
