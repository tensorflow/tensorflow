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
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/gather_nd_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
using ::perftools::gputools::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_activation.h"
using ::perftools::gputools::rocm::ScopedActivateExecutorContext;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Op to convert dense matrices to CSR SparseMatrices on the CPU.
// Takes a Tensor of rank 2 or (if batched) 3 and a corresponding list of
// indices as input.
//
// The (batched) CSR SparseMatrix is constructed using only
// the values at the given indices. This implementation assumes that the indices
// are sorted with respect to batch indices and are in row-major order.
template <typename Device, typename T>
class DenseToCSRSparseMatrixCPUOp : public OpKernel {
 public:
  explicit DenseToCSRSparseMatrixCPUOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& indices = ctx->input(1);

    // TODO(anudhyan): Factor out common input validation for CPU and GPU ops
    // into a single function.
    const TensorShape& dense_tensor_shape = params.shape();
    const int rank = params.dims();
    OP_REQUIRES(ctx, rank == 2 || rank == 3,
                errors::InvalidArgument(
                    "params must have rank == 2 or 3; ",
                    "but saw shape: ", dense_tensor_shape.DebugString()));
    OP_REQUIRES(
        ctx, indices.dims() == 2,
        errors::InvalidArgument("indices must be a matrix, but saw shape: ",
                                indices.shape().DebugString()));
    OP_REQUIRES(
        ctx, indices.dim_size(1) == rank,
        errors::InvalidArgument(
            "indices.shape[1] must be equal to the rank of params, but saw: ",
            indices.dim_size(1), " vs. ", rank));

    Tensor dense_shape(cpu_allocator(), DT_INT64, TensorShape({rank}));
    auto dense_shape_mutable = dense_shape.vec<int64_t>();
    for (int i = 0; i < rank; ++i) {
      dense_shape_mutable(i) = dense_tensor_shape.dim_size(i);
    }

    const int64_t batch_size = (rank == 2) ? 1 : dense_tensor_shape.dim_size(0);
    const int64_t num_rows = dense_tensor_shape.dim_size((rank == 2) ? 0 : 1);
    const int64_t total_nnz = indices.NumElements() / rank;

    Tensor values;
    OP_REQUIRES_OK(ctx, functor::DoGatherNd<Device, T, int64_t>(
                            ctx, params, indices, &values));

    Tensor batch_ptr(cpu_allocator(), DT_INT32, TensorShape({batch_size + 1}));
    Tensor csr_col_ind(cpu_allocator(), DT_INT32, TensorShape({total_nnz}));
    Tensor csr_row_ptr(cpu_allocator(), DT_INT32,
                       TensorShape({(num_rows + 1) * batch_size}));

    // Fill the row pointers with zeros.
    functor::SetZeroFunctor<Device, int32> set_zero;
    set_zero(ctx->eigen_device<Device>(), csr_row_ptr.flat<int32>());

    // Convert from COO to CSR format.
    functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
    OP_REQUIRES_OK(ctx,
                   coo_to_csr(batch_size, num_rows, indices.matrix<int64_t>(),
                              batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                              csr_col_ind.vec<int32>()));

    CSRSparseMatrix output_csr_matrix;
    OP_REQUIRES_OK(ctx, CSRSparseMatrix::CreateCSRSparseMatrix(
                            values.dtype(), dense_shape, batch_ptr, csr_row_ptr,
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

#define REGISTER_CPU(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("DenseToCSRSparseMatrix") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T"),   \
                          DenseToCSRSparseMatrixCPUOp<CPUDevice, T>);

REGISTER_CPU(float)
REGISTER_CPU(double)
REGISTER_CPU(complex64)
REGISTER_CPU(complex128)

#undef REGISTER_CPU

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class DenseToCSRSparseMatrixGPUOp : public AsyncOpKernel {
 public:
  explicit DenseToCSRSparseMatrixGPUOp(OpKernelConstruction* c)
      : AsyncOpKernel(c) {}

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    auto stream = c->op_device_context()->stream();
    const Device& d = c->eigen_device<Device>();

    const Tensor& params_t = c->input(0);
    const Tensor& indices_t = c->input(1);
    const TensorShape& dense_tensor_shape = params_t.shape();
    const int rank = params_t.dims();
    OP_REQUIRES_ASYNC(c, rank == 2 || rank == 3,
                      errors::InvalidArgument(
                          "params must have rank == 2 or 3; ",
                          "but saw shape: ", dense_tensor_shape.DebugString()),
                      done);
    OP_REQUIRES_ASYNC(
        c, indices_t.dims() == 2,
        errors::InvalidArgument("indices must be a matrix, but saw shape: ",
                                indices_t.shape().DebugString()),
        done);
    OP_REQUIRES_ASYNC(
        c, indices_t.dim_size(1) == rank,
        errors::InvalidArgument(
            "indices.shape[1] must be equal to the rank of params, but saw: ",
            indices_t.dim_size(1), " vs. ", rank),
        done);
    const int64_t batch_size = (rank == 2) ? 1 : dense_tensor_shape.dim_size(0);
    const int64_t rows = dense_tensor_shape.dim_size((rank == 2) ? 0 : 1);
    const int64_t cols = dense_tensor_shape.dim_size((rank == 2) ? 1 : 2);

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
          errors::Internal("DenseToSparseMatrixGPUOp: failed to copy "
                           "nnz_per_batch from device"),
          done);
    }

    // TODO(ebrevdo): write a custom pair of kernels: one that
    // calculates the batched csr_row_ptr vector, another that fills in
    // the col_ind and values vectors.
    TensorReference nnz_per_batch_device_ref(nnz_per_batch_device_t);
    auto convert_to_csr = [this, c, rank, batch_size, nnz_per_batch_host,
                           nnz_per_batch_device_ref, stream, &d, &params_t,
                           &indices_t, dense_tensor_shape, rows, cols, done]() {
      // The data has been copied out of the nnz_per_batch_device
      // tensor by the time we get here; we can unreference it.
      nnz_per_batch_device_ref.Unref();

      auto nnz_per_batch = nnz_per_batch_host.tensor().vec<int32>();

      // Ensure that within the callback, the proper GPU settings are
      // configured.
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      // Extract out the values.
      Tensor temp_values_t;
      OP_REQUIRES_OK_ASYNC(c,
                           (functor::DoGatherNd<Device, T, int64>(
                               c, params_t, indices_t, &temp_values_t)),
                           done);
      const Tensor& values_t = const_cast<const Tensor&>(temp_values_t);

      OP_REQUIRES_ASYNC(
          c, TensorShapeUtils::IsVector(values_t.shape()),
          errors::Internal("Expected values_t to be a vector, but saw shape: ",
                           values_t.shape().DebugString()),
          done);

      Tensor dense_shape_t(cpu_allocator(), DT_INT64, TensorShape({rank}));
      auto dense_shape_mutable = dense_shape_t.vec<int64_t>();
      for (int i = 0; i < rank; ++i) {
        dense_shape_mutable(i) = dense_tensor_shape.dim_size(i);
      }
      auto dense_shape =
          const_cast<const Tensor&>(dense_shape_t).vec<int64_t>();

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

#define REGISTER_GPU(DEV, T)                             \
  REGISTER_KERNEL_BUILDER(Name("DenseToCSRSparseMatrix") \
                              .Device(DEVICE_##DEV)      \
                              .TypeConstraint<T>("T"),   \
                          DenseToCSRSparseMatrixGPUOp<DEV##Device, T>);

REGISTER_GPU(GPU, float)
REGISTER_GPU(GPU, double)
REGISTER_GPU(GPU, complex64)
REGISTER_GPU(GPU, complex128)

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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_GPU

}  // namespace tensorflow
