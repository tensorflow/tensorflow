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
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRMulOp : public OpKernel {
 public:
  explicit CSRMulOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* a_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    const Tensor& b_t = ctx->input(1);

    OP_REQUIRES(ctx, a_matrix->dtype() == b_t.dtype(),
                errors::InvalidArgument(
                    "Input types don't match.  a.dtype == ",
                    DataTypeString(a_matrix->dtype()),
                    " vs. b.dtype == ", DataTypeString(b_t.dtype())));

    const int b_rank = b_t.dims();

    const Tensor& a_dense_shape_t = a_matrix->dense_shape();
    auto a_dense_shape = a_dense_shape_t.vec<int64>();
    const int batch_size = a_dense_shape(0);
    if (b_rank == 3) {
      OP_REQUIRES(
          ctx,
          ((a_matrix->dims() == 3) && (b_t.dim_size(0) == batch_size) &&
           (b_t.NumElements() == batch_size)),
          errors::InvalidArgument(
              "If b is a rank-3 tensor, then a must be a rank 3 and the size "
              "of b be "
              "[batch_size, 1, 1].  But the shape of b is: ",
              b_t.shape().DebugString(),
              " and the shape of a is: ", a_dense_shape_t.DebugString()));
    } else {
      OP_REQUIRES(ctx, b_rank == 0,
                  errors::Unimplemented(
                      "Multiplying by a 2D+ dense tensor is not currently "
                      "supported, but shape of b is: ",
                      b_t.shape().DebugString()));
    }

    Tensor c_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    CSRSparseMatrix c_matrix;
    if (b_rank == 0) {
      auto b = b_t.scalar<T>();
      // TODO(ebrevdo): call other functor if b is nonscalar.
      functor::CSRSparseMatrixMulScalar<Device, T> csrmul_scalar;
      OP_REQUIRES_OK(ctx, csrmul_scalar.Compute(ctx, *a_matrix, b, &c_matrix));
    } else {
      // b_rank == 1 and a_matrix is rank-3.
      auto b = b_t.flat<T>();
      functor::CSRSparseMatrixBatchMulVec<Device, T> csrmul_batch_vec;
      OP_REQUIRES_OK(ctx,
                     csrmul_batch_vec.Compute(ctx, *a_matrix, b, &c_matrix));
    }
    c_t.scalar<Variant>()() = std::move(c_matrix);
    ctx->set_output(0, c_t);
  }
};

#define REGISTER(DEV, T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("SparseMatrixMul").Device(DEVICE_##DEV).TypeConstraint<T>("T"), \
      CSRMulOp<DEV##Device, T>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(T) REGISTER(GPU, T)

REGISTER_GPU(float)
REGISTER_GPU(double)
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T>
class CSRSparseMatrixMulScalar<GPUDevice, T> {
 public:
  explicit CSRSparseMatrixMulScalar() {}

  Status Compute(OpKernelContext* ctx, const CSRSparseMatrix& a,
                 typename TTypes<T>::ConstScalar b, CSRSparseMatrix* c) {
    const int total_nnz = a.total_nnz();
    Tensor c_values_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({total_nnz}), &c_values_t));
    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(),
        a.row_pointers(), a.col_indices(), c_values_t, c));

    auto a_values = a.values().flat<T>();
    auto c_values = c_values_t.flat<T>();

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();
    bool error;
    bool* const error_ptr = functor::mul<T>::has_errors ? &error : nullptr;

    // tensor * scalar
    functor::BinaryFunctor<GPUDevice, functor::mul<T>, 1>().Right(
        d, c_values, a_values, b, error_ptr);

    return Status::OK();
  }
};

#define DECLARE_GPU_SPEC(T)                                 \
  template <>                                               \
  Status CSRSparseMatrixBatchMulVec<GPUDevice, T>::Compute( \
      OpKernelContext* ctx, const CSRSparseMatrix& a,       \
      typename TTypes<T>::ConstFlat b, CSRSparseMatrix* c); \
  extern template struct CSRSparseMatrixBatchMulVec<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
DECLARE_GPU_SPEC(std::complex<float>);
DECLARE_GPU_SPEC(std::complex<double>);

#undef DECLARE_GPU_SPEC

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
