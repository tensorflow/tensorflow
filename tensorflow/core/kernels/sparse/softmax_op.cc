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

// Implements the kernel for the CSRSoftmax op, which performs softmax
// along the innermost (col) dimension of a CSRSparseMatrix object
// stored in a DT_VARIANT.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_sparse.h"
#define EIGEN_USE_GPU
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRSoftmaxOp : public OpKernel {
 public:
  explicit CSRSoftmaxOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const CSRSparseMatrix* logits_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &logits_matrix));
    OP_REQUIRES(
        ctx, logits_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of logits is not equal to 'type': ",
                                DataTypeString(logits_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    // Allocate output shapes
    const int total_nnz = logits_matrix->total_nnz();
    Tensor output_values_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({total_nnz}), &output_values_t));

    CSRSparseMatrix output_matrix;

    Tensor dense_shape_t = logits_matrix->dense_shape();

    OP_REQUIRES_OK(
        ctx,
        CSRSparseMatrix::CreateCSRSparseMatrix(
            DataTypeToEnum<T>::value, dense_shape_t,
            logits_matrix->batch_pointers(), logits_matrix->row_pointers(),
            logits_matrix->col_indices(), output_values_t, &output_matrix));

    if (total_nnz > 0) {
      functor::CSRSparseMatrixSoftmax<Device, T> softmax;
      OP_REQUIRES_OK(
          ctx, softmax(ctx, *logits_matrix, output_matrix.values().vec<T>()));
    }

    Tensor output_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    output_t.scalar<Variant>()() = std::move(output_matrix);
    ctx->set_output(0, output_t);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(DEV, T)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSoftmax")     \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRSoftmaxOp<DEV##Device, T>);

REGISTER(GPU, float)
REGISTER(GPU, double)

#undef REGISTER

namespace functor {
#define DECLARE_GPU_SPEC(T)                                \
  template <>                                              \
  Status CSRSparseMatrixSoftmax<GPUDevice, T>::operator()( \
      OpKernelContext* ctx, const CSRSparseMatrix& logits, \
      typename TTypes<T>::Vec softmax_values);             \
  extern template struct CSRSparseMatrixSoftmax<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T>
class CSRSoftmaxGradOp : public OpKernel {
 public:
  explicit CSRSoftmaxGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const CSRSparseMatrix* softmax_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &softmax_matrix));
    OP_REQUIRES(ctx, softmax_matrix->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of softmax is not equal to 'type': ",
                    DataTypeString(softmax_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));

    const CSRSparseMatrix* grad_softmax_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 1, &grad_softmax_matrix));
    OP_REQUIRES(ctx, grad_softmax_matrix->dtype() == DataTypeToEnum<T>::value,
                errors::InvalidArgument(
                    "dtype of grad_softmax is not equal to 'type': ",
                    DataTypeString(grad_softmax_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));

    OP_REQUIRES(
        ctx, softmax_matrix->dims() == grad_softmax_matrix->dims(),
        errors::InvalidArgument(
            "Ranks of softmax and grad_softmax matrices differ: ",
            softmax_matrix->dims(), " vs. ", grad_softmax_matrix->dims()));

    OP_REQUIRES(
        ctx, softmax_matrix->dims() == grad_softmax_matrix->dims(),
        errors::InvalidArgument(
            "Ranks of softmax and grad_softmax matrices differ: ",
            softmax_matrix->dims(), " vs. ", grad_softmax_matrix->dims()));

    Tensor dense_shape_t = softmax_matrix->dense_shape();
    auto host_dense_shape =
        static_cast<const Tensor>(dense_shape_t).vec<int64_t>();

    auto host_grad_dense_shape =
        grad_softmax_matrix->dense_shape().vec<int64_t>();

    for (int i = 0; i < host_dense_shape.size(); ++i) {
      OP_REQUIRES(ctx, host_dense_shape(i) == host_grad_dense_shape(i),
                  errors::InvalidArgument(
                      "Shapes of softmax and grad_softmax matrices differ: ",
                      dense_shape_t.SummarizeValue(3), " vs. ",
                      grad_softmax_matrix->dense_shape().SummarizeValue(3)));
    }

    // Allocate output shapes.  Note that since the Softmax Gradient
    // tensor is the elementwise product of some function with the
    // softmax value, it will keep the sparsity structure of the softmax.
    const int total_nnz = softmax_matrix->total_nnz();
    Tensor gradient_values;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({total_nnz}), &gradient_values));

    CSRSparseMatrix gradient_matrix;

    OP_REQUIRES_OK(
        ctx,
        CSRSparseMatrix::CreateCSRSparseMatrix(
            DataTypeToEnum<T>::value, dense_shape_t,
            softmax_matrix->batch_pointers(), softmax_matrix->row_pointers(),
            softmax_matrix->col_indices(), gradient_values, &gradient_matrix));

    if (total_nnz > 0) {
      functor::CSRSparseMatrixSoftmaxGrad<Device, T> softmax_grad;
      OP_REQUIRES_OK(ctx,
                     softmax_grad(ctx, *softmax_matrix, *grad_softmax_matrix,
                                  gradient_matrix.values().vec<T>()));
    }

    Tensor gradient_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    gradient_t.scalar<Variant>()() = std::move(gradient_matrix);
    ctx->set_output(0, gradient_t);
  }
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(DEV, T)                                  \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixSoftmaxGrad") \
                              .Device(DEVICE_##DEV)       \
                              .TypeConstraint<T>("type"), \
                          CSRSoftmaxGradOp<DEV##Device, T>);

REGISTER(GPU, float)
REGISTER(GPU, double)

#undef REGISTER

namespace functor {
#define DECLARE_GPU_SPEC(T)                                    \
  template <>                                                  \
  Status CSRSparseMatrixSoftmaxGrad<GPUDevice, T>::operator()( \
      OpKernelContext* ctx, const CSRSparseMatrix& softmax,    \
      const CSRSparseMatrix& grad_softmax,                     \
      typename TTypes<T>::Vec gradient_values);                \
  extern template struct CSRSparseMatrixSoftmaxGrad<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
