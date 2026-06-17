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

#include "tensorflow/core/kernels/sparse/zeros_op.h"

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

template <typename Device>
class CSRZerosOp : public OpKernel {
 public:
  explicit CSRZerosOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("type", &dtype_));
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& dense_shape_t = c->input(0);
    CSRSparseMatrix matrix;
    functor::CSRSparseMatrixZeros<Device> csr_sparse_matrix_zeros;
    OP_REQUIRES_OK(c,
                   csr_sparse_matrix_zeros(c, dtype_, dense_shape_t, &matrix));
    Tensor* matrix_t;
    AllocatorAttributes cpu_alloc;
    cpu_alloc.set_on_host(true);
    OP_REQUIRES_OK(
        c, c->allocate_output(0, TensorShape({}), &matrix_t, cpu_alloc));
    matrix_t->scalar<Variant>()() = matrix;
  }

 private:
  DataType dtype_;
};

namespace {

template <typename Device>
absl::Status CSRSparseMatrixZerosLikeHelper(OpKernelContext* ctx,
                                            const CSRSparseMatrix& x,
                                            CSRSparseMatrix* y) {
  functor::CSRSparseMatrixZeros<Device> csr_sparse_matrix_zeros;
  return csr_sparse_matrix_zeros(ctx, x.dtype(), x.dense_shape(), y);
}

}  // namespace

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER(DEV)                                     \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixZeros")       \
                              .Device(DEVICE_##DEV)       \
                              .HostMemory("dense_shape"), \
                          CSRZerosOp<DEV##Device>);

REGISTER(GPU)

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    ZEROS_LIKE_VARIANT_UNARY_OP, DEVICE_GPU, CSRSparseMatrix,
    CSRSparseMatrixZerosLikeHelper<GPUDevice>);

#undef REGISTER
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
