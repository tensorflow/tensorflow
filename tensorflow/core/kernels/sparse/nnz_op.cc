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
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/cuda_solvers.h"
#include "tensorflow/core/util/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class CSRNNZOp : public OpKernel {
 public:
  explicit CSRNNZOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));
    Tensor* nnz_t;
    TensorShape nnz_shape;
    if (csr_sparse_matrix->dims() == 3) {
      nnz_shape.AddDim(csr_sparse_matrix->batch_size());
    }
    OP_REQUIRES_OK(c, c->allocate_output(0, nnz_shape, &nnz_t));
    auto nnz = nnz_t->flat<int32>();
    for (int i = 0; i < csr_sparse_matrix->batch_size(); ++i) {
      nnz(i) = csr_sparse_matrix->nnz(i);
    }
  }
};

#define REGISTER(DEV)                                      \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixNNZ")          \
                              .Device(DEVICE_##DEV)        \
                              .HostMemory("nnz"),          \
                          CSRNNZOp<DEV##Device>);

REGISTER(CPU)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER(GPU)

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER

}  // namespace tensorflow
