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
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
template <typename Device, typename T>
class CSRSparseMatrixConjFunctor {
 public:
  explicit CSRSparseMatrixConjFunctor(OpKernelContext* ctx) : ctx_(ctx) {}

  Status operator()(const CSRSparseMatrix& a, CSRSparseMatrix* b) {
    const int total_nnz = a.total_nnz();
    Tensor b_values_t;
    TF_RETURN_IF_ERROR(ctx_->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({total_nnz}), &b_values_t));
    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(),
        a.row_pointers(), a.col_indices(), b_values_t, b));

    const Device& d = ctx_->eigen_device<Device>();
    functor::UnaryFunctor<Device, functor::conj<T>> func;
    func(d, b->values().flat<T>() /*out*/, a.values().flat<T>() /*in*/);

    return Status::OK();
  }

 private:
  OpKernelContext* ctx_;
};

// Partial specialization for real types where conjugation is a noop.
#define NOOP_CONJ_FUNCTOR(T)                                             \
  template <typename Device>                                             \
  class CSRSparseMatrixConjFunctor<Device, T> {                          \
   public:                                                               \
    explicit CSRSparseMatrixConjFunctor(OpKernelContext* ctx) {}         \
    Status operator()(const CSRSparseMatrix& a, CSRSparseMatrix* b) {    \
      TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(         \
          DataTypeToEnum<T>::value, a.dense_shape(), a.batch_pointers(), \
          a.row_pointers(), a.col_indices(), a.values(), b));            \
      return Status::OK();                                               \
    }                                                                    \
  };

NOOP_CONJ_FUNCTOR(float);
NOOP_CONJ_FUNCTOR(double);

#undef NOOP_CONJ_FUNCTOR

}  // namespace

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    CONJ_VARIANT_UNARY_OP, DEVICE_CPU, CSRSparseMatrix,
    (CSRSparseMatrixUnaryHelper<CPUDevice, CSRSparseMatrixConjFunctor>));

#if GOOGLE_CUDA

REGISTER_UNARY_VARIANT_UNARY_OP_FUNCTION(
    CONJ_VARIANT_UNARY_OP, DEVICE_GPU, CSRSparseMatrix,
    (CSRSparseMatrixUnaryHelper<GPUDevice, CSRSparseMatrixConjFunctor>));

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
