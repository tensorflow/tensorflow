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
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class CSRSparseMatrixComponentsOp : public OpKernel {
 public:
  explicit CSRSparseMatrixComponentsOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* c) final {
    const CSRSparseMatrix* csr_sparse_matrix;
    OP_REQUIRES_OK(c, ExtractVariantFromInput(c, 0, &csr_sparse_matrix));

    const Tensor& index_t = c->input(1);
    OP_REQUIRES(c, DataTypeToEnum<T>::value == csr_sparse_matrix->dtype(),
                errors::InvalidArgument(
                    "dtype of input is not equal to 'type': ",
                    DataTypeString(csr_sparse_matrix->dtype()), " vs. ",
                    DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(c, index_t.dims() == 0,
                errors::InvalidArgument("index should be a scalar, but saw: ",
                                        index_t.DebugString()));
    int32 index = index_t.scalar<int32>()();
    OP_REQUIRES(c, index >= 0 && index < csr_sparse_matrix->batch_size(),
                errors::InvalidArgument("index (", index, ") not in [0, ",
                                        csr_sparse_matrix->batch_size(), ")"));

    if (csr_sparse_matrix->dims() == 2) {
      c->set_output(0, csr_sparse_matrix->row_pointers());
      c->set_output(1, csr_sparse_matrix->col_indices());
      c->set_output(2, csr_sparse_matrix->values());
    } else {
      auto batch_ptrs = csr_sparse_matrix->batch_pointers().vec<int32>();
      auto dense_shape = csr_sparse_matrix->dense_shape().vec<int64>();
      int64 rows = dense_shape(1);
      int nnz = batch_ptrs(index + 1) - batch_ptrs(index);
      Tensor* row_ptrs_t;
      Tensor* col_inds_t;
      Tensor* values_t;
      OP_REQUIRES_OK(
          c, c->allocate_output(0, TensorShape({rows + 1}), &row_ptrs_t));
      OP_REQUIRES_OK(c, c->allocate_output(1, TensorShape({nnz}), &col_inds_t));
      OP_REQUIRES_OK(c, c->allocate_output(2, TensorShape({nnz}), &values_t));
      auto row_ptrs = row_ptrs_t->vec<int32>();
      auto col_inds = col_inds_t->vec<int32>();
      auto values = values_t->vec<T>();

      functor::Slice<Device, int32, 1> slice_int;
      functor::Slice<Device, T, 1> slice_t;
      typedef Eigen::DSizes<Eigen::DenseIndex, 1> EVec;
      const Device& d = c->eigen_device<Device>();
      slice_int(d,
                /*output*/ row_ptrs,
                /*input*/ csr_sparse_matrix->row_pointers().vec<int32>(),
                /*slice_indices*/ EVec{index * (rows + 1)},
                /*slice_sizes*/ EVec{rows + 1});
      slice_int(d,
                /*output*/ col_inds,
                /*input*/ csr_sparse_matrix->col_indices().vec<int32>(),
                /*slice_indices*/ EVec{batch_ptrs(index)},
                /*slice_sizes*/ EVec{nnz});
      slice_t(d,
              /*output*/ values, /*input*/ csr_sparse_matrix->values().vec<T>(),
              /*slice_indices*/ EVec{batch_ptrs(index)},
              /*slice_sizes*/ EVec{nnz});
    }
  }
};

#define REGISTER(DEV, T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CSRSparseMatrixComponents") \
                              .Device(DEVICE_##DEV)         \
                              .TypeConstraint<T>("type")    \
                              .HostMemory("index"),         \
                          CSRSparseMatrixComponentsOp<DEV##Device, T>);

REGISTER(CPU, float)
REGISTER(CPU, double)
REGISTER(CPU, complex64)
REGISTER(CPU, complex128)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

REGISTER(GPU, float)
REGISTER(GPU, double)
#if GOOGLE_CUDA
REGISTER(GPU, complex64)
REGISTER(GPU, complex128)
#endif

#undef REGISTER

namespace functor {
// TODO(ebrevdo): This should move to a slice_functor.cc
#define DECLARE_GPU_SPEC(T)                                     \
  template <>                                                   \
  void Slice<GPUDevice, T, 1>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 1>::Tensor output, \
      typename TTypes<T, 1>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, 1>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, 1>& sizes);        \
  extern template struct Slice<GPUDevice, T, 1>;

DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#if GOOGLE_CUDA
DECLARE_GPU_SPEC(complex64);
DECLARE_GPU_SPEC(complex128);
#endif

#undef DECLARE_GPU_SPEC
}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
