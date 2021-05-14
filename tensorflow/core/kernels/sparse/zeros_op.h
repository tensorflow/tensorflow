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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_ZEROS_OP_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_ZEROS_OP_H_

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
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename Device>
struct CSRSparseMatrixZeros {
  Status operator()(OpKernelContext* c, DataType dtype,
                    const Tensor& dense_shape_t, CSRSparseMatrix* matrix) {
    auto dense_shape = dense_shape_t.vec<int64>();
    const int rank = dense_shape.size();
    if (!(rank == 2 || rank == 3)) {
      return errors::InvalidArgument("sparse tensor must have rank == 2 or 3; ",
                                     "but dense shape has ", rank, " entries");
    }
    const int64 batch_size = (rank == 2) ? 1 : dense_shape(0);
    const int64 rows = dense_shape((rank == 2) ? 0 : 1);

    Tensor batch_ptr_t(cpu_allocator(), DT_INT32,
                       TensorShape({batch_size + 1}));
    batch_ptr_t.vec<int32>().setZero();  // On host.

    Allocator* allocator = c->device()->GetAllocator(AllocatorAttributes());
    // An all-zeros CSR matrix is composed of an empty set of column
    // indices, an empty set of values, and a vector of all zero row
    // pointers.  The length of the row pointers vector is #rows + 1.
    // Each row pointer is just an offset into the cols and
    // values vectors, and those are empty, all coefficients are zero.
    Tensor csr_row_ptr_t;
    Tensor coo_col_ind_t(allocator, DT_INT32, TensorShape({0}));
    Tensor csr_values_t(allocator, dtype, TensorShape({0}));
    const Device& d = c->eigen_device<Device>();
    functor::SetZeroFunctor<Device, int32> set_zero;
    TF_RETURN_IF_ERROR(c->allocate_temp(
        DT_INT32, TensorShape({batch_size * (rows + 1)}), &csr_row_ptr_t));
    set_zero(d, csr_row_ptr_t.flat<int32>());

    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        dtype, dense_shape_t, batch_ptr_t, csr_row_ptr_t, coo_col_ind_t,
        csr_values_t, matrix));

    return Status::OK();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_ZEROS_OP_H_
