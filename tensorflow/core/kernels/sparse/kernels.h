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

#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_KERNELS_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_KERNELS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace functor {

// Calculates number of nonzero entries per batch of a sorted rank-3
// SparseTensor's indices.  indices is expected to have columns
// corresponding to [batch, row, column],  where indices[:,0] < B.
//
// REQUIRES:
//  indices.dimension(1) == 3
//  nnz_per_batch.dimension(0) == B
template <typename Device>
struct CalculateNNZPerBatchMatrixFromIndices {
  Status operator()(OpKernelContext* c, TTypes<int64_t>::ConstMatrix indices,
                    TTypes<int32>::Vec nnz_per_batch);
};

// Split a subset of a SparseTensors' indices into two vectors:
// COO row inds and COO col inds.  Outputs are:
//
//   coo_row_ind = indices[:, row_dim]
//   coo_col_ind = indices[:, row_dim + 1]
//
// where n = coo_row_ind.size()
// and row_dim = #cols(indices) - 1
//
// REQUIRES:
//   host_dense_shape.size() in [2, 3]
//   indices.dim_size(1) == host_dense_shape.size()
//   coo_row_ind.size() == coo_col_ind.size()
//   coo_row_ind.size() == indices.dim_size(0)
template <typename Device>
struct SparseTensorToCOOSparseMatrix {
  void operator()(const Device& d, TTypes<int64_t>::ConstVec host_dense_shape,
                  TTypes<int64_t>::ConstMatrix indices,
                  TTypes<int32>::Vec coo_row_ind,
                  TTypes<int32>::Vec coo_col_ind);
};

// Write coo batch, row, and column vectors to output matrix indices:
//
//   indices[:, row_dim] = coo_row_ind
//   indices[:, col_dim] = coo_col_ind
//
// where row_dim = #cols(indices) - 1 and n = coo_row_ind.size().
// In addition, if #cols(indices) == 3, also store the batch:
//
//   indices[i, 0] = batch_of(i) where
//      host_batch_ptrs(batch_of(i)) <= i < host_batch_ptrs(batch_of(i) + 1)
//
// REQUIRES:
//
//   host_dense_shape.size() in [2, 3]
//   indices.dim_size(1) == host_dense_shape.size()
//   host_batch_ptr.size() ==
//   coo_row_ind.size() == coo_col_ind.size()
//
template <typename Device>
struct COOSparseMatrixToSparseTensor {
  Status operator()(OpKernelContext* c,
                    TTypes<int64_t>::ConstVec host_dense_shape,
                    TTypes<int32>::ConstVec host_batch_ptrs,
                    TTypes<int32>::Vec coo_row_ind,
                    TTypes<int32>::ConstVec coo_col_ind,
                    TTypes<int64_t>::Matrix indices);
};

// Convert a vector of coo row indices to csr row pointers.
//
// REQUIRES:
//
//   csr_row_ptr.size() == rows + 1.
//   max(coo_row_ptr) < rows.
//
template <typename Device>
struct COOSparseMatrixToCSRSparseMatrix {
  Status operator()(OpKernelContext* c, const int rows, const int cols,
                    TTypes<int32>::UnalignedVec coo_row_ind,
                    TTypes<int32>::UnalignedVec csr_row_ptr);
};

// Convert a matrix of (batched) coo row and column indices to CSR SparseMatrix
// batch ptrs, csr row pointers and coo column indices.
//
// REQUIRES:
//   batch_ptr.size() == batch_size + 1
//   csr_row_ptr.size() == batch_size * (num_rows + 1)
//   csr_col_ind.size() == total_nnz
//   batch_size == 1 if rank == 2
//
//   where
//     total_nnz = indices.dim_size(0)
//     rank = indices.dim_size(1)
//   Also csr_row_ptr should be initially filled with zeros.
//
struct SparseTensorToCSRSparseMatrixCPUFunctor {
  Status operator()(const int64_t batch_size, const int num_rows,
                    TTypes<int64_t>::ConstMatrix indices,
                    TTypes<int32>::Vec batch_ptr,
                    TTypes<int32>::Vec csr_row_ptr,
                    TTypes<int32>::Vec csr_col_ind);
};

// Convert a vector of csr row pointers to coo row indices.
//
// REQUIRES:
//
//   coo_row_ptr.size() == nnz.
//   csr_row_ptr[-1] == nnz.
//
template <typename Device>
struct CSRSparseMatrixToCOOSparseMatrix {
  Status operator()(OpKernelContext* c,
                    TTypes<int32>::UnalignedConstVec csr_row_ptr,
                    TTypes<int32>::UnalignedVec coo_row_ind);
};

// Calculates C = matmul(A, B) or C = matmul(A, B)^T, where A is in CSR format
// and B and C are dense.
template <typename Device, typename T>
struct CSRSparseMatrixMatMul {
  explicit CSRSparseMatrixMatMul(const bool transpose_output);
  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 typename TTypes<T>::ConstMatrix b,
                 typename TTypes<T>::Matrix c);
};

// Calculates y = A * x, y = A^T * x, or y = A^H * x, where A is in CSR format
// and x and y are dense vectors.
template <typename Device, typename T>
class CSRSparseMatrixMatVec {
  CSRSparseMatrixMatVec(bool transpose_a, bool adjoint_a);
  Status Compute(OpKernelContext* ctx, const ConstCSRComponent<T>& a,
                 const T* x, T* y);
};

// Calculates C = functor(A, B) where A and B are CSR and C is CSR
// with a different sparsity pattern.
template <typename Device, typename T>
struct CSRStructureModifyingFunctor {
  virtual ~CSRStructureModifyingFunctor() {}

  virtual Status Initialize() = 0;

  virtual Status GetWorkspaceSize(const ConstCSRComponent<T>& a,
                                  const ConstCSRComponent<T>& b,
                                  size_t* bufferSize) = 0;

  virtual Status GetOutputStructure(const ConstCSRComponent<T>& a,
                                    const ConstCSRComponent<T>& b,
                                    TTypes<int32>::UnalignedVec c_row_ptr,
                                    int* output_nnz, void* workspace) = 0;

  virtual Status Compute(const ConstCSRComponent<T>& a,
                         const ConstCSRComponent<T>& b, CSRComponent<T>* c,
                         void* workspace) = 0;
};

// Calculates C = alpha * A + beta * B, where A and B are in CSR
// format, and alpha and beta are scalars on the host.
template <typename Device, typename T>
struct CSRSparseMatrixAdd : public CSRStructureModifyingFunctor<Device, T> {
  explicit CSRSparseMatrixAdd(OpKernelContext* ctx, const T alpha,
                              const T beta);
};

// Calculates C = matmul(A, B), where A, B, and C are in CSR format.
template <typename Device, typename T>
struct CSRSparseSparseMatrixMatMul
    : public CSRStructureModifyingFunctor<Device, T> {
  explicit CSRSparseSparseMatrixMatMul(OpKernelContext* ctx, bool transpose_a,
                                       bool transpose_b);
};

// Calculates Y = transpose(X) where X and Y are CSR format components.
template <typename Device, typename T>
struct CSRSparseMatrixTransposeComponent {
  Status operator()(OpKernelContext* ctx, const ConstCSRComponent<T>& x,
                    CSRComponent<T>* y);
};

// Calculates Y = transpose(X) where X and Y are in CSR format.
template <typename Device, typename T>
struct CSRSparseMatrixTranspose {
  Status operator()(OpKernelContext* ctx, bool conjugate,
                    const CSRSparseMatrix& input_matrix,
                    CSRSparseMatrix* output_matrix);
};

// Calculates Y = softmax(X) where X and Y are in CSR format;
// missing coefficients in X are treates as -inf (logits of 0 probability).
template <typename Device, typename T>
struct CSRSparseMatrixSoftmax {
  Status operator()(OpKernelContext* ctx, const CSRSparseMatrix& logits,
                    typename TTypes<T>::Vec softmax_values);
};

template <typename Device, typename T>
struct CSRSparseMatrixSoftmaxGrad {
  Status operator()(OpKernelContext* ctx, const CSRSparseMatrix& softmax,
                    const CSRSparseMatrix& grad_softmax,
                    typename TTypes<T>::Vec gradient_values);
};

template <typename Device, typename T>
class CSRSparseMatrixMulScalar {
 public:
  explicit CSRSparseMatrixMulScalar() {}

  Status Compute(OpKernelContext* ctx, const CSRSparseMatrix& a,
                 typename TTypes<T>::ConstScalar b, CSRSparseMatrix* c);
};

template <typename Device, typename T>
class CSRSparseMatrixBatchMulVec {
 public:
  explicit CSRSparseMatrixBatchMulVec() {}

  Status Compute(OpKernelContext* ctx, const CSRSparseMatrix& a,
                 typename TTypes<T>::ConstFlat b, CSRSparseMatrix* c);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_KERNELS_H_
