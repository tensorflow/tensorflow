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

#ifndef TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_
#define TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_

// This header declares the class CudaSparse, which contains wrappers of
// cuSparse libraries for use in TensorFlow kernels.

#ifdef GOOGLE_CUDA

#include <functional>
#include <vector>

#include "third_party/gpus/cuda/include/cusparse.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/public/version.h"

// Macro that specializes a sparse method for all 4 standard
// numeric types.
// TODO: reuse with cuda_solvers
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

namespace tensorflow {

inline string ConvertCUSparseErrorToString(const cusparseStatus_t status) {
  switch (status) {
#define STRINGIZE(q) #q
#define RETURN_IF_STATUS(err) \
  case err:                   \
    return STRINGIZE(err);

    RETURN_IF_STATUS(CUSPARSE_STATUS_SUCCESS)
    RETURN_IF_STATUS(CUSPARSE_STATUS_NOT_INITIALIZED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ALLOC_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INVALID_VALUE)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ARCH_MISMATCH)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MAPPING_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_EXECUTION_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INTERNAL_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)

#undef RETURN_IF_STATUS
#undef STRINGIZE
    default:
      return strings::StrCat("Unknown CUSPARSE error: ",
                             static_cast<int>(status));
  }
}

#define TF_RETURN_IF_CUSPARSE_ERROR(expr)                                  \
  do {                                                                     \
    auto status = (expr);                                                  \
    if (TF_PREDICT_FALSE(status != CUSPARSE_STATUS_SUCCESS)) {             \
      return errors::Internal(__FILE__, ":", __LINE__, " (", TF_STR(expr), \
                              "): cuSparse call failed with status ",      \
                              ConvertCUSparseErrorToString(status));       \
    }                                                                      \
  } while (0)

inline cusparseOperation_t TransposeAndConjugateToCuSparseOp(bool transpose,
                                                             bool conjugate,
                                                             Status* status) {
  if (transpose) {
    return conjugate ? CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                     : CUSPARSE_OPERATION_TRANSPOSE;
  } else {
    if (conjugate) {
      DCHECK(status != nullptr);
      *status = errors::InvalidArgument(
          "Conjugate == True and transpose == False is not supported.");
    }
    return CUSPARSE_OPERATION_NON_TRANSPOSE;
  }
}

// The CudaSparse class provides a simplified templated API for cuSparse
// (http://docs.nvidia.com/cuda/cusparse/index.html).
// An object of this class wraps static cuSparse instances,
// and will launch Cuda kernels on the stream wrapped by the GPU device
// in the OpKernelContext provided to the constructor.
//
// Notice: All the computational member functions are asynchronous and simply
// launch one or more Cuda kernels on the Cuda stream wrapped by the CudaSparse
// object.

class CudaSparse {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit CudaSparse(OpKernelContext* context);
  virtual ~CudaSparse() {}

  // This initializes the CudaSparse class if it hasn't
  // been initialized yet.  All following public methods require the
  // class has been initialized.  Can be run multiple times; all
  // subsequent calls after the first have no effect.
  Status Initialize();  // Move to constructor?

  // ====================================================================
  // Wrappers for cuSparse start here.
  //

  // Solves tridiagonal system of equations.
  // Note: Cuda Toolkit 9.0+ has better-performing gtsv2 routine. gtsv will be
  // removed in Cuda Toolkit 11.0.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-gtsv
  // Returns Status::OK() if the kernel was launched successfully.
  template <typename Scalar>
  Status Gtsv(int m, int n, const Scalar *dl, const Scalar *d, const Scalar *du,
              Scalar *B, int ldb) const;

  // Solves tridiagonal system of equations without pivoting.
  // Note: Cuda Toolkit 9.0+ has better-performing gtsv2_nopivot routine.
  // gtsv_nopivot will be removed in Cuda Toolkit 11.0.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-gtsv_nopivot
  // Returns Status::OK() if the kernel was launched successfully.
  template <typename Scalar>
  Status GtsvNoPivot(int m, int n, const Scalar *dl, const Scalar *d,
                     const Scalar *du, Scalar *B, int ldb) const;

  // Solves a batch of tridiagonal systems of equations. Doesn't support
  // multiple right-hand sides per each system. Doesn't do pivoting.
  // Note: Cuda Toolkit 9.0+ has better-performing gtsv2StridedBatch routine.
  // gtsvStridedBatch will be removed in Cuda Toolkit 11.0.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-gtsvstridedbatch
  // Returns Status::OK() if the kernel was launched successfully.
  template <typename Scalar>
  Status GtsvStridedBatch(int m, const Scalar *dl, const Scalar *d,
                          const Scalar *du, Scalar *x, int batchCount,
                          int batchStride) const;

  // Solves tridiagonal system of equations.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2
  template <typename Scalar>
  Status Gtsv2(int m, int n, const Scalar *dl, const Scalar *d,
               const Scalar *du, Scalar *B, int ldb, void *pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_bufferSize
  template <typename Scalar>
  Status Gtsv2BufferSizeExt(int m, int n, const Scalar *dl, const Scalar *d,
                            const Scalar *du, const Scalar *B, int ldb,
                            size_t *bufferSizeInBytes) const;

  // Solves tridiagonal system of equations without partial pivoting.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot
  template <typename Scalar>
  Status Gtsv2NoPivot(int m, int n, const Scalar *dl, const Scalar *d,
                      const Scalar *du, Scalar *B, int ldb,
                      void *pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2NoPivot.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot_bufferSize
  template <typename Scalar>
  Status Gtsv2NoPivotBufferSizeExt(int m, int n, const Scalar *dl,
                                   const Scalar *d, const Scalar *du,
                                   const Scalar *B, int ldb,
                                   size_t *bufferSizeInBytes) const;

  // Solves a batch of tridiagonal systems of equations. Doesn't support
  // multiple right-hand sides per each system. Doesn't do pivoting.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch
  template <typename Scalar>
  Status Gtsv2StridedBatch(int m, const Scalar *dl, const Scalar *d,
                           const Scalar *du, Scalar *x, int batchCount,
                           int batchStride, void *pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2StridedBatch.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch_bufferSize
  template <typename Scalar>
  Status Gtsv2StridedBatchBufferSizeExt(int m, const Scalar *dl,
                                        const Scalar *d, const Scalar *du,
                                        const Scalar *x, int batchCount,
                                        int batchStride,
                                        size_t *bufferSizeInBytes) const;

  // Compresses the indices of rows or columns. It can be interpreted as a
  // conversion from COO to CSR sparse storage format. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csr2coo.
  Status Csr2coo(const int* CsrRowPtr, int nnz, int m, int* cooRowInd) const;

  // Uncompresses the indices of rows or columns. It can be interpreted as a
  // conversion from CSR to COO sparse storage format. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr.
  Status Coo2csr(const int* cooRowInd, int nnz, int m, int* csrRowPtr) const;

  // Sparse-dense matrix multiplication C = alpha * op(A) * op(B)  + beta * C,
  // where A is a sparse matrix in CSR format, B and C are dense tall
  // matrices.  This routine allows transposition of matrix B, which
  // may improve performance.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmm2
  //
  // **NOTE** Matrices B and C are expected to be in column-major
  // order; to make them consistent with TensorFlow they
  // must be transposed (or the matmul op's pre/post-procesisng must take this
  // into account).
  //
  // **NOTE** This is an in-place operation for data in C.
  template <typename Scalar>
  Status Csrmm(cusparseOperation_t transA, cusparseOperation_t transB, int m,
               int n, int k, int nnz, const Scalar* alpha_host,
               const cusparseMatDescr_t descrA, const Scalar* csrSortedValA,
               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
               const Scalar* B, int ldb, const Scalar* beta_host, Scalar* C,
               int ldc) const;

  // Sparse-dense vector multiplication y = alpha * op(A) * x  + beta * y,
  // where A is a sparse matrix in CSR format, x and y are dense vectors. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv_mergepath
  //
  // **NOTE** This is an in-place operation for data in y.
  template <typename Scalar>
  Status Csrmv(cusparseOperation_t transA, int m, int n, int nnz,
               const Scalar* alpha_host, const cusparseMatDescr_t descrA,
               const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
               const int* csrSortedColIndA, const Scalar* x,
               const Scalar* beta_host, Scalar* y) const;

  // Computes sparse-sparse matrix addition of matrices
  // stored in CSR format.  This is part one: calculate nnz of the
  // output.  csrSortedRowPtrC must be preallocated on device with
  // m + 1 entries.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgeam.
  Status CsrgeamNnz(int m, int n, const cusparseMatDescr_t descrA, int nnzA,
                    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                    const cusparseMatDescr_t descrB, int nnzB,
                    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                    const cusparseMatDescr_t descrC, int* csrSortedRowPtrC,
                    int* nnzTotalDevHostPtr);

  // Computes sparse - sparse matrix addition of matrices
  // stored in CSR format.  This is part two: perform sparse-sparse
  // addition.  csrValC and csrColIndC must be allocated on the device
  // with nnzTotalDevHostPtr entries (as calculated by CsrgeamNnz).  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgeam.
  template <typename Scalar>
  Status Csrgeam(int m, int n, const Scalar* alpha,
                 const cusparseMatDescr_t descrA, int nnzA,
                 const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
                 const int* csrSortedColIndA, const Scalar* beta,
                 const cusparseMatDescr_t descrB, int nnzB,
                 const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
                 const int* csrSortedColIndB, const cusparseMatDescr_t descrC,
                 Scalar* csrSortedValC, int* csrSortedRowPtrC,
                 int* csrSortedColIndC);

  // Computes sparse-sparse matrix multiplication of matrices
  // stored in CSR format.  This is part one: calculate nnz of the
  // output.  csrSortedRowPtrC must be preallocated on device with
  // m + 1 entries.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm.
  Status CsrgemmNnz(cusparseOperation_t transA, cusparseOperation_t transB,
                    int m, int k, int n, const cusparseMatDescr_t descrA,
                    int nnzA, const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const cusparseMatDescr_t descrB, int nnzB,
                    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                    const cusparseMatDescr_t descrC, int* csrSortedRowPtrC,
                    int* nnzTotalDevHostPtr);

  // Computes sparse - sparse matrix matmul of matrices
  // stored in CSR format.  This is part two: perform sparse-sparse
  // addition.  csrValC and csrColIndC must be allocated on the device
  // with nnzTotalDevHostPtr entries (as calculated by CsrgemmNnz).  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm.
  template <typename Scalar>
  Status Csrgemm(cusparseOperation_t transA, cusparseOperation_t transB, int m,
                 int k, int n, const cusparseMatDescr_t descrA, int nnzA,
                 const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
                 const int* csrSortedColIndA, const cusparseMatDescr_t descrB,
                 int nnzB, const Scalar* csrSortedValB,
                 const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                 const cusparseMatDescr_t descrC, Scalar* csrSortedValC,
                 int* csrSortedRowPtrC, int* csrSortedColIndC);

  // In-place reordering of unsorted CSR to sorted CSR.
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csru2csr
  template <typename Scalar>
  Status Csru2csr(int m, int n, int nnz, const cusparseMatDescr_t descrA,
                  Scalar* csrVal, const int* csrRowPtr, int* csrColInd);

  // Converts from CSR to CSC format (equivalently, transpose).
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-csr2cscEx
  template <typename Scalar>
  Status Csr2csc(int m, int n, int nnz, const Scalar* csrVal,
                 const int* csrRowPtr, const int* csrColInd, Scalar* cscVal,
                 int* cscRowInd, int* cscColPtr,
                 const cusparseAction_t copyValues);

 private:
  bool initialized_;
  OpKernelContext *context_;  // not owned.
  cudaStream_t cuda_stream_;
  cusparseHandle_t *cusparse_handle_;  // not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparse);
};

// A wrapper class to ensure that a CUDA sparse matrix descriptor is initialized
// only once. For more details on the descriptor (cusparseMatDescr_t), see:
// https://docs.nvidia.com/cuda/cusparse/index.html#cusparsematdescrt
class CudaSparseMatrixDescriptor {
 public:
  explicit CudaSparseMatrixDescriptor() : initialized_(false) {}

  CudaSparseMatrixDescriptor(CudaSparseMatrixDescriptor&& rhs)
      : initialized_(rhs.initialized_), descr_(std::move(rhs.descr_)) {
    rhs.initialized_ = false;
  }

  CudaSparseMatrixDescriptor& operator=(CudaSparseMatrixDescriptor&& rhs) {
    if (this == &rhs) return *this;
    Release();
    initialized_ = rhs.initialized_;
    descr_ = std::move(rhs.descr_);
    rhs.initialized_ = false;
    return *this;
  }

  ~CudaSparseMatrixDescriptor() { Release(); }

  // Initializes the underlying descriptor.  Will fail on the second call if
  // called more than once.
  Status Initialize() {
    DCHECK(!initialized_);
    TF_RETURN_IF_CUSPARSE_ERROR(cusparseCreateMatDescr(&descr_));
    initialized_ = true;
    return Status::OK();
  }

  cusparseMatDescr_t& descr() {
    DCHECK(initialized_);
    return descr_;
  }

  const cusparseMatDescr_t& descr() const {
    DCHECK(initialized_);
    return descr_;
  }

 private:
  void Release() {
    if (initialized_) {
      cusparseDestroyMatDescr(descr_);
      initialized_ = false;
    }
  }

  bool initialized_;
  cusparseMatDescr_t descr_;

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparseMatrixDescriptor);
};

// A wrapper class to ensure that an unsorted/sorted CSR conversion information
// struct (csru2csrInfo_t) is initialized only once. See:
// https://docs.nvidia.com/cuda/cusparse/index.html#csru2csr
class CudaSparseCsrSortingConversionInfo {
 public:
  explicit CudaSparseCsrSortingConversionInfo() : initialized_(false) {}

  CudaSparseCsrSortingConversionInfo(CudaSparseCsrSortingConversionInfo&& rhs)
      : initialized_(rhs.initialized_), info_(std::move(rhs.info_)) {
    rhs.initialized_ = false;
  }

  CudaSparseCsrSortingConversionInfo& operator=(
      CudaSparseCsrSortingConversionInfo&& rhs) {
    if (this == &rhs) return *this;
    Release();
    initialized_ = rhs.initialized_;
    info_ = std::move(rhs.info_);
    rhs.initialized_ = false;
    return *this;
  }

  ~CudaSparseCsrSortingConversionInfo() { Release(); }

  // Initializes the underlying info. Will fail on the second call if called
  // more than once.
  Status Initialize() {
    DCHECK(!initialized_);
    TF_RETURN_IF_CUSPARSE_ERROR(cusparseCreateCsru2csrInfo(&info_));
    initialized_ = true;
    return Status::OK();
  }

  csru2csrInfo_t& info() {
    DCHECK(initialized_);
    return info_;
  }

  const csru2csrInfo_t& info() const {
    DCHECK(initialized_);
    return info_;
  }

 private:
  void Release() {
    if (initialized_) {
      cusparseDestroyCsru2csrInfo(info_);
      initialized_ = false;
    }
  }

  bool initialized_;
  csru2csrInfo_t info_;

  TF_DISALLOW_COPY_AND_ASSIGN(CudaSparseCsrSortingConversionInfo);
};

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_KERNELS_CUDA_SPARSE_H_
