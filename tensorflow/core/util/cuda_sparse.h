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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_CUDA_SPARSE_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_CUDA_SPARSE_H_

// This header declares the class GpuSparse, which contains wrappers of
// cuSparse libraries for use in TensorFlow kernels.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <functional>
#include <vector>

#if GOOGLE_CUDA

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cusparse.h"

using gpusparseStatus_t = cusparseStatus_t;
using gpusparseOperation_t = cusparseOperation_t;
using gpusparseMatDescr_t = cusparseMatDescr_t;
using gpusparseAction_t = cusparseAction_t;
using gpusparseHandle_t = cusparseHandle_t;
using gpuStream_t = cudaStream_t;
#if CUDA_VERSION >= 10020
using gpusparseDnMatDescr_t = cusparseDnMatDescr_t;
using gpusparseSpMatDescr_t = cusparseSpMatDescr_t;
using gpusparseSpMMAlg_t = cusparseSpMMAlg_t;
#endif

#define GPUSPARSE(postfix) CUSPARSE_##postfix
#define gpusparse(postfix) cusparse##postfix

#elif TENSORFLOW_USE_ROCM

#include "tensorflow/stream_executor/rocm/hipsparse_wrapper.h"

using gpusparseStatus_t = hipsparseStatus_t;
using gpusparseOperation_t = hipsparseOperation_t;
using gpusparseMatDescr_t = hipsparseMatDescr_t;
using gpusparseAction_t = hipsparseAction_t;
using gpusparseHandle_t = hipsparseHandle_t;
using gpuStream_t = hipStream_t;
#if TF_ROCM_VERSION >= 40200
using gpusparseDnMatDescr_t = hipsparseDnMatDescr_t;
using gpusparseSpMatDescr_t = hipsparseSpMatDescr_t;
using gpusparseSpMMAlg_t = hipsparseSpMMAlg_t;
#endif
#define GPUSPARSE(postfix) HIPSPARSE_##postfix
#define gpusparse(postfix) hipsparse##postfix

#endif

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

inline std::string ConvertGPUSparseErrorToString(
    const gpusparseStatus_t status) {
  switch (status) {
#define STRINGIZE(q) #q
#define RETURN_IF_STATUS(err) \
  case err:                   \
    return STRINGIZE(err);

#if GOOGLE_CUDA

    RETURN_IF_STATUS(CUSPARSE_STATUS_SUCCESS)
    RETURN_IF_STATUS(CUSPARSE_STATUS_NOT_INITIALIZED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ALLOC_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INVALID_VALUE)
    RETURN_IF_STATUS(CUSPARSE_STATUS_ARCH_MISMATCH)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MAPPING_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_EXECUTION_FAILED)
    RETURN_IF_STATUS(CUSPARSE_STATUS_INTERNAL_ERROR)
    RETURN_IF_STATUS(CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)

    default:
      return strings::StrCat("Unknown CUSPARSE error: ",
                             static_cast<int>(status));
#elif TENSORFLOW_USE_ROCM

    RETURN_IF_STATUS(HIPSPARSE_STATUS_SUCCESS)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_NOT_INITIALIZED)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_ALLOC_FAILED)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_INVALID_VALUE)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_ARCH_MISMATCH)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_MAPPING_ERROR)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_EXECUTION_FAILED)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_INTERNAL_ERROR)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
    RETURN_IF_STATUS(HIPSPARSE_STATUS_ZERO_PIVOT)

    default:
      return strings::StrCat("Unknown hipSPARSE error: ",
                             static_cast<int>(status));
#endif

#undef RETURN_IF_STATUS
#undef STRINGIZE
  }
}

#if GOOGLE_CUDA

#define TF_RETURN_IF_GPUSPARSE_ERROR(expr)                                 \
  do {                                                                     \
    auto status = (expr);                                                  \
    if (TF_PREDICT_FALSE(status != CUSPARSE_STATUS_SUCCESS)) {             \
      return errors::Internal(__FILE__, ":", __LINE__, " (", TF_STR(expr), \
                              "): cuSparse call failed with status ",      \
                              ConvertGPUSparseErrorToString(status));      \
    }                                                                      \
  } while (0)

#elif TENSORFLOW_USE_ROCM

#define TF_RETURN_IF_GPUSPARSE_ERROR(expr)                                 \
  do {                                                                     \
    auto status = (expr);                                                  \
    if (TF_PREDICT_FALSE(status != HIPSPARSE_STATUS_SUCCESS)) {            \
      return errors::Internal(__FILE__, ":", __LINE__, " (", TF_STR(expr), \
                              "): hipSPARSE call failed with status ",     \
                              ConvertGPUSparseErrorToString(status));      \
    }                                                                      \
  } while (0)

#endif

inline gpusparseOperation_t TransposeAndConjugateToGpuSparseOp(bool transpose,
                                                               bool conjugate,
                                                               Status* status) {
#if GOOGLE_CUDA
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
#elif TENSORFLOW_USE_ROCM
  if (transpose) {
    return conjugate ? HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                     : HIPSPARSE_OPERATION_TRANSPOSE;
  } else {
    if (conjugate) {
      DCHECK(status != nullptr);
      *status = errors::InvalidArgument(
          "Conjugate == True and transpose == False is not supported.");
    }
    return HIPSPARSE_OPERATION_NON_TRANSPOSE;
  }
#endif
}

// The GpuSparse class provides a simplified templated API for cuSparse
// (http://docs.nvidia.com/cuda/cusparse/index.html).
// An object of this class wraps static cuSparse instances,
// and will launch Cuda kernels on the stream wrapped by the GPU device
// in the OpKernelContext provided to the constructor.
//
// Notice: All the computational member functions are asynchronous and simply
// launch one or more Cuda kernels on the Cuda stream wrapped by the GpuSparse
// object.

class GpuSparse {
 public:
  // This object stores a pointer to context, which must outlive it.
  explicit GpuSparse(OpKernelContext* context);
  virtual ~GpuSparse() {}

  // This initializes the GpuSparse class if it hasn't
  // been initialized yet.  All following public methods require the
  // class has been initialized.  Can be run multiple times; all
  // subsequent calls after the first have no effect.
  Status Initialize();  // Move to constructor?

  // ====================================================================
  // Wrappers for cuSparse start here.
  //

  // Solves tridiagonal system of equations.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2
  template <typename Scalar>
  Status Gtsv2(int m, int n, const Scalar* dl, const Scalar* d,
               const Scalar* du, Scalar* B, int ldb, void* pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_bufferSize
  template <typename Scalar>
  Status Gtsv2BufferSizeExt(int m, int n, const Scalar* dl, const Scalar* d,
                            const Scalar* du, const Scalar* B, int ldb,
                            size_t* bufferSizeInBytes) const;

  // Solves tridiagonal system of equations without partial pivoting.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot
  template <typename Scalar>
  Status Gtsv2NoPivot(int m, int n, const Scalar* dl, const Scalar* d,
                      const Scalar* du, Scalar* B, int ldb,
                      void* pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2NoPivot.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot_bufferSize
  template <typename Scalar>
  Status Gtsv2NoPivotBufferSizeExt(int m, int n, const Scalar* dl,
                                   const Scalar* d, const Scalar* du,
                                   const Scalar* B, int ldb,
                                   size_t* bufferSizeInBytes) const;

  // Solves a batch of tridiagonal systems of equations. Doesn't support
  // multiple right-hand sides per each system. Doesn't do pivoting.
  // See: https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch
  template <typename Scalar>
  Status Gtsv2StridedBatch(int m, const Scalar* dl, const Scalar* d,
                           const Scalar* du, Scalar* x, int batchCount,
                           int batchStride, void* pBuffer) const;

  // Computes the size of a temporary buffer used by Gtsv2StridedBatch.
  // See:
  // https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2stridedbatch_bufferSize
  template <typename Scalar>
  Status Gtsv2StridedBatchBufferSizeExt(int m, const Scalar* dl,
                                        const Scalar* d, const Scalar* du,
                                        const Scalar* x, int batchCount,
                                        int batchStride,
                                        size_t* bufferSizeInBytes) const;

  // Compresses the indices of rows or columns. It can be interpreted as a
  // conversion from COO to CSR sparse storage format. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csr2coo.
  Status Csr2coo(const int* CsrRowPtr, int nnz, int m, int* cooRowInd) const;

  // Uncompresses the indices of rows or columns. It can be interpreted as a
  // conversion from CSR to COO sparse storage format. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-coo2csr.
  Status Coo2csr(const int* cooRowInd, int nnz, int m, int* csrRowPtr) const;

#if (GOOGLE_CUDA && (CUDA_VERSION < 10020)) || \
    (TENSORFLOW_USE_ROCM && TF_ROCM_VERSION < 40200)
  // Sparse-dense matrix multiplication C = alpha * op(A) * op(B)  + beta * C,
  // where A is a sparse matrix in CSR format, B and C are dense tall
  // matrices.  This routine allows transposition of matrix B, which
  // may improve performance.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmm2
  //
  // **NOTE** Matrices B and C are expected to be in column-major
  // order; to make them consistent with TensorFlow they
  // must be transposed (or the matmul op's pre/post-processing must take this
  // into account).
  //
  // **NOTE** This is an in-place operation for data in C.
  template <typename Scalar>
  Status Csrmm(gpusparseOperation_t transA, gpusparseOperation_t transB, int m,
               int n, int k, int nnz, const Scalar* alpha_host,
               const gpusparseMatDescr_t descrA, const Scalar* csrSortedValA,
               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
               const Scalar* B, int ldb, const Scalar* beta_host, Scalar* C,
               int ldc) const;
#else  // CUDA_VERSION >=10200 || TF_ROCM_VERSION >= 40200
  // Workspace size query for sparse-dense matrix multiplication. Helper
  // function for SpMM which computes y = alpha * op(A) * op(B) + beta * C,
  // where A is a sparse matrix in CSR format, B and C are dense matricies in
  // column-major format. Returns needed workspace size in bytes.
  template <typename Scalar>
  Status SpMMBufferSize(gpusparseOperation_t transA,
                        gpusparseOperation_t transB, const Scalar* alpha,
                        const gpusparseSpMatDescr_t matA,
                        const gpusparseDnMatDescr_t matB, const Scalar* beta,
                        gpusparseDnMatDescr_t matC, gpusparseSpMMAlg_t alg,
                        size_t* bufferSize) const;

  // Sparse-dense matrix multiplication y = alpha * op(A) * op(B) + beta * C,
  // where A is a sparse matrix in CSR format, B and C are dense matricies in
  // column-major format. Buffer is assumed to be at least as large as the
  // workspace size returned by SpMMBufferSize().
  //
  // **NOTE** This is an in-place operation for data in C.
  template <typename Scalar>
  Status SpMM(gpusparseOperation_t transA, gpusparseOperation_t transB,
              const Scalar* alpha, const gpusparseSpMatDescr_t matA,
              const gpusparseDnMatDescr_t matB, const Scalar* beta,
              gpusparseDnMatDescr_t matC, gpusparseSpMMAlg_t alg,
              int8* buffer) const;
#endif

  // Sparse-dense vector multiplication y = alpha * op(A) * x  + beta * y,
  // where A is a sparse matrix in CSR format, x and y are dense vectors. See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrmv_mergepath
  //
  // **NOTE** This is an in-place operation for data in y.
#if (GOOGLE_CUDA && (CUDA_VERSION < 10020)) || TENSORFLOW_USE_ROCM
  template <typename Scalar>
  Status Csrmv(gpusparseOperation_t transA, int m, int n, int nnz,
               const Scalar* alpha_host, const gpusparseMatDescr_t descrA,
               const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
               const int* csrSortedColIndA, const Scalar* x,
               const Scalar* beta_host, Scalar* y) const;
#else
  template <typename Scalar>
  Status Csrmv(gpusparseOperation_t transA, int m, int n, int nnz,
               const Scalar* alpha_host, const Scalar* csrSortedValA,
               const int* csrSortedRowPtrA, const int* csrSortedColIndA,
               const Scalar* x, const Scalar* beta_host, Scalar* y) const;
#endif  // CUDA_VERSION < 10020

  // Computes workspace size for sparse - sparse matrix addition of matrices
  // stored in CSR format.
  template <typename Scalar>
  Status CsrgeamBufferSizeExt(
      int m, int n, const Scalar* alpha, const gpusparseMatDescr_t descrA,
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
      const int* csrSortedColIndA, const Scalar* beta,
      const gpusparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,
      const gpusparseMatDescr_t descrC, Scalar* csrSortedValC,
      int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize);

  // Computes sparse-sparse matrix addition of matrices
  // stored in CSR format.  This is part one: calculate nnz of the
  // output.  csrSortedRowPtrC must be preallocated on device with
  // m + 1 entries.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgeam.
  Status CsrgeamNnz(int m, int n, const gpusparseMatDescr_t descrA, int nnzA,
                    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                    const gpusparseMatDescr_t descrB, int nnzB,
                    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                    const gpusparseMatDescr_t descrC, int* csrSortedRowPtrC,
                    int* nnzTotalDevHostPtr, void* workspace);

  // Computes sparse - sparse matrix addition of matrices
  // stored in CSR format.  This is part two: perform sparse-sparse
  // addition.  csrValC and csrColIndC must be allocated on the device
  // with nnzTotalDevHostPtr entries (as calculated by CsrgeamNnz).  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgeam.
  template <typename Scalar>
  Status Csrgeam(int m, int n, const Scalar* alpha,
                 const gpusparseMatDescr_t descrA, int nnzA,
                 const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
                 const int* csrSortedColIndA, const Scalar* beta,
                 const gpusparseMatDescr_t descrB, int nnzB,
                 const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
                 const int* csrSortedColIndB, const gpusparseMatDescr_t descrC,
                 Scalar* csrSortedValC, int* csrSortedRowPtrC,
                 int* csrSortedColIndC, void* workspace);

#if GOOGLE_CUDA && (CUDA_VERSION >= 10000)
  // Computes sparse-sparse matrix multiplication of matrices
  // stored in CSR format.  This is part zero: calculate required workspace
  // size.
  template <typename Scalar>
  Status CsrgemmBufferSize(
      int m, int n, int k, const gpusparseMatDescr_t descrA, int nnzA,
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,
      const gpusparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
      const int* csrSortedColIndB, csrgemm2Info_t info, size_t* workspaceBytes);
#endif

  // Computes sparse-sparse matrix multiplication of matrices
  // stored in CSR format.  This is part one: calculate nnz of the
  // output.  csrSortedRowPtrC must be preallocated on device with
  // m + 1 entries.  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm.
#if (GOOGLE_CUDA && (CUDA_VERSION < 10000)) || TENSORFLOW_USE_ROCM
  Status CsrgemmNnz(gpusparseOperation_t transA, gpusparseOperation_t transB,
                    int m, int k, int n, const gpusparseMatDescr_t descrA,
                    int nnzA, const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const gpusparseMatDescr_t descrB, int nnzB,
                    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                    const gpusparseMatDescr_t descrC, int* csrSortedRowPtrC,
                    int* nnzTotalDevHostPtr);
#else
  Status CsrgemmNnz(int m, int n, int k, const gpusparseMatDescr_t descrA,
                    int nnzA, const int* csrSortedRowPtrA,
                    const int* csrSortedColIndA,
                    const gpusparseMatDescr_t descrB, int nnzB,
                    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
                    const gpusparseMatDescr_t descrC, int* csrSortedRowPtrC,
                    int* nnzTotalDevHostPtr, csrgemm2Info_t info,
                    void* workspace);
#endif

  // Computes sparse - sparse matrix matmul of matrices
  // stored in CSR format.  This is part two: perform sparse-sparse
  // addition.  csrValC and csrColIndC must be allocated on the device
  // with nnzTotalDevHostPtr entries (as calculated by CsrgemmNnz).  See:
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm.
#if (GOOGLE_CUDA && (CUDA_VERSION < 10000)) || TENSORFLOW_USE_ROCM
  template <typename Scalar>
  Status Csrgemm(gpusparseOperation_t transA, gpusparseOperation_t transB,
                 int m, int k, int n, const gpusparseMatDescr_t descrA,
                 int nnzA, const Scalar* csrSortedValA,
                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                 const gpusparseMatDescr_t descrB, int nnzB,
                 const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
                 const int* csrSortedColIndB, const gpusparseMatDescr_t descrC,
                 Scalar* csrSortedValC, int* csrSortedRowPtrC,
                 int* csrSortedColIndC);
#else
  template <typename Scalar>
  Status Csrgemm(int m, int n, int k, const gpusparseMatDescr_t descrA,
                 int nnzA, const Scalar* csrSortedValA,
                 const int* csrSortedRowPtrA, const int* csrSortedColIndA,
                 const gpusparseMatDescr_t descrB, int nnzB,
                 const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
                 const int* csrSortedColIndB, const gpusparseMatDescr_t descrC,
                 Scalar* csrSortedValC, int* csrSortedRowPtrC,
                 int* csrSortedColIndC, const csrgemm2Info_t info,
                 void* workspace);
#endif

  // In-place reordering of unsorted CSR to sorted CSR.
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csru2csr
  template <typename Scalar>
  Status Csru2csr(int m, int n, int nnz, const gpusparseMatDescr_t descrA,
                  Scalar* csrVal, const int* csrRowPtr, int* csrColInd);

  // Converts from CSR to CSC format (equivalently, transpose).
  // http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-csr2cscEx
  template <typename Scalar>
  Status Csr2csc(int m, int n, int nnz, const Scalar* csrVal,
                 const int* csrRowPtr, const int* csrColInd, Scalar* cscVal,
                 int* cscRowInd, int* cscColPtr,
                 const gpusparseAction_t copyValues);

 private:
  bool initialized_;
  OpKernelContext* context_;  // not owned.
  gpuStream_t gpu_stream_;
  gpusparseHandle_t* gpusparse_handle_;  // not owned.

  TF_DISALLOW_COPY_AND_ASSIGN(GpuSparse);
};

// A wrapper class to ensure that a CUDA sparse matrix descriptor is initialized
// only once. For more details on the descriptor (gpusparseMatDescr_t), see:
// https://docs.nvidia.com/cuda/cusparse/index.html#cusparsematdescrt
class GpuSparseMatrixDescriptor {
 public:
  explicit GpuSparseMatrixDescriptor() : initialized_(false) {}

  GpuSparseMatrixDescriptor(GpuSparseMatrixDescriptor&& rhs)
      : initialized_(rhs.initialized_), descr_(std::move(rhs.descr_)) {
    rhs.initialized_ = false;
  }

  GpuSparseMatrixDescriptor& operator=(GpuSparseMatrixDescriptor&& rhs) {
    if (this == &rhs) return *this;
    Release();
    initialized_ = rhs.initialized_;
    descr_ = std::move(rhs.descr_);
    rhs.initialized_ = false;
    return *this;
  }

  ~GpuSparseMatrixDescriptor() { Release(); }

  // Initializes the underlying descriptor.  Will fail on the second call if
  // called more than once.
  Status Initialize() {
    DCHECK(!initialized_);
#if GOOGLE_CUDA
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateMatDescr(&descr_));
#elif TENSORFLOW_USE_ROCM
    TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseCreateMatDescr(&descr_));
#endif
    initialized_ = true;
    return Status::OK();
  }

  gpusparseMatDescr_t& descr() {
    DCHECK(initialized_);
    return descr_;
  }

  const gpusparseMatDescr_t& descr() const {
    DCHECK(initialized_);
    return descr_;
  }

 private:
  void Release() {
    if (initialized_) {
#if GOOGLE_CUDA
      cusparseDestroyMatDescr(descr_);
#elif TENSORFLOW_USE_ROCM
      wrap::hipsparseDestroyMatDescr(descr_);
#endif
      initialized_ = false;
    }
  }

  bool initialized_;
  gpusparseMatDescr_t descr_;

  TF_DISALLOW_COPY_AND_ASSIGN(GpuSparseMatrixDescriptor);
};

#if GOOGLE_CUDA

// A wrapper class to ensure that an unsorted/sorted CSR conversion information
// struct (csru2csrInfo_t) is initialized only once. See:
// https://docs.nvidia.com/cuda/cusparse/index.html#csru2csr
class GpuSparseCsrSortingConversionInfo {
 public:
  explicit GpuSparseCsrSortingConversionInfo() : initialized_(false) {}

  GpuSparseCsrSortingConversionInfo(GpuSparseCsrSortingConversionInfo&& rhs)
      : initialized_(rhs.initialized_), info_(std::move(rhs.info_)) {
    rhs.initialized_ = false;
  }

  GpuSparseCsrSortingConversionInfo& operator=(
      GpuSparseCsrSortingConversionInfo&& rhs) {
    if (this == &rhs) return *this;
    Release();
    initialized_ = rhs.initialized_;
    info_ = std::move(rhs.info_);
    rhs.initialized_ = false;
    return *this;
  }

  ~GpuSparseCsrSortingConversionInfo() { Release(); }

  // Initializes the underlying info. Will fail on the second call if called
  // more than once.
  Status Initialize() {
    DCHECK(!initialized_);
    TF_RETURN_IF_GPUSPARSE_ERROR(cusparseCreateCsru2csrInfo(&info_));
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

  TF_DISALLOW_COPY_AND_ASSIGN(GpuSparseCsrSortingConversionInfo);
};

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_CUDA_SPARSE_H_
