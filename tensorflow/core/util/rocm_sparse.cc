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

#if TENSORFLOW_USE_ROCM

#include <complex>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

namespace {

// A set of initialized handles to the underlying ROCm libraries used by
// GpuSparse. We maintain one such set of handles per unique stream.
class HipSparseHandles {
 public:
  explicit HipSparseHandles(hipStream_t stream)
      : initialized_(false), stream_(stream) {}

  HipSparseHandles(HipSparseHandles&& rhs)
      : initialized_(rhs.initialized_),
        stream_(std::move(rhs.stream_)),
        hipsparse_handle_(rhs.hipsparse_handle_) {
    rhs.initialized_ = false;
  }

  HipSparseHandles& operator=(HipSparseHandles&& rhs) {
    if (this == &rhs) return *this;
    Release();
    stream_ = std::move(rhs.stream_);
    hipsparse_handle_ = std::move(rhs.hipsparse_handle_);
    initialized_ = rhs.initialized_;
    rhs.initialized_ = false;
    return *this;
  }

  ~HipSparseHandles() { Release(); }

  Status Initialize() {
    if (initialized_) return Status::OK();
    TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseCreate(&hipsparse_handle_));
    TF_RETURN_IF_GPUSPARSE_ERROR(
        wrap::hipsparseSetStream(hipsparse_handle_, stream_));
    initialized_ = true;
    return Status::OK();
  }

  hipsparseHandle_t& handle() {
    DCHECK(initialized_);
    return hipsparse_handle_;
  }

  const hipsparseHandle_t& handle() const {
    DCHECK(initialized_);
    return hipsparse_handle_;
  }

 private:
  void Release() {
    if (initialized_) {
      // This should never return anything other than success
      auto err = wrap::hipsparseDestroy(hipsparse_handle_);
      DCHECK(err == HIPSPARSE_STATUS_SUCCESS)
          << "Failed to destroy hipSPARSE instance.";
      initialized_ = false;
    }
  }
  bool initialized_;
  hipStream_t stream_;
  hipsparseHandle_t hipsparse_handle_;

  TF_DISALLOW_COPY_AND_ASSIGN(HipSparseHandles);
};

// TODO(ebrevdo): Replace global mutex guarding CudaSparseHandles
// lookup with one of:
//    1. Adding the handle to the CudaStream structure; do the lookup there.
//    2. Add a thread-local cusparse, set it to the current stream
//       upon each call.
// #1 seems like the cleanest option but will need to wait until this
// is moved into TF core.
static mutex handle_map_mutex(LINKER_INITIALIZED);

using HandleMap = std::unordered_map<hipStream_t, HipSparseHandles>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

GpuSparse::GpuSparse(OpKernelContext* context)
    : initialized_(false), context_(context) {
  auto hip_stream_ptr =
      reinterpret_cast<const hipStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack());
  DCHECK(hip_stream_ptr);
  gpu_stream_ = *hip_stream_ptr;
}

Status GpuSparse::Initialize() {
  HandleMap* handle_map = GetHandleMapSingleton();
  DCHECK(handle_map);
  mutex_lock lock(handle_map_mutex);
  auto it = handle_map->find(gpu_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating GpuSparse handles for stream " << gpu_stream_;
    // Previously unseen ROCm stream. Initialize a set of ROCm sparse library
    // handles for it.
    HipSparseHandles new_handles(gpu_stream_);
    TF_RETURN_IF_ERROR(new_handles.Initialize());
    it = handle_map->insert(std::make_pair(gpu_stream_, std::move(new_handles)))
             .first;
  }
  gpusparse_handle_ = &it->second.handle();
  initialized_ = true;
  return Status::OK();
}

#define TF_CALL_HIPSPARSE_DTYPES(m)          \
  m(float, ROCM_R_32F) m(double, ROCM_R_64F) \
      m(std::complex<float>, ROCM_C_32F) m(std::complex<double>, ROCM_C_64F)

// Macro that specializes a sparse method for all 4 standard
// numeric types.
#define TF_CALL_HIP_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct hipsparse method names.
#define SPARSE_FN(method, sparse_prefix) wrap::hipsparse##sparse_prefix##method
#define BUFSIZE_FN(method, sparse_prefix) \
  wrap::hipsparse##sparse_prefix##method##_bufferSizeExt

Status GpuSparse::Coo2csr(const int* cooRowInd, int nnz, int m,
                          int* csrRowPtr) const {
  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(
      wrap::hipsparseXcoo2csr(*gpusparse_handle_, cooRowInd, nnz, m, csrRowPtr,
                              HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

Status GpuSparse::Csr2coo(const int* csrRowPtr, int nnz, int m,
                          int* cooRowInd) const {
  DCHECK(initialized_);
  TF_RETURN_IF_GPUSPARSE_ERROR(
      wrap::hipsparseXcsr2coo(*gpusparse_handle_, csrRowPtr, nnz, m, cooRowInd,
                              HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

#if TF_ROCM_VERSION < 40200
template <typename Scalar, typename SparseFnT>
static inline Status CsrmmImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t hipsparse_handle,
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, int nnz, const Scalar* alpha_host, const hipsparseMatDescr_t descrA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* B, int ldb,
    const Scalar* beta_host, Scalar* C, int ldc) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      hipsparse_handle, transA, transB, m, n, k, nnz, AsHipComplex(alpha_host),
      descrA, AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(B), ldb, AsHipComplex(beta_host), AsHipComplex(C), ldc));
  return Status::OK();
}

#define CSRMM_INSTANCE(Scalar, sparse_prefix)                                 \
  template <>                                                                 \
  Status GpuSparse::Csrmm<Scalar>(                                            \
      hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n, \
      int k, int nnz, const Scalar* alpha_host,                               \
      const hipsparseMatDescr_t descrA, const Scalar* csrSortedValA,          \
      const int* csrSortedRowPtrA, const int* csrSortedColIndA,               \
      const Scalar* B, int ldb, const Scalar* beta_host, Scalar* C, int ldc)  \
      const {                                                                 \
    DCHECK(initialized_);                                                     \
    return CsrmmImpl(SPARSE_FN(csrmm2, sparse_prefix), context_,              \
                     *gpusparse_handle_, transA, transB, m, n, k, nnz,        \
                     alpha_host, descrA, csrSortedValA, csrSortedRowPtrA,     \
                     csrSortedColIndA, B, ldb, beta_host, C, ldc);            \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRMM_INSTANCE);

#else

#define SPMM_BUFFERSIZE_INSTANCE(Scalar, dtype)                               \
  template <>                                                                 \
  Status GpuSparse::SpMMBufferSize<Scalar>(                                   \
      hipsparseOperation_t transA, hipsparseOperation_t transB,               \
      const Scalar* alpha, const hipsparseSpMatDescr_t matA,                  \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                   \
      gpusparseDnMatDescr_t matC, hipsparseSpMMAlg_t alg, size_t* bufferSize) \
      const {                                                                 \
    DCHECK(initialized_);                                                     \
    TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseSpMM_bufferSize(              \
        *gpusparse_handle_, transA, transB, alpha, matA, matB, beta, matC,    \
        dtype, alg, bufferSize));                                             \
    return Status::OK();                                                      \
  }

TF_CALL_HIPSPARSE_DTYPES(SPMM_BUFFERSIZE_INSTANCE);

#define SPMM_INSTANCE(Scalar, dtype)                                         \
  template <>                                                                \
  Status GpuSparse::SpMM<Scalar>(                                            \
      hipsparseOperation_t transA, hipsparseOperation_t transB,              \
      const Scalar* alpha, const hipsparseSpMatDescr_t matA,                 \
      const gpusparseDnMatDescr_t matB, const Scalar* beta,                  \
      gpusparseDnMatDescr_t matC, hipsparseSpMMAlg_t alg, int8* buffer)      \
      const {                                                                \
    DCHECK(initialized_);                                                    \
    TF_RETURN_IF_GPUSPARSE_ERROR(                                            \
        wrap::hipsparseSpMM(*gpusparse_handle_, transA, transB, alpha, matA, \
                            matB, beta, matC, dtype, alg, buffer));          \
    return Status::OK();                                                     \
  }

TF_CALL_HIPSPARSE_DTYPES(SPMM_INSTANCE);

#endif

template <typename Scalar, typename SparseFnT>
static inline Status CsrmvImpl(SparseFnT op, OpKernelContext* context,
                               hipsparseHandle_t hipsparse_handle,
                               hipsparseOperation_t transA, int m, int n,
                               int nnz, const Scalar* alpha_host,
                               const hipsparseMatDescr_t descrA,
                               const Scalar* csrSortedValA,
                               const int* csrSortedRowPtrA,
                               const int* csrSortedColIndA, const Scalar* x,
                               const Scalar* beta_host, Scalar* y) {
  TF_RETURN_IF_GPUSPARSE_ERROR(
      op(hipsparse_handle, transA, m, n, nnz, AsHipComplex(alpha_host), descrA,
         AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
         AsHipComplex(x), AsHipComplex(beta_host), AsHipComplex(y)));
  return Status::OK();
}

// TODO(ebrevdo,rmlarsen): Use csrmv_mp for all cases when available in CUDA 9.
#define CSRMV_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                \
  Status GpuSparse::Csrmv<Scalar>(                                           \
      hipsparseOperation_t transA, int m, int n, int nnz,                    \
      const Scalar* alpha_host, const hipsparseMatDescr_t descrA,            \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,              \
      const int* csrSortedColIndA, const Scalar* x, const Scalar* beta_host, \
      Scalar* y) const {                                                     \
    DCHECK(initialized_);                                                    \
    return CsrmvImpl(SPARSE_FN(csrmv, sparse_prefix), context_,              \
                     *gpusparse_handle_, transA, m, n, nnz, alpha_host,      \
                     descrA, csrSortedValA, csrSortedRowPtrA,                \
                     csrSortedColIndA, x, beta_host, y);                     \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRMV_INSTANCE);

Status GpuSparse::CsrgemmNnz(
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, const hipsparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const hipsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr) {
  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
  TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseXcsrgemmNnz(
      *gpusparse_handle_, transA, transB, m, n, k, descrA, nnzA,
      csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
      csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status CsrgemmImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t hipsparse_handle,
    hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,
    int k, const hipsparseMatDescr_t descrA, int nnzA,
    const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const hipsparseMatDescr_t descrB, int nnzB,
    const Scalar* csrSortedValB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      hipsparse_handle, transA, transB, m, n, k, descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA, descrB,
      nnzB, AsHipComplex(csrSortedValB), csrSortedRowPtrB, csrSortedColIndB,
      descrC, AsHipComplex(csrSortedValC), csrSortedRowPtrC, csrSortedColIndC));
  return Status::OK();
}

#define CSRGEMM_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                  \
  Status GpuSparse::Csrgemm<Scalar>(                                           \
      hipsparseOperation_t transA, hipsparseOperation_t transB, int m, int n,  \
      int k, const hipsparseMatDescr_t descrA, int nnzA,                       \
      const Scalar* csrSortedValA, const int* csrSortedRowPtrA,                \
      const int* csrSortedColIndA, const hipsparseMatDescr_t descrB, int nnzB, \
      const Scalar* csrSortedValB, const int* csrSortedRowPtrB,                \
      const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,           \
      Scalar* csrSortedValC, int* csrSortedRowPtrC, int* csrSortedColIndC) {   \
    DCHECK(initialized_);                                                      \
    return CsrgemmImpl(SPARSE_FN(csrgemm, sparse_prefix), context_,            \
                       *gpusparse_handle_, transA, transB, m, n, k, descrA,    \
                       nnzA, csrSortedValA, csrSortedRowPtrA,                  \
                       csrSortedColIndA, descrB, nnzB, csrSortedValB,          \
                       csrSortedRowPtrB, csrSortedColIndB, descrC,             \
                       csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);     \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEMM_INSTANCE);

#if TF_ROCM_VERSION >= 40200

template <typename Scalar, typename BufferSizeFnT, typename SparseFnT>
static inline Status Csru2csrImpl(SparseFnT op, BufferSizeFnT buffer_size_op,
                                  OpKernelContext* context,
                                  hipsparseHandle_t hipsparse_handle, int m,
                                  int n, int nnz,
                                  const hipsparseMatDescr_t descrA,
                                  Scalar* csrVal, const int* csrRowPtr,
                                  int* csrColInd) {
  GpuSparseCsrSortingConversionInfo info;
  TF_RETURN_IF_ERROR(info.Initialize());

  size_t pBufferSizeInBytes = 0;

  TF_RETURN_IF_GPUSPARSE_ERROR(
      buffer_size_op(hipsparse_handle, m, n, nnz, AsHipComplex(csrVal),
                     csrRowPtr, csrColInd, info.info(), &pBufferSizeInBytes));

  Tensor pBuffer_t;
  TF_RETURN_IF_ERROR(context->allocate_temp(
      DT_INT8, TensorShape({static_cast<int64_t>(pBufferSizeInBytes)}),
      &pBuffer_t));
  auto pBuffer = pBuffer_t.flat<int8>();
  DCHECK(pBuffer.data() != nullptr);

  TF_RETURN_IF_GPUSPARSE_ERROR(op(hipsparse_handle, m, n, nnz, descrA,
                                  AsHipComplex(csrVal), csrRowPtr, csrColInd,
                                  info.info(), pBuffer.data()));

  return Status::OK();
}

#define CSRU2CSR_INSTANCE(Scalar, sparse_prefix)                               \
  template <>                                                                  \
  Status GpuSparse::Csru2csr<Scalar>(                                          \
      int m, int n, int nnz, const hipsparseMatDescr_t descrA, Scalar* csrVal, \
      const int* csrRowPtr, int* csrColInd) {                                  \
    DCHECK(initialized_);                                                      \
    return Csru2csrImpl(SPARSE_FN(csru2csr, sparse_prefix),                    \
                        BUFSIZE_FN(csru2csr, sparse_prefix), context_,         \
                        *gpusparse_handle_, m, n, nnz, descrA, csrVal,         \
                        csrRowPtr, csrColInd);                                 \
  }

TF_CALL_LAPACK_TYPES(CSRU2CSR_INSTANCE);

#endif

template <typename Scalar, typename SparseFnT>
static inline Status Csr2cscImpl(SparseFnT op, OpKernelContext* context,
                                 hipsparseHandle_t hipsparse_handle, int m,
                                 int n, int nnz, const Scalar* csrVal,
                                 const int* csrRowPtr, const int* csrColInd,
                                 Scalar* cscVal, int* cscRowInd, int* cscColPtr,
                                 const hipsparseAction_t copyValues) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(hipsparse_handle, m, n, nnz,
                                  AsHipComplex(csrVal), csrRowPtr, csrColInd,
                                  AsHipComplex(cscVal), cscRowInd, cscColPtr,
                                  copyValues, HIPSPARSE_INDEX_BASE_ZERO));
  return Status::OK();
}

#define CSR2CSC_INSTANCE(Scalar, sparse_prefix)                              \
  template <>                                                                \
  Status GpuSparse::Csr2csc<Scalar>(                                         \
      int m, int n, int nnz, const Scalar* csrVal, const int* csrRowPtr,     \
      const int* csrColInd, Scalar* cscVal, int* cscRowInd, int* cscColPtr,  \
      const hipsparseAction_t copyValues) {                                  \
    DCHECK(initialized_);                                                    \
    return Csr2cscImpl(SPARSE_FN(csr2csc, sparse_prefix), context_,          \
                       *gpusparse_handle_, m, n, nnz, csrVal, csrRowPtr,     \
                       csrColInd, cscVal, cscRowInd, cscColPtr, copyValues); \
  }

TF_CALL_HIP_LAPACK_TYPES(CSR2CSC_INSTANCE);

template <typename Scalar, typename SparseFnT>
static inline Status CsrgeamBufferSizeExtImpl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t sparse_handle,
    int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      sparse_handle, m, n, AsHipComplex(alpha), descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(beta), descrB, nnzB, AsHipComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsHipComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, bufferSize));
  return Status::OK();
}

#define CSRGEAM_BUFFERSIZE_INSTANCE(Scalar, sparse_prefix)                     \
  template <>                                                                  \
  Status GpuSparse::CsrgeamBufferSizeExt<Scalar>(                              \
      int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,      \
      const int* csrSortedColIndA, const Scalar* beta,                         \
      const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,                \
      const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, size_t* bufferSize) {      \
    DCHECK(initialized_);                                                      \
    return CsrgeamBufferSizeExtImpl(                                           \
        SPARSE_FN(csrgeam2_bufferSizeExt, sparse_prefix), context_,            \
        *gpusparse_handle_, m, n, alpha, descrA, nnzA, csrSortedValA,          \
        csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, \
        csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,             \
        csrSortedRowPtrC, csrSortedColIndC, bufferSize);                       \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEAM_BUFFERSIZE_INSTANCE);

Status GpuSparse::CsrgeamNnz(
    int m, int n, const hipsparseMatDescr_t descrA, int nnzA,
    const int* csrSortedRowPtrA, const int* csrSortedColIndA,
    const hipsparseMatDescr_t descrB, int nnzB, const int* csrSortedRowPtrB,
    const int* csrSortedColIndB, const hipsparseMatDescr_t descrC,
    int* csrSortedRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
  DCHECK(initialized_);
  DCHECK(nnzTotalDevHostPtr != nullptr);
  TF_RETURN_IF_GPUSPARSE_ERROR(wrap::hipsparseXcsrgeam2Nnz(
      *gpusparse_handle_, m, n, descrA, nnzA, csrSortedRowPtrA,
      csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
      descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace));
  return Status::OK();
}

template <typename Scalar, typename SparseFnT>
static inline Status Csrgeam2Impl(
    SparseFnT op, OpKernelContext* context, hipsparseHandle_t cusparse_handle,
    int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,
    int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,
    const int* csrSortedColIndA, const Scalar* beta,
    const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB,
    const int* csrSortedRowPtrB, const int* csrSortedColIndB,
    const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,
    int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {
  TF_RETURN_IF_GPUSPARSE_ERROR(op(
      cusparse_handle, m, n, AsHipComplex(alpha), descrA, nnzA,
      AsHipComplex(csrSortedValA), csrSortedRowPtrA, csrSortedColIndA,
      AsHipComplex(beta), descrB, nnzB, AsHipComplex(csrSortedValB),
      csrSortedRowPtrB, csrSortedColIndB, descrC, AsHipComplex(csrSortedValC),
      csrSortedRowPtrC, csrSortedColIndC, workspace));
  return Status::OK();
}

#define CSRGEAM_INSTANCE(Scalar, sparse_prefix)                                \
  template <>                                                                  \
  Status GpuSparse::Csrgeam<Scalar>(                                           \
      int m, int n, const Scalar* alpha, const hipsparseMatDescr_t descrA,     \
      int nnzA, const Scalar* csrSortedValA, const int* csrSortedRowPtrA,      \
      const int* csrSortedColIndA, const Scalar* beta,                         \
      const hipsparseMatDescr_t descrB, int nnzB, const Scalar* csrSortedValB, \
      const int* csrSortedRowPtrB, const int* csrSortedColIndB,                \
      const hipsparseMatDescr_t descrC, Scalar* csrSortedValC,                 \
      int* csrSortedRowPtrC, int* csrSortedColIndC, void* workspace) {         \
    DCHECK(initialized_);                                                      \
    return Csrgeam2Impl(SPARSE_FN(csrgeam2, sparse_prefix), context_,          \
                        *gpusparse_handle_, m, n, alpha, descrA, nnzA,         \
                        csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,     \
                        beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,   \
                        csrSortedColIndB, descrC, csrSortedValC,               \
                        csrSortedRowPtrC, csrSortedColIndC, workspace);        \
  }

TF_CALL_HIP_LAPACK_TYPES(CSRGEAM_INSTANCE);

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
