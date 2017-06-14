/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/
#ifdef GOOGLE_CUDA
#include "tensorflow/core/kernels/cuda_solvers.h"

#include <chrono>
#include <complex>
#include <unordered_map>
#include <vector>

#include "cuda/include/cublas_v2.h"
#include "cuda/include/cusolverDn.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

inline bool CopyHostToDevice(OpKernelContext* context, void* dst,
                             const void* src, uint64 bytes) {
  auto stream = context->op_device_context()->stream();
  perftools::gputools::DeviceMemoryBase wrapped_dst(dst);
  return stream->ThenMemcpy(&wrapped_dst, src, bytes).ok();
}

// Type traits to get CUDA complex types from std::complex<>.
template <typename T>
struct CUDAComplexT {
  typedef T type;
};
template <>
struct CUDAComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct CUDAComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};
// Converts pointers of std::complex<> to pointers of
// cuComplex/cuDoubleComplex. No type conversion for non-complex types.
template <typename T>
inline const typename CUDAComplexT<T>::type* CUDAComplex(const T* p) {
  return reinterpret_cast<const typename CUDAComplexT<T>::type*>(p);
}
template <typename T>
inline typename CUDAComplexT<T>::type* CUDAComplex(T* p) {
  return reinterpret_cast<typename CUDAComplexT<T>::type*>(p);
}

// A set of initialized handles to the underlying Cuda libraries used by
// CudaSolver. We maintain one such set of handles per unique stream.
struct CudaSolverHandles {
  explicit CudaSolverHandles(cudaStream_t stream) {
    CHECK(cusolverDnCreate(&cusolver_dn_handle) == CUSOLVER_STATUS_SUCCESS)
        << "Failed to create cuSolverDN instance.";
    CHECK(cusolverDnSetStream(cusolver_dn_handle, stream) ==
          CUSOLVER_STATUS_SUCCESS)
        << "Failed to set cuSolverDN stream.";
    CHECK(cublasCreate(&cublas_handle) == CUBLAS_STATUS_SUCCESS)
        << "Failed to create cuBlas instance.";
    CHECK(cublasSetStream(cublas_handle, stream) == CUBLAS_STATUS_SUCCESS)
        << "Failed to set cuBlas stream.";
  }

  ~CudaSolverHandles() {
    CHECK(cublasDestroy(cublas_handle) == CUBLAS_STATUS_SUCCESS)
        << "Failed to destroy cuBlas instance.";
    CHECK(cusolverDnDestroy(cusolver_dn_handle) == CUSOLVER_STATUS_SUCCESS)
        << "Failed to destroy cuSolverDN instance.";
  }
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_dn_handle;
};

static mutex handle_map_mutex(LINKER_INITIALIZED);

using HandleMap =
    std::unordered_map<cudaStream_t, std::unique_ptr<CudaSolverHandles>>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

#define TF_RETURN_IF_CUSOLVER_ERROR(expr)                                      \
  do {                                                                         \
    auto status = (expr);                                                      \
    if (TF_PREDICT_FALSE(status != CUSOLVER_STATUS_SUCCESS)) {                 \
      return errors::Internal("cuSolverDN call failed with status =", status); \
    }                                                                          \
  } while (0)

#define TF_RETURN_IF_CUBLAS_ERROR(expr)                                \
  do {                                                                 \
    auto status = (expr);                                              \
    if (TF_PREDICT_FALSE(status != CUBLAS_STATUS_SUCCESS)) {           \
      return errors::Internal("cuBlas call failed status = ", status); \
    }                                                                  \
  } while (0)

CudaSolver::CudaSolver(OpKernelContext* context) : context_(context) {
  const cudaStream_t* cu_stream_ptr = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));
  cuda_stream_ = *cu_stream_ptr;
  HandleMap* handle_map = CHECK_NOTNULL(GetHandleMapSingleton());
  mutex_lock lock(handle_map_mutex);
  auto it = handle_map->find(cuda_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating CudaSolver handles for stream " << cuda_stream_;
    // Previously unseen Cuda stream. Initialize a set of Cuda solver library
    // handles for it.
    std::unique_ptr<CudaSolverHandles> new_handles(
        new CudaSolverHandles(cuda_stream_));
    it =
        handle_map->insert(std::make_pair(cuda_stream_, std::move(new_handles)))
            .first;
  }
  cusolver_dn_handle_ = it->second->cusolver_dn_handle;
  cublas_handle_ = it->second->cublas_handle;
}

Status CudaSolver::CopyLapackInfoToHostAsync(
    const std::vector<DeviceLapackInfo>& dev_lapack_infos,
    std::function<void(const Status&, const std::vector<HostLapackInfo>&)>
        info_checker_callback) const {
  std::vector<HostLapackInfo> host_lapack_infos;
  if (dev_lapack_infos.empty()) {
    info_checker_callback(Status::OK(), host_lapack_infos);
    return Status::OK();
  }

  // Launch memcpys to copy info back from the device to the host.
  for (const auto& dev_lapack_info : dev_lapack_infos) {
    bool success = true;
    auto host_copy = dev_lapack_info.CopyToHost(&success);
    if (!success) {
      return errors::Internal(
          "Failed to launch copy of dev_lapack_info to host, debug_info = ",
          dev_lapack_info.debug_info());
    }
    host_lapack_infos.push_back(std::move(host_copy));
  }

  // This callback checks that all batch items in all calls were processed
  // successfully and passes status to the info_checker_callback accordingly.
  auto wrapped_info_checker_callback =
      [info_checker_callback](std::vector<HostLapackInfo> host_lapack_infos) {
        Status status;
        for (const auto& host_lapack_info : host_lapack_infos) {
          for (int i = 0; i < host_lapack_info.size() && status.ok(); ++i) {
            const int info_value = (host_lapack_info.data())[i];
            if (info_value != 0) {
              status = errors::InvalidArgument(
                  "Got info = ", info_value, " for batch index ", i,
                  ", expected info = 0. Debug_info =",
                  host_lapack_info.debug_info());
            }
          }
          if (!status.ok()) {
            break;
          }
        }
        info_checker_callback(status, host_lapack_infos);
      };
  auto cb =
      std::bind(wrapped_info_checker_callback, std::move(host_lapack_infos));
  auto stream = context_->op_device_context()->stream();
  context_->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      stream, std::move(cb));
  return Status::OK();
}

// Macro that specializes a solver method for all 4 standard
// numeric types.
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct cusolverDn method names.
#define DN_SOLVER_FN(method, lapack_prefix) cusolverDn##lapack_prefix##method
#define DN_SOLVER_NAME(method, lapack_prefix) \
  "cusolverDn" #lapack_prefix #method
#define DN_BUFSIZE_FN(method, lapack_prefix) \
  cusolverDn##lapack_prefix##method##_bufferSize

// Macros to construct cublas method names.
#define BLAS_SOLVER_FN(method, lapack_prefix) cublas##lapack_prefix##method
#define BLAS_SOLVER_NAME(method, lapack_prefix) "cublas" #lapack_prefix #method

//=============================================================================
// Wrappers of cuSolverDN computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cusolver_api.h header file.
//=============================================================================
template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status PotrfImpl(BufSizeFnT bufsize, SolverFnT solver,
                               OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               cublasFillMode_t uplo, int n, Scalar* A, int lda,
                               int* dev_lapack_info) {
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(
      bufsize(cusolver_dn_handle, uplo, n, CUDAComplex(A), lda, &lwork));
  /* Allocate device memory for workspace. */
  ScratchSpace<Scalar> dev_workspace(context, lwork, /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, uplo, n, CUDAComplex(A), lda,
      CUDAComplex(dev_workspace.mutable_data()), lwork, dev_lapack_info));
  return Status::OK();
}

#define POTRF_INSTANCE(Scalar, lapack_prefix)                                \
  template <>                                                                \
  Status CudaSolver::Potrf<Scalar>(cublasFillMode_t uplo, int n, Scalar* A,  \
                                   int lda, int* dev_lapack_info) const {    \
    return PotrfImpl(DN_BUFSIZE_FN(potrf, lapack_prefix),                    \
                     DN_SOLVER_FN(potrf, lapack_prefix), context_,           \
                     cusolver_dn_handle_, uplo, n, A, lda, dev_lapack_info); \
  }

TF_CALL_LAPACK_TYPES(POTRF_INSTANCE);

//=============================================================================
// Wrappers of cuBlas computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cublas_api.h header file.
//=============================================================================
template <typename Scalar, typename SolverFnT>
static inline Status GetrfBatchedImpl(
    SolverFnT solver, OpKernelContext* context, cublasHandle_t cublas_handle,
    int n, const Scalar* host_a_dev_ptrs[], int lda, int* dev_pivots,
    DeviceLapackInfo* dev_lapack_info, int batch_size) {
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs(context, sizeof(CudaScalar*) * batch_size,
                                     /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes())) {
    return errors::Internal("GetrfBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(
      solver(cublas_handle, n, (CudaScalar**)dev_a_dev_ptrs.mutable_data(), lda,
             dev_pivots, dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define GETRF_BATCHED_INSTANCE(Scalar, lapack_prefix)                          \
  template <>                                                                  \
  Status CudaSolver::GetrfBatched(                                             \
      int n, const Scalar* host_a_dev_ptrs[], int lda, int* dev_pivots,        \
      DeviceLapackInfo* dev_lapack_info, int batch_size) const {               \
    return GetrfBatchedImpl(BLAS_SOLVER_FN(getrfBatched, lapack_prefix),       \
                            context_, cublas_handle_, n, host_a_dev_ptrs, lda, \
                            dev_pivots, dev_lapack_info, batch_size);          \
  }

TF_CALL_LAPACK_TYPES(GETRF_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status GetriBatchedImpl(
    SolverFnT solver, OpKernelContext* context, cublasHandle_t cublas_handle,
    int n, const Scalar* host_a_dev_ptrs[], int lda, const int* dev_pivots,
    const Scalar* host_a_inv_dev_ptrs[], int ldainv,
    DeviceLapackInfo* dev_lapack_info, int batch_size) {
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs(context, sizeof(CudaScalar*) * batch_size,
                                     /* on_host */ false);
  ScratchSpace<uint8> dev_a_inv_dev_ptrs(
      context, sizeof(CudaScalar*) * batch_size, /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes()) ||
      !CopyHostToDevice(context, dev_a_inv_dev_ptrs.mutable_data(),
                        host_a_inv_dev_ptrs, dev_a_inv_dev_ptrs.bytes())) {
    return errors::Internal("GetriBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(
      solver(cublas_handle, n, (const CudaScalar**)dev_a_dev_ptrs.data(), lda,
             dev_pivots, (CudaScalar**)dev_a_inv_dev_ptrs.mutable_data(),
             ldainv, dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define GETRI_BATCHED_INSTANCE(Scalar, lapack_prefix)                          \
  template <>                                                                  \
  Status CudaSolver::GetriBatched(                                             \
      int n, const Scalar* host_a_dev_ptrs[], int lda, const int* dev_pivots,  \
      const Scalar* host_a_inv_dev_ptrs[], int ldainv,                         \
      DeviceLapackInfo* dev_lapack_info, int batch_size) const {               \
    return GetriBatchedImpl(BLAS_SOLVER_FN(getriBatched, lapack_prefix),       \
                            context_, cublas_handle_, n, host_a_dev_ptrs, lda, \
                            dev_pivots, host_a_inv_dev_ptrs, ldainv,           \
                            dev_lapack_info, batch_size);                      \
  }

TF_CALL_LAPACK_TYPES(GETRI_BATCHED_INSTANCE);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
