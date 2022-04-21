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
#include <chrono>
#include <complex>
#include <unordered_map>
#include <vector>

#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
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
#include "tensorflow/core/util/gpu_solvers.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"

// The CUDA cublas_api.h API contains const-correctness errors. Instead of
// casting away constness on our data, we instead reinterpret the CuBLAS
// functions as what they were clearly meant to be, and thus we can call
// the functions naturally.
//
// (The error is that input-only arrays are bound to parameter types
// "const T**" instead of the correct "const T* const*".)
extern "C" {
using getrs_S = cublasStatus_t(cublasContext*, cublasOperation_t, int, int,
                               const float* const*, int, const int*, float**,
                               int, int*, int);
using getrs_D = cublasStatus_t(cublasContext*, cublasOperation_t, int, int,
                               const double* const*, int, const int*, double**,
                               int, int*, int);
using getrs_C = cublasStatus_t(cublasContext*, cublasOperation_t, int, int,
                               const float2* const*, int, const int*, float2**,
                               int, int*, int);
using getrs_Z = cublasStatus_t(cublasContext*, cublasOperation_t, int, int,
                               const double2* const*, int, const int*,
                               double2**, int, int*, int);

using getri_S = cublasStatus_t(cublasContext*, int, const float* const*, int,
                               const int*, float**, int, int*, int);
using getri_D = cublasStatus_t(cublasContext*, int, const double* const*, int,
                               const int*, double**, int, int*, int);
using getri_C = cublasStatus_t(cublasContext*, int, const float2* const*, int,
                               const int*, float2**, int, int*, int);
using getri_Z = cublasStatus_t(cublasContext*, int, const double2* const*, int,
                               const int*, double2**, int, int*, int);

using matinv_S = cublasStatus_t(cublasContext*, int, const float* const*, int,
                                float**, int, int*, int);
using matinv_D = cublasStatus_t(cublasContext*, int, const double* const*, int,
                                double**, int, int*, int);
using matinv_C = cublasStatus_t(cublasContext*, int, const float2* const*, int,
                                float2**, int, int*, int);
using matinv_Z = cublasStatus_t(cublasContext*, int, const double2* const*, int,
                                double2**, int, int*, int);

using trsm_S = cublasStatus_t(cublasContext*, cublasSideMode_t,
                              cublasFillMode_t, cublasOperation_t,
                              cublasDiagType_t, int, int, const float*,
                              const float* const*, int, float* const*, int,
                              int);
using trsm_D = cublasStatus_t(cublasContext*, cublasSideMode_t,
                              cublasFillMode_t, cublasOperation_t,
                              cublasDiagType_t, int, int, const double*,
                              const double* const*, int, double* const*, int,
                              int);
using trsm_C = cublasStatus_t(cublasContext*, cublasSideMode_t,
                              cublasFillMode_t, cublasOperation_t,
                              cublasDiagType_t, int, int, const float2*,
                              const float2* const*, int, float2* const*, int,
                              int);
using trsm_Z = cublasStatus_t(cublasContext*, cublasSideMode_t,
                              cublasFillMode_t, cublasOperation_t,
                              cublasDiagType_t, int, int, const double2*,
                              const double2* const*, int, double2* const*, int,
                              int);
}

namespace tensorflow {
namespace {

using se::cuda::ScopedActivateExecutorContext;

inline bool CopyHostToDevice(OpKernelContext* context, void* dst,
                             const void* src, uint64 bytes) {
  auto stream = context->op_device_context()->stream();
  se::DeviceMemoryBase wrapped_dst(dst);
  return stream->ThenMemcpy(&wrapped_dst, src, bytes).ok();
}

// A set of initialized handles to the underlying Cuda libraries used by
// GpuSolver. We maintain one such set of handles per unique stream.
struct GpuSolverHandles {
  explicit GpuSolverHandles(cudaStream_t stream) {
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

  ~GpuSolverHandles() {
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
    std::unordered_map<cudaStream_t, std::unique_ptr<GpuSolverHandles>>;

// Returns a singleton map used for storing initialized handles for each unique
// cuda stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

}  // namespace

#define TF_RETURN_IF_CUSOLVER_ERROR(expr)                      \
  do {                                                         \
    auto status = (expr);                                      \
    if (TF_PREDICT_FALSE(status != CUSOLVER_STATUS_SUCCESS)) { \
      return errors::Internal(                                 \
          __FILE__, ":", __LINE__,                             \
          ": cuSolverDN call failed with status =", status);   \
    }                                                          \
  } while (0)

#define TF_RETURN_IF_CUBLAS_ERROR(expr)                                  \
  do {                                                                   \
    auto status = (expr);                                                \
    if (TF_PREDICT_FALSE(status != CUBLAS_STATUS_SUCCESS)) {             \
      return errors::Internal(__FILE__, ":", __LINE__,                   \
                              ": cuBlas call failed status = ", status); \
    }                                                                    \
  } while (0)

GpuSolver::GpuSolver(OpKernelContext* context) : context_(context) {
  mutex_lock lock(handle_map_mutex);
  const cudaStream_t* cu_stream_ptr = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->GpuStreamMemberHack()));
  cuda_stream_ = *cu_stream_ptr;
  HandleMap* handle_map = CHECK_NOTNULL(GetHandleMapSingleton());
  auto it = handle_map->find(cuda_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating GpuSolver handles for stream " << cuda_stream_;
    // Previously unseen Cuda stream. Initialize a set of Cuda solver library
    // handles for it.
    std::unique_ptr<GpuSolverHandles> new_handles(
        new GpuSolverHandles(cuda_stream_));
    it =
        handle_map->insert(std::make_pair(cuda_stream_, std::move(new_handles)))
            .first;
  }
  cusolver_dn_handle_ = it->second->cusolver_dn_handle;
  cublas_handle_ = it->second->cublas_handle;
}

GpuSolver::~GpuSolver() {
  for (const auto& tensor_ref : scratch_tensor_refs_) {
    tensor_ref.Unref();
  }
}

// static
void GpuSolver::CheckLapackInfoAndDeleteSolverAsync(
    std::unique_ptr<GpuSolver> solver,
    const std::vector<DeviceLapackInfo>& dev_lapack_infos,
    std::function<void(const Status&, const std::vector<HostLapackInfo>&)>
        info_checker_callback) {
  CHECK(info_checker_callback != nullptr);
  std::vector<HostLapackInfo> host_lapack_infos;
  if (dev_lapack_infos.empty()) {
    info_checker_callback(Status::OK(), host_lapack_infos);
    return;
  }

  // Launch memcpys to copy info back from the device to the host.
  for (const auto& dev_lapack_info : dev_lapack_infos) {
    bool success = true;
    auto host_copy = dev_lapack_info.CopyToHost(&success);
    OP_REQUIRES(
        solver->context(), success,
        errors::Internal(
            "Failed to launch copy of dev_lapack_info to host, debug_info = ",
            dev_lapack_info.debug_info()));
    host_lapack_infos.push_back(std::move(host_copy));
  }

  // This callback checks that all batch items in all calls were processed
  // successfully and passes status to the info_checker_callback accordingly.
  auto* stream = solver->context()->op_device_context()->stream();
  auto wrapped_info_checker_callback =
      [stream](
          GpuSolver* solver,
          std::function<void(const Status&, const std::vector<HostLapackInfo>&)>
              info_checker_callback,
          std::vector<HostLapackInfo> host_lapack_infos) {
        ScopedActivateExecutorContext scoped_activation{stream->parent()};
        Status status;
        for (const auto& host_lapack_info : host_lapack_infos) {
          for (int i = 0; i < host_lapack_info.size() && status.ok(); ++i) {
            const int info_value = host_lapack_info(i);
            if (info_value != 0) {
              status = errors::InvalidArgument(
                  "Got info = ", info_value, " for batch index ", i,
                  ", expected info = 0. Debug_info = ",
                  host_lapack_info.debug_info());
            }
          }
          if (!status.ok()) {
            break;
          }
        }
        // Delete solver to release temp tensor refs.
        delete solver;

        // Delegate further error checking to provided functor.
        info_checker_callback(status, host_lapack_infos);
      };
  // Note: An std::function cannot have unique_ptr arguments (it must be copy
  // constructible and therefore so must its arguments). Therefore, we release
  // solver into a raw pointer to be deleted at the end of
  // wrapped_info_checker_callback.
  // Release ownership of solver. It will be deleted in the cb callback.
  auto solver_raw_ptr = solver.release();
  auto cb =
      std::bind(wrapped_info_checker_callback, solver_raw_ptr,
                std::move(info_checker_callback), std::move(host_lapack_infos));

  solver_raw_ptr->context()
      ->device()
      ->tensorflow_accelerator_device_info()
      ->event_mgr->ThenExecute(stream, std::move(cb));
}

// static
void GpuSolver::CheckLapackInfoAndDeleteSolverAsync(
    std::unique_ptr<GpuSolver> solver,
    const std::vector<DeviceLapackInfo>& dev_lapack_info,
    AsyncOpKernel::DoneCallback done) {
  OpKernelContext* context = solver->context();
  auto wrapped_done = [context, done](
                          const Status& status,
                          const std::vector<HostLapackInfo>& /* unused */) {
    if (done != nullptr) {
      OP_REQUIRES_OK_ASYNC(context, status, done);
      done();
    } else {
      OP_REQUIRES_OK(context, status);
    }
  };
  CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_lapack_info,
                                      wrapped_done);
}

// Allocates a temporary tensor. The GpuSolver object maintains a
// TensorReference to the underlying Tensor to prevent it from being deallocated
// prematurely.
Status GpuSolver::allocate_scoped_tensor(DataType type,
                                         const TensorShape& shape,
                                         Tensor* out_temp) {
  const Status status = context_->allocate_temp(type, shape, out_temp);
  if (status.ok()) {
    scratch_tensor_refs_.emplace_back(*out_temp);
  }
  return status;
}

Status GpuSolver::forward_input_or_allocate_scoped_tensor(
    gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, Tensor* out_temp) {
  const Status status = context_->forward_input_or_allocate_temp(
      candidate_input_indices, type, shape, out_temp);
  if (status.ok()) {
    scratch_tensor_refs_.emplace_back(*out_temp);
  }
  return status;
}

// Macro that specializes a solver method for all 4 standard
// numeric types.
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)
#define TF_CALL_LAPACK_TYPES_NO_COMPLEX(m) m(float, S) m(double, D)

// Macros to construct cusolverDn method names.
#define DN_SOLVER_FN(method, type_prefix) cusolverDn##type_prefix##method
#define DN_SOLVER_NAME(method, type_prefix) "cusolverDn" #type_prefix #method
#define DN_BUFSIZE_FN(method, type_prefix) \
  cusolverDn##type_prefix##method##_bufferSize

// Macros to construct cublas method names.
#define BLAS_SOLVER_FN(method, type_prefix) cublas##type_prefix##method
#define BLAS_SOLVER_NAME(method, type_prefix) "cublas" #type_prefix #method

//=============================================================================
// Wrappers of cuSolverDN computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cusolver_api.h header file.
//
// NOTE: The cuSolver functions called below appear not to be threadsafe.
// so we put a global lock around the calls. Since these functions only put a
// kernel on the shared stream, it is not a big performance hit.
// TODO(rmlarsen): Investigate if the locking is still needed in Cuda 9.
//=============================================================================

template <typename Scalar, typename SolverFnT>
static inline Status GeamImpl(SolverFnT solver, cublasHandle_t cublas_handle,
                              cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n,
                              const Scalar* alpha, /* host or device pointer */
                              const Scalar* A, int lda,
                              const Scalar* beta, /* host or device pointer */
                              const Scalar* B, int ldb, Scalar* C, int ldc) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  TF_RETURN_IF_CUBLAS_ERROR(solver(cublas_handle, transa, transb, m, n,
                                   reinterpret_cast<const CudaScalar*>(alpha),
                                   reinterpret_cast<const CudaScalar*>(A), lda,
                                   reinterpret_cast<const CudaScalar*>(beta),
                                   reinterpret_cast<const CudaScalar*>(B), ldb,
                                   reinterpret_cast<CudaScalar*>(C), ldc));
  return Status::OK();
}

#define GEAM_INSTANCE(Scalar, type_prefix)                                     \
  template <>                                                                  \
  Status GpuSolver::Geam<Scalar>(                                              \
      cublasOperation_t transa, cublasOperation_t transb, int m, int n,        \
      const Scalar* alpha, /* host or device pointer */                        \
      const Scalar* A, int lda,                                                \
      const Scalar* beta, /* host or device pointer */                         \
      const Scalar* B, int ldb, Scalar* C, int ldc) const {                    \
    return GeamImpl(BLAS_SOLVER_FN(geam, type_prefix), cublas_handle_, transa, \
                    transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);        \
  }

TF_CALL_LAPACK_TYPES(GEAM_INSTANCE);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status PotrfImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               cublasFillMode_t uplo, int n, Scalar* A, int lda,
                               int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(
      bufsize(cusolver_dn_handle, uplo, n, CUDAComplex(A), lda, &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, uplo, n, CUDAComplex(A), lda,
      CUDAComplex(dev_workspace.mutable_data()), lwork, dev_lapack_info));
  return Status::OK();
}

#define POTRF_INSTANCE(Scalar, type_prefix)                                  \
  template <>                                                                \
  Status GpuSolver::Potrf<Scalar>(cublasFillMode_t uplo, int n, Scalar* A,   \
                                  int lda, int* dev_lapack_info) {           \
    return PotrfImpl(DN_BUFSIZE_FN(potrf, type_prefix),                      \
                     DN_SOLVER_FN(potrf, type_prefix), this, context_,       \
                     cusolver_dn_handle_, uplo, n, A, lda, dev_lapack_info); \
  }

TF_CALL_LAPACK_TYPES(POTRF_INSTANCE);

#if CUDA_VERSION >= 9020
template <typename Scalar, typename SolverFnT>
static inline Status PotrfBatchedImpl(
    SolverFnT solver, GpuSolver* cuda_solver, OpKernelContext* context,
    cusolverDnHandle_t cusolver_dn_handle, cublasFillMode_t uplo, int n,
    const Scalar* const host_a_dev_ptrs[], int lda,
    DeviceLapackInfo* dev_lapack_info, int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes())) {
    return errors::Internal("PotrfBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUSOLVER_ERROR(
      solver(cusolver_dn_handle, uplo, n,
             reinterpret_cast<CudaScalar**>(dev_a_dev_ptrs.mutable_data()), lda,
             dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define POTRF_BATCHED_INSTANCE(Scalar, type_prefix)                        \
  template <>                                                              \
  Status GpuSolver::PotrfBatched(                                          \
      cublasFillMode_t uplo, int n, const Scalar* const host_a_dev_ptrs[], \
      int lda, DeviceLapackInfo* dev_lapack_info, int batch_size) {        \
    return PotrfBatchedImpl(DN_SOLVER_FN(potrfBatched, type_prefix), this, \
                            context_, cusolver_dn_handle_, uplo, n,        \
                            host_a_dev_ptrs, lda, dev_lapack_info,         \
                            batch_size);                                   \
  }

TF_CALL_LAPACK_TYPES(POTRF_BATCHED_INSTANCE);
#endif  // CUDA_VERSION >= 9020

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status GetrfImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle, int m,
                               int n, Scalar* A, int lda, int* dev_pivots,
                               int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(
      bufsize(cusolver_dn_handle, m, n, CUDAComplex(A), lda, &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, m, n, CUDAComplex(A), lda,
      CUDAComplex(dev_workspace.mutable_data()), dev_pivots, dev_lapack_info));
  return Status::OK();
}

#define GETRF_INSTANCE(Scalar, type_prefix)                                \
  template <>                                                              \
  Status GpuSolver::Getrf<Scalar>(int m, int n, Scalar* A, int lda,        \
                                  int* dev_pivots, int* dev_lapack_info) { \
    return GetrfImpl(DN_BUFSIZE_FN(getrf, type_prefix),                    \
                     DN_SOLVER_FN(getrf, type_prefix), this, context_,     \
                     cusolver_dn_handle_, m, n, A, lda, dev_pivots,        \
                     dev_lapack_info);                                     \
  }

TF_CALL_LAPACK_TYPES(GETRF_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status GetrsImpl(SolverFnT solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               cublasOperation_t trans, int n, int nrhs,
                               const Scalar* A, int lda, const int* pivots,
                               Scalar* B, int ldb, int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(cusolver_dn_handle, trans, n, nrhs,
                                     CUDAComplex(A), lda, pivots,
                                     CUDAComplex(B), ldb, dev_lapack_info));
  return Status::OK();
}

#define GETRS_INSTANCE(Scalar, type_prefix)                                  \
  template <>                                                                \
  Status GpuSolver::Getrs<Scalar>(                                           \
      cublasOperation_t trans, int n, int nrhs, const Scalar* A, int lda,    \
      const int* pivots, Scalar* B, int ldb, int* dev_lapack_info) const {   \
    return GetrsImpl(DN_SOLVER_FN(getrs, type_prefix), context_,             \
                     cusolver_dn_handle_, trans, n, nrhs, A, lda, pivots, B, \
                     ldb, dev_lapack_info);                                  \
  }

TF_CALL_LAPACK_TYPES(GETRS_INSTANCE);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status GeqrfImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle, int m,
                               int n, Scalar* A, int lda, Scalar* tau,
                               int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(
      bufsize(cusolver_dn_handle, m, n, CUDAComplex(A), lda, &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, m, n, CUDAComplex(A), lda, CUDAComplex(tau),
      CUDAComplex(dev_workspace.mutable_data()), lwork, dev_lapack_info));
  return Status::OK();
}

#define GEQRF_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Geqrf<Scalar>(int m, int n, Scalar* A, int lda,            \
                                  Scalar* tau, int* dev_lapack_info) {         \
    return GeqrfImpl(DN_BUFSIZE_FN(geqrf, type_prefix),                        \
                     DN_SOLVER_FN(geqrf, type_prefix), this, context_,         \
                     cusolver_dn_handle_, m, n, A, lda, tau, dev_lapack_info); \
  }

TF_CALL_LAPACK_TYPES(GEQRF_INSTANCE);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status UnmqrImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               cublasSideMode_t side, cublasOperation_t trans,
                               int m, int n, int k, const Scalar* dev_a,
                               int lda, const Scalar* dev_tau, Scalar* dev_c,
                               int ldc, int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(
      bufsize(cusolver_dn_handle, side, trans, m, n, k, CUDAComplex(dev_a), lda,
              CUDAComplex(dev_tau), CUDAComplex(dev_c), ldc, &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, side, trans, m, n, k, CUDAComplex(dev_a), lda,
      CUDAComplex(dev_tau), CUDAComplex(dev_c), ldc,
      CUDAComplex(dev_workspace.mutable_data()), lwork, dev_lapack_info));
  return Status::OK();
}

// Unfortunately the LAPACK function name differs for the real and complex case
// (complex ones are prefixed with "UN" for "unitary"), so we instantiate each
// one separately.
#define UNMQR_INSTANCE(Scalar, function_prefix, type_prefix)                 \
  template <>                                                                \
  Status GpuSolver::Unmqr(cublasSideMode_t side, cublasOperation_t trans,    \
                          int m, int n, int k, const Scalar* dev_a, int lda, \
                          const Scalar* dev_tau, Scalar* dev_c, int ldc,     \
                          int* dev_lapack_info) {                            \
    return UnmqrImpl(DN_BUFSIZE_FN(function_prefix##mqr, type_prefix),       \
                     DN_SOLVER_FN(function_prefix##mqr, type_prefix), this,  \
                     context_, cusolver_dn_handle_, side, trans, m, n, k,    \
                     dev_a, lda, dev_tau, dev_c, ldc, dev_lapack_info);      \
  }

UNMQR_INSTANCE(float, or, S);
UNMQR_INSTANCE(double, or, D);
UNMQR_INSTANCE(complex64, un, C);
UNMQR_INSTANCE(complex128, un, Z);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status UngqrImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle, int m,
                               int n, int k, Scalar* dev_a, int lda,
                               const Scalar* dev_tau, int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(bufsize(cusolver_dn_handle, m, n, k,
                                      CUDAComplex(dev_a), lda,
                                      CUDAComplex(dev_tau), &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(
      solver(cusolver_dn_handle, m, n, k, CUDAComplex(dev_a), lda,
             CUDAComplex(dev_tau), CUDAComplex(dev_workspace.mutable_data()),
             lwork, dev_lapack_info));
  return Status::OK();
}

#define UNGQR_INSTANCE(Scalar, function_prefix, type_prefix)                \
  template <>                                                               \
  Status GpuSolver::Ungqr(int m, int n, int k, Scalar* dev_a, int lda,      \
                          const Scalar* dev_tau, int* dev_lapack_info) {    \
    return UngqrImpl(DN_BUFSIZE_FN(function_prefix##gqr, type_prefix),      \
                     DN_SOLVER_FN(function_prefix##gqr, type_prefix), this, \
                     context_, cusolver_dn_handle_, m, n, k, dev_a, lda,    \
                     dev_tau, dev_lapack_info);                             \
  }

UNGQR_INSTANCE(float, or, S);
UNGQR_INSTANCE(double, or, D);
UNGQR_INSTANCE(complex64, un, C);
UNGQR_INSTANCE(complex128, un, Z);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status HeevdImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               cusolverEigMode_t jobz, cublasFillMode_t uplo,
                               int n, Scalar* dev_A, int lda,
                               typename Eigen::NumTraits<Scalar>::Real* dev_W,
                               int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(bufsize(cusolver_dn_handle, jobz, uplo, n,
                                      CUDAComplex(dev_A), lda,
                                      CUDAComplex(dev_W), &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  /* Launch the solver kernel. */
  TF_RETURN_IF_CUSOLVER_ERROR(
      solver(cusolver_dn_handle, jobz, uplo, n, CUDAComplex(dev_A), lda,
             CUDAComplex(dev_W), CUDAComplex(dev_workspace.mutable_data()),
             lwork, dev_lapack_info));
  return Status::OK();
}

#define HEEVD_INSTANCE(Scalar, function_prefix, type_prefix)                   \
  template <>                                                                  \
  Status GpuSolver::Heevd(cusolverEigMode_t jobz, cublasFillMode_t uplo,       \
                          int n, Scalar* dev_A, int lda,                       \
                          typename Eigen::NumTraits<Scalar>::Real* dev_W,      \
                          int* dev_lapack_info) {                              \
    return HeevdImpl(DN_BUFSIZE_FN(function_prefix##evd, type_prefix),         \
                     DN_SOLVER_FN(function_prefix##evd, type_prefix), this,    \
                     context_, cusolver_dn_handle_, jobz, uplo, n, dev_A, lda, \
                     dev_W, dev_lapack_info);                                  \
  }

HEEVD_INSTANCE(float, sy, S);
HEEVD_INSTANCE(double, sy, D);
HEEVD_INSTANCE(complex64, he, C);
HEEVD_INSTANCE(complex128, he, Z);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status GesvdImpl(BufSizeFnT bufsize, SolverFnT solver,
                               GpuSolver* cuda_solver, OpKernelContext* context,
                               cusolverDnHandle_t cusolver_dn_handle,
                               signed char jobu, signed char jobvt, int m,
                               int n, Scalar* A, int lda, Scalar* S, Scalar* U,
                               int ldu, Scalar* VT, int ldvt,
                               int* dev_lapack_info) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  TF_RETURN_IF_CUSOLVER_ERROR(bufsize(cusolver_dn_handle, m, n, &lwork));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  TF_RETURN_IF_CUSOLVER_ERROR(solver(cusolver_dn_handle, jobu, jobvt, m, n,
                                     CUDAComplex(A), lda, S, CUDAComplex(U),
                                     ldu, CUDAComplex(VT), ldvt,
                                     CUDAComplex(dev_workspace.mutable_data()),
                                     lwork, nullptr, dev_lapack_info));
  return Status::OK();
}

#define GESVD_INSTANCE(Scalar, type_prefix)                              \
  template <>                                                            \
  Status GpuSolver::Gesvd<Scalar>(                                       \
      signed char jobu, signed char jobvt, int m, int n, Scalar* dev_A,  \
      int lda, Scalar* dev_S, Scalar* dev_U, int ldu, Scalar* dev_VT,    \
      int ldvt, int* dev_lapack_info) {                                  \
    return GesvdImpl(DN_BUFSIZE_FN(gesvd, type_prefix),                  \
                     DN_SOLVER_FN(gesvd, type_prefix), this, context_,   \
                     cusolver_dn_handle_, jobu, jobvt, m, n, dev_A, lda, \
                     dev_S, dev_U, ldu, dev_VT, ldvt, dev_lapack_info);  \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(GESVD_INSTANCE);

template <typename Scalar, typename BufSizeFnT, typename SolverFnT>
static inline Status GesvdjBatchedImpl(BufSizeFnT bufsize, SolverFnT solver,
                                       GpuSolver* cuda_solver,
                                       OpKernelContext* context,
                                       cusolverDnHandle_t cusolver_dn_handle,
                                       cusolverEigMode_t jobz, int m, int n,
                                       Scalar* A, int lda, Scalar* S, Scalar* U,
                                       int ldu, Scalar* V, int ldv,
                                       int* dev_lapack_info, int batch_size) {
  mutex_lock lock(handle_map_mutex);
  /* Get amount of workspace memory required. */
  int lwork;
  /* Default parameters for gesvdj and gesvdjBatched. */
  gesvdjInfo_t svdj_info;
  TF_RETURN_IF_CUSOLVER_ERROR(cusolverDnCreateGesvdjInfo(&svdj_info));
  TF_RETURN_IF_CUSOLVER_ERROR(bufsize(
      cusolver_dn_handle, jobz, m, n, CUDAComplex(A), lda, S, CUDAComplex(U),
      ldu, CUDAComplex(V), ldv, &lwork, svdj_info, batch_size));
  /* Allocate device memory for workspace. */
  auto dev_workspace =
      cuda_solver->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);
  TF_RETURN_IF_CUSOLVER_ERROR(solver(
      cusolver_dn_handle, jobz, m, n, CUDAComplex(A), lda, S, CUDAComplex(U),
      ldu, CUDAComplex(V), ldv, CUDAComplex(dev_workspace.mutable_data()),
      lwork, dev_lapack_info, svdj_info, batch_size));
  TF_RETURN_IF_CUSOLVER_ERROR(cusolverDnDestroyGesvdjInfo(svdj_info));
  return Status::OK();
}

#define GESVDJBATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                  \
  Status GpuSolver::GesvdjBatched<Scalar>(                                     \
      cusolverEigMode_t jobz, int m, int n, Scalar* dev_A, int lda,            \
      Scalar* dev_S, Scalar* dev_U, int ldu, Scalar* dev_V, int ldv,           \
      int* dev_lapack_info, int batch_size) {                                  \
    return GesvdjBatchedImpl(DN_BUFSIZE_FN(gesvdjBatched, type_prefix),        \
                             DN_SOLVER_FN(gesvdjBatched, type_prefix), this,   \
                             context_, cusolver_dn_handle_, jobz, m, n, dev_A, \
                             lda, dev_S, dev_U, ldu, dev_V, ldv,               \
                             dev_lapack_info, batch_size);                     \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(GESVDJBATCHED_INSTANCE);

//=============================================================================
// Wrappers of cuBlas computational methods begin here.
//
// WARNING to implementers: The function signatures listed in the online docs
// are sometimes inaccurate, e.g., are missing 'const' on pointers
// to immutable arguments, while the actual headers have them as expected.
// Check the actual declarations in the cublas_api.h header file.
//=============================================================================
template <typename Scalar, typename SolverFnT>
static inline Status GetrfBatchedImpl(SolverFnT solver, GpuSolver* cuda_solver,
                                      OpKernelContext* context,
                                      cublasHandle_t cublas_handle, int n,
                                      const Scalar* const host_a_dev_ptrs[],
                                      int lda, int* dev_pivots,
                                      DeviceLapackInfo* dev_lapack_info,
                                      int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes())) {
    return errors::Internal("GetrfBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(
      solver(cublas_handle, n,
             reinterpret_cast<CudaScalar**>(dev_a_dev_ptrs.mutable_data()), lda,
             dev_pivots, dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define GETRF_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                  \
  Status GpuSolver::GetrfBatched(                                              \
      int n, const Scalar* const host_a_dev_ptrs[], int lda, int* dev_pivots,  \
      DeviceLapackInfo* dev_lapack_info, int batch_size) {                     \
    return GetrfBatchedImpl(BLAS_SOLVER_FN(getrfBatched, type_prefix), this,   \
                            context_, cublas_handle_, n, host_a_dev_ptrs, lda, \
                            dev_pivots, dev_lapack_info, batch_size);          \
  }

TF_CALL_LAPACK_TYPES(GETRF_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status GetrsBatchedImpl(
    SolverFnT solver, GpuSolver* cuda_solver, OpKernelContext* context,
    cublasHandle_t cublas_handle, cublasOperation_t trans, int n, int nrhs,
    const Scalar* const host_a_dev_ptrs[], int lda, const int* dev_pivots,
    const Scalar* const host_b_dev_ptrs[], int ldb, int* host_lapack_info,
    int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  ScratchSpace<uint8> dev_b_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes())) {
    return errors::Internal("GetrsBatched: failed to copy pointers to device");
  }
  if (!CopyHostToDevice(context, dev_b_dev_ptrs.mutable_data() /* dest */,
                        host_b_dev_ptrs /* source */, dev_b_dev_ptrs.bytes())) {
    return errors::Internal("GetrsBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(solver(
      cublas_handle, trans, n, nrhs,
      reinterpret_cast<const CudaScalar* const*>(dev_a_dev_ptrs.data()), lda,
      dev_pivots, reinterpret_cast<CudaScalar**>(dev_b_dev_ptrs.mutable_data()),
      ldb, host_lapack_info, batch_size));
  return Status::OK();
}

#define GETRS_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                  \
  Status GpuSolver::GetrsBatched(                                              \
      cublasOperation_t trans, int n, int nrhs,                                \
      const Scalar* const host_a_dev_ptrs[], int lda, const int* dev_pivots,   \
      const Scalar* const host_b_dev_ptrs[], int ldb, int* host_lapack_info,   \
      int batch_size) {                                                        \
    return GetrsBatchedImpl(reinterpret_cast<getrs_##type_prefix*>(            \
                                BLAS_SOLVER_FN(getrsBatched, type_prefix)),    \
                            this, context_, cublas_handle_, trans, n, nrhs,    \
                            host_a_dev_ptrs, lda, dev_pivots, host_b_dev_ptrs, \
                            ldb, host_lapack_info, batch_size);                \
  }

TF_CALL_LAPACK_TYPES(GETRS_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status GetriBatchedImpl(
    SolverFnT solver, GpuSolver* cuda_solver, OpKernelContext* context,
    cublasHandle_t cublas_handle, int n, const Scalar* const host_a_dev_ptrs[],
    int lda, const int* dev_pivots, const Scalar* const host_a_inv_dev_ptrs[],
    int ldainv, DeviceLapackInfo* dev_lapack_info, int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  ScratchSpace<uint8> dev_a_inv_dev_ptrs = cuda_solver->GetScratchSpace<uint8>(
      sizeof(CudaScalar*) * batch_size, "", /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes()) ||
      !CopyHostToDevice(context, dev_a_inv_dev_ptrs.mutable_data(),
                        host_a_inv_dev_ptrs, dev_a_inv_dev_ptrs.bytes())) {
    return errors::Internal("GetriBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(
      solver(cublas_handle, n,
             reinterpret_cast<const CudaScalar* const*>(dev_a_dev_ptrs.data()),
             lda, dev_pivots,
             reinterpret_cast<CudaScalar**>(dev_a_inv_dev_ptrs.mutable_data()),
             ldainv, dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define GETRI_BATCHED_INSTANCE(Scalar, type_prefix)                          \
  template <>                                                                \
  Status GpuSolver::GetriBatched(                                            \
      int n, const Scalar* const host_a_dev_ptrs[], int lda,                 \
      const int* dev_pivots, const Scalar* const host_a_inv_dev_ptrs[],      \
      int ldainv, DeviceLapackInfo* dev_lapack_info, int batch_size) {       \
    return GetriBatchedImpl(                                                 \
        reinterpret_cast<getri_##type_prefix*>(                              \
            BLAS_SOLVER_FN(getriBatched, type_prefix)),                      \
        this, context_, cublas_handle_, n, host_a_dev_ptrs, lda, dev_pivots, \
        host_a_inv_dev_ptrs, ldainv, dev_lapack_info, batch_size);           \
  }

TF_CALL_LAPACK_TYPES(GETRI_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status MatInvBatchedImpl(
    SolverFnT solver, GpuSolver* cuda_solver, OpKernelContext* context,
    cublasHandle_t cublas_handle, int n, const Scalar* const host_a_dev_ptrs[],
    int lda, const Scalar* const host_a_inv_dev_ptrs[], int ldainv,
    DeviceLapackInfo* dev_lapack_info, int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  ScratchSpace<uint8> dev_a_inv_dev_ptrs = cuda_solver->GetScratchSpace<uint8>(
      sizeof(CudaScalar*) * batch_size, "", /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes()) ||
      !CopyHostToDevice(context, dev_a_inv_dev_ptrs.mutable_data(),
                        host_a_inv_dev_ptrs, dev_a_inv_dev_ptrs.bytes())) {
    return errors::Internal("MatInvBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(solver(
      cublas_handle, n,
      reinterpret_cast<const CudaScalar* const*>(dev_a_dev_ptrs.data()), lda,
      reinterpret_cast<CudaScalar**>(dev_a_inv_dev_ptrs.mutable_data()), ldainv,
      dev_lapack_info->mutable_data(), batch_size));
  return Status::OK();
}

#define MATINV_BATCHED_INSTANCE(Scalar, type_prefix)                          \
  template <>                                                                 \
  Status GpuSolver::MatInvBatched(                                            \
      int n, const Scalar* const host_a_dev_ptrs[], int lda,                  \
      const Scalar* const host_a_inv_dev_ptrs[], int ldainv,                  \
      DeviceLapackInfo* dev_lapack_info, int batch_size) {                    \
    return MatInvBatchedImpl(reinterpret_cast<matinv_##type_prefix*>(         \
                                 BLAS_SOLVER_FN(matinvBatched, type_prefix)), \
                             this, context_, cublas_handle_, n,               \
                             host_a_dev_ptrs, lda, host_a_inv_dev_ptrs,       \
                             ldainv, dev_lapack_info, batch_size);            \
  }

TF_CALL_LAPACK_TYPES(MATINV_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status TrsmImpl(SolverFnT solver, cublasHandle_t cublas_handle,
                              cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag,
                              int m, int n,
                              const Scalar* alpha, /* host or device pointer */
                              const Scalar* A, int lda, Scalar* B, int ldb) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  TF_RETURN_IF_CUBLAS_ERROR(solver(cublas_handle, side, uplo, trans, diag, m, n,
                                   reinterpret_cast<const CudaScalar*>(alpha),
                                   reinterpret_cast<const CudaScalar*>(A), lda,
                                   reinterpret_cast<CudaScalar*>(B), ldb));
  return Status::OK();
}

#define TRSM_INSTANCE(Scalar, type_prefix)                                   \
  template <>                                                                \
  Status GpuSolver::Trsm<Scalar>(                                            \
      cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, \
      cublasDiagType_t diag, int m, int n,                                   \
      const Scalar* alpha, /* host or device pointer */                      \
      const Scalar* A, int lda, Scalar* B, int ldb) {                        \
    return TrsmImpl(BLAS_SOLVER_FN(trsm, type_prefix), cublas_handle_, side, \
                    uplo, trans, diag, m, n, alpha, A, lda, B, ldb);         \
  }

TF_CALL_LAPACK_TYPES(TRSM_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status TrsvImpl(SolverFnT solver, cublasHandle_t cublas_handle,
                              cublasFillMode_t uplo, cublasOperation_t trans,
                              cublasDiagType_t diag, int n, const Scalar* A,
                              int lda, Scalar* x, int incx) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  TF_RETURN_IF_CUBLAS_ERROR(solver(cublas_handle, uplo, trans, diag, n,
                                   reinterpret_cast<const CudaScalar*>(A), lda,
                                   reinterpret_cast<CudaScalar*>(x), incx));
  return Status::OK();
}

#define TRSV_INSTANCE(Scalar, type_prefix)                                   \
  template <>                                                                \
  Status GpuSolver::Trsv<Scalar>(                                            \
      cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, \
      int n, const Scalar* A, int lda, Scalar* x, int incx) {                \
    return TrsvImpl(BLAS_SOLVER_FN(trsv, type_prefix), cublas_handle_, uplo, \
                    trans, diag, n, A, lda, x, incx);                        \
  }

TF_CALL_LAPACK_TYPES(TRSV_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status TrsmBatchedImpl(
    SolverFnT solver, GpuSolver* cuda_solver, OpKernelContext* context,
    cublasHandle_t cublas_handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const Scalar* alpha, const Scalar* const host_a_dev_ptrs[], int lda,
    Scalar* host_b_dev_ptrs[], int ldb, int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using CudaScalar = typename CUDAComplexT<Scalar>::type;
  ScratchSpace<uint8> dev_a_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  ScratchSpace<uint8> dev_b_dev_ptrs =
      cuda_solver->GetScratchSpace<uint8>(sizeof(CudaScalar*) * batch_size, "",
                                          /* on_host */ false);
  if (!CopyHostToDevice(context, dev_a_dev_ptrs.mutable_data() /* dest */,
                        host_a_dev_ptrs /* source */, dev_a_dev_ptrs.bytes())) {
    return errors::Internal("TrsmBatched: failed to copy pointers to device");
  }
  if (!CopyHostToDevice(context, dev_b_dev_ptrs.mutable_data() /* dest */,
                        host_b_dev_ptrs /* source */, dev_b_dev_ptrs.bytes())) {
    return errors::Internal("TrsmBatched: failed to copy pointers to device");
  }
  TF_RETURN_IF_CUBLAS_ERROR(
      solver(cublas_handle, side, uplo, trans, diag, m, n,
             reinterpret_cast<const CudaScalar*>(alpha),
             reinterpret_cast<const CudaScalar* const*>(dev_a_dev_ptrs.data()),
             lda, reinterpret_cast<CudaScalar**>(dev_b_dev_ptrs.mutable_data()),
             ldb, batch_size));
  return Status::OK();
}

#define TRSM_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                 \
  Status GpuSolver::TrsmBatched(                                              \
      cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,  \
      cublasDiagType_t diag, int m, int n, const Scalar* alpha,               \
      const Scalar* const dev_Aarray[], int lda, Scalar* dev_Barray[],        \
      int ldb, int batch_size) {                                              \
    return TrsmBatchedImpl(reinterpret_cast<trsm_##type_prefix*>(             \
                               BLAS_SOLVER_FN(trsmBatched, type_prefix)),     \
                           this, context_, cublas_handle_, side, uplo, trans, \
                           diag, m, n, alpha, dev_Aarray, lda, dev_Barray,    \
                           ldb, batch_size);                                  \
  }

TF_CALL_LAPACK_TYPES(TRSM_BATCHED_INSTANCE);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
