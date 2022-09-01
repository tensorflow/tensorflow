/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
/*
Mapping of GpuSolver Methods to respective ROCm Library APIs.
/-----------------------------------------------------------=-----------------------------/
/ GpuSolverMethod  //    rocblasAPI    //   rocsolverAPI    //   hipsolverAPI
(ROCM>4.5)  / / Geam             //  rocblas_Xgeam   //    ----           //
----                   / / Getrf            //      ----        //
rocsolver_Xgetrf  //   hipsolverXgetrf          / / GetrfBatched     // ---- //
""_Xgetrf_batched //     ----                   / / GetriBatched     // ---- //
""_Xgetri_batched //     ----                   / / Getrs            // ---- //
rocsolver_Xgetrs  //   hipsolverXgetrs          / / GetrsBatched     // ---- //
""_Xgetrs_batched //     ----                   / / Geqrf            // ---- //
rocsolver_Xgeqrf  //   hipsolverXgeqrf          / / Heevd            // ---- //
----           //   hipsolverXheevd          / / Potrf            //      ----
// rocsolver_Xpotrf  //   hipsolverXpotrf          / / PotrfBatched     // ----
// ""_Xpotrf_batched //   ""XpotrfBatched          / / Trsm             //
rocblas_Xtrsm   //    ----           //     ----                   / / Ungqr //
----        // rocsolver_Xungqr  //   hipsolverXungqr          / / Unmqr // ----
// rocsolver_Xunmqr  //   hipsolverXunmqr          /
/-----------------------------------------------------------------------------------------/
*/
#if TENSORFLOW_USE_ROCM
#include <complex>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/lib/env.h"
#include "tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocblas_wrapper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/blocking_counter.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {
namespace {

using stream_executor::gpu::GpuExecutor;
using stream_executor::gpu::ScopedActivateExecutorContext;

inline bool CopyHostToDevice(OpKernelContext* context, void* dst,
                             const void* src, uint64 bytes) {
  auto stream = context->op_device_context()->stream();
  se::DeviceMemoryBase wrapped_dst(dst);
  return stream->ThenMemcpy(&wrapped_dst, src, bytes).ok();
}

struct GpuSolverHandles {
  explicit GpuSolverHandles(GpuExecutor* parent, hipStream_t stream) {
    parent_ = parent;
    ScopedActivateExecutorContext sac{parent_};
#if TF_ROCM_VERSION >= 40500
    CHECK(wrap::hipsolverCreate(&hipsolver_handle) == rocblas_status_success)
        << "Failed to create hipsolver instance";
#endif
    CHECK(wrap::rocblas_create_handle(&rocm_blas_handle) ==
          rocblas_status_success)
        << "Failed to create rocBlas instance.";
    CHECK(wrap::rocblas_set_stream(rocm_blas_handle, stream) ==
          rocblas_status_success)
        << "Failed to set rocBlas stream.";
  }

  ~GpuSolverHandles() {
    ScopedActivateExecutorContext sac{parent_};
    CHECK(wrap::rocblas_destroy_handle(rocm_blas_handle) ==
          rocblas_status_success)
        << "Failed to destroy rocBlas instance.";
#if TF_ROCM_VERSION >= 40500
    CHECK(wrap::hipsolverDestroy(hipsolver_handle) == rocblas_status_success)
        << "Failed to destroy hipsolver instance.";
#endif
  }
  GpuExecutor* parent_;
  rocblas_handle rocm_blas_handle;
#if TF_ROCM_VERSION >= 40500
  hipsolverHandle_t hipsolver_handle;
#endif
};

using HandleMap =
    std::unordered_map<hipStream_t, std::unique_ptr<GpuSolverHandles>>;

// Returns a singleton map used for storing initialized handles for each unique
// gpu stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

static mutex handle_map_mutex(LINKER_INITIALIZED);

}  // namespace

GpuSolver::GpuSolver(OpKernelContext* context) : context_(context) {
  mutex_lock lock(handle_map_mutex);
  GpuExecutor* gpu_executor = static_cast<GpuExecutor*>(
      context->op_device_context()->stream()->parent()->implementation());
  const hipStream_t* hip_stream_ptr = CHECK_NOTNULL(
      reinterpret_cast<const hipStream_t*>(context->op_device_context()
                                               ->stream()
                                               ->implementation()
                                               ->GpuStreamMemberHack()));

  hip_stream_ = *hip_stream_ptr;
  HandleMap* handle_map = CHECK_NOTNULL(GetHandleMapSingleton());
  auto it = handle_map->find(hip_stream_);
  if (it == handle_map->end()) {
    LOG(INFO) << "Creating GpuSolver handles for stream " << hip_stream_;
    // Previously unseen Gpu stream. Initialize a set of Gpu solver library
    // handles for it.
    std::unique_ptr<GpuSolverHandles> new_handles(
        new GpuSolverHandles(gpu_executor, hip_stream_));
    it = handle_map->insert(std::make_pair(hip_stream_, std::move(new_handles)))
             .first;
  }
  rocm_blas_handle_ = it->second->rocm_blas_handle;
#if TF_ROCM_VERSION >= 40500
  hipsolver_handle_ = it->second->hipsolver_handle;
#endif
}

GpuSolver::~GpuSolver() {
  for (auto tensor_ref : scratch_tensor_refs_) {
    tensor_ref.Unref();
  }
}

// Static
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

  // Launch memcpys to copy info back from device to host
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

#define TF_RETURN_IF_ROCBLAS_ERROR(expr)                                  \
  do {                                                                    \
    auto status = (expr);                                                 \
    if (TF_PREDICT_FALSE(status != rocblas_status_success)) {             \
      return errors::Internal(__FILE__, ":", __LINE__,                    \
                              ": rocBlas call failed status = ", status); \
    }                                                                     \
  } while (0)

// Macro that specializes a solver method for all 4 standard
// numeric types.
#define TF_CALL_ROCSOLV_TYPES(m) \
  m(float, s) m(double, d) m(std::complex<float>, c) m(std::complex<double>, z)
#define TF_CALL_LAPACK_TYPES_NO_COMPLEX(m) m(float, s) m(double, d)
#define TF_CALL_HIP_LAPACK_TYPES_NO_COMPLEX(m) m(float, S) m(double, D)

#define BLAS_SOLVER_FN(method, type_prefix) \
  wrap::rocblas##_##type_prefix##method

#if TF_ROCM_VERSION >= 40500
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)
#define TF_CALL_LAPACK_TYPES_NO_REAL(m) \
  m(std::complex<float>, C) m(std::complex<double>, Z)
#define SOLVER_FN(method, hip_prefix) wrap::hipsolver##hip_prefix##method
#else
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, s) m(double, d) m(std::complex<float>, c) m(std::complex<double>, z)
#define TF_CALL_LAPACK_TYPES_NO_REAL(m) \
  m(std::complex<float>, c) m(std::complex<double>, z)
#define SOLVER_FN(method, type_prefix) wrap::rocsolver##_##type_prefix##method
#endif

// Macros to construct rocsolver/hipsolver method names.
#define ROCSOLVER_FN(method, type_prefix) \
  wrap::rocsolver##_##type_prefix##method
#define BUFSIZE_FN(method, hip_prefix) \
  wrap::hipsolver##hip_prefix##method##_bufferSize

//=============================================================================
// Wrappers of hip/rocSolver computational methods begin here.
//  Please check actual declarations here
//  https://github.com/ROCmSoftwarePlatform/hipSOLVER
//  https://github.com/ROCmSoftwarePlatform/rocSOLVER
//=============================================================================
#if TF_ROCM_VERSION >= 40500

#define GETRF_INSTANCE(Scalar, type_prefix)                                \
  template <>                                                              \
  Status GpuSolver::Getrf<Scalar>(int m, int n, Scalar* A, int lda,        \
                                  int* dev_pivots, int* dev_lapack_info) { \
    mutex_lock lock(handle_map_mutex);                                     \
    int lwork;                                                             \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(getrf, type_prefix)(             \
        hipsolver_handle_, m, n, AsHipComplex(A), lda, &lwork));           \
    auto dev_work =                                                        \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);       \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrf, type_prefix)(              \
        hipsolver_handle_, m, n, AsHipComplex(A), lda,                     \
        AsHipComplex(dev_work.mutable_data()), lwork, dev_pivots,          \
        dev_lapack_info));                                                 \
    return Status::OK();                                                   \
  }

TF_CALL_LAPACK_TYPES(GETRF_INSTANCE);

#define GEQRF_INSTANCE(Scalar, type_prefix)                                  \
  template <>                                                                \
  Status GpuSolver::Geqrf(int m, int n, Scalar* dev_A, int lda,              \
                          Scalar* dev_tau, int* dev_lapack_info) {           \
    mutex_lock lock(handle_map_mutex);                                       \
    int lwork;                                                               \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(geqrf, type_prefix)(               \
        hipsolver_handle_, m, n, AsHipComplex(dev_A), lda, &lwork));         \
    auto dev_work =                                                          \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);         \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(geqrf, type_prefix)(                \
        hipsolver_handle_, m, n, AsHipComplex(dev_A), lda,                   \
        AsHipComplex(dev_tau), AsHipComplex(dev_work.mutable_data()), lwork, \
        dev_lapack_info));                                                   \
    return Status::OK();                                                     \
  }

TF_CALL_LAPACK_TYPES(GEQRF_INSTANCE);

#define UNMQR_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Unmqr(hipsolverSideMode_t side,                            \
                          hipsolverOperation_t trans, int m, int n, int k,     \
                          const Scalar* dev_a, int lda, const Scalar* dev_tau, \
                          Scalar* dev_c, int ldc, int* dev_lapack_info) {      \
    mutex_lock lock(handle_map_mutex);                                         \
    using HipScalar = typename HipComplexT<Scalar>::type;                      \
    ScratchSpace<uint8> dev_a_copy = this->GetScratchSpace<uint8>(             \
        sizeof(Scalar*) * m * k, "", /*on host */ false);                      \
    if (!CopyHostToDevice(context_, dev_a_copy.mutable_data(), dev_a,          \
                          dev_a_copy.bytes())) {                               \
      return errors::Internal("Unmqr: Failed to copy ptrs to device");         \
    }                                                                          \
    ScratchSpace<uint8> dev_tau_copy = this->GetScratchSpace<uint8>(           \
        sizeof(Scalar*) * k * n, "", /*on host */ false);                      \
    if (!CopyHostToDevice(context_, dev_tau_copy.mutable_data(), dev_tau,      \
                          dev_tau_copy.bytes())) {                             \
      return errors::Internal("Unmqr: Failed to copy ptrs to device");         \
    }                                                                          \
    int lwork;                                                                 \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(unmqr, type_prefix)(                 \
        hipsolver_handle_, side, trans, m, n, k,                               \
        reinterpret_cast<HipScalar*>(dev_a_copy.mutable_data()), lda,          \
        reinterpret_cast<HipScalar*>(dev_tau_copy.mutable_data()),             \
        AsHipComplex(dev_c), ldc, &lwork));                                    \
    auto dev_work =                                                            \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);           \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(unmqr, type_prefix)(                  \
        hipsolver_handle_, side, trans, m, n, k,                               \
        reinterpret_cast<HipScalar*>(dev_a_copy.mutable_data()), lda,          \
        reinterpret_cast<HipScalar*>(dev_tau_copy.mutable_data()),             \
        AsHipComplex(dev_c), ldc, AsHipComplex(dev_work.mutable_data()),       \
        lwork, dev_lapack_info));                                              \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES_NO_REAL(UNMQR_INSTANCE);

#define UNGQR_INSTANCE(Scalar, type_prefix)                                  \
  template <>                                                                \
  Status GpuSolver::Ungqr(int m, int n, int k, Scalar* dev_a, int lda,       \
                          const Scalar* dev_tau, int* dev_lapack_info) {     \
    mutex_lock lock(handle_map_mutex);                                       \
    using HipScalar = typename HipComplexT<Scalar>::type;                    \
    ScratchSpace<uint8> dev_tau_copy = this->GetScratchSpace<uint8>(         \
        sizeof(HipScalar*) * k * n, "", /*on host */ false);                 \
    if (!CopyHostToDevice(context_, dev_tau_copy.mutable_data(), dev_tau,    \
                          dev_tau_copy.bytes())) {                           \
      return errors::Internal("Ungqr: Failed to copy ptrs to device");       \
    }                                                                        \
    int lwork;                                                               \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(ungqr, type_prefix)(               \
        hipsolver_handle_, m, n, k, AsHipComplex(dev_a), lda,                \
        reinterpret_cast<HipScalar*>(dev_tau_copy.mutable_data()), &lwork)); \
    auto dev_work =                                                          \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);         \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(ungqr, type_prefix)(                \
        hipsolver_handle_, m, n, k, AsHipComplex(dev_a), lda,                \
        reinterpret_cast<HipScalar*>(dev_tau_copy.mutable_data()),           \
        AsHipComplex(dev_work.mutable_data()), lwork, dev_lapack_info));     \
    return Status::OK();                                                     \
  }

TF_CALL_LAPACK_TYPES_NO_REAL(UNGQR_INSTANCE);

#define POTRF_INSTANCE(Scalar, type_prefix)                              \
  template <>                                                            \
  Status GpuSolver::Potrf<Scalar>(hipsolverFillMode_t uplo, int n,       \
                                  Scalar* dev_A, int lda,                \
                                  int* dev_lapack_info) {                \
    mutex_lock lock(handle_map_mutex);                                   \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;              \
    int lwork;                                                           \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(potrf, type_prefix)(           \
        hipsolver_handle_, uplo, n, AsHipComplex(dev_A), lda, &lwork));  \
    auto dev_work =                                                      \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);     \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(potrf, type_prefix)(            \
        hipsolver_handle_, uplo, n, AsHipComplex(dev_A), lda,            \
        AsHipComplex(dev_work.mutable_data()), lwork, dev_lapack_info)); \
    return Status::OK();                                                 \
  }

TF_CALL_LAPACK_TYPES(POTRF_INSTANCE);

#define GETRS_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Getrs<Scalar>(hipsolverOperation_t trans, int n, int nrhs, \
                                  Scalar* A, int lda, int* dev_pivots,         \
                                  Scalar* B, int ldb, int* dev_lapack_info) {  \
    mutex_lock lock(handle_map_mutex);                                         \
    int lwork;                                                                 \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(getrs, type_prefix)(                 \
        hipsolver_handle_, trans, n, nrhs, AsHipComplex(A), lda, dev_pivots,   \
        AsHipComplex(B), ldb, &lwork));                                        \
    auto dev_work =                                                            \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);           \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrs, type_prefix)(                  \
        hipsolver_handle_, trans, n, nrhs, AsHipComplex(A), lda, dev_pivots,   \
        AsHipComplex(B), ldb, AsHipComplex(dev_work.mutable_data()), lwork,    \
        dev_lapack_info));                                                     \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES(GETRS_INSTANCE);

#define POTRF_BATCHED_INSTANCE(Scalar, type_prefix)                           \
  template <>                                                                 \
  Status GpuSolver::PotrfBatched<Scalar>(                                     \
      hipsolverFillMode_t uplo, int n, const Scalar* const host_a_dev_ptrs[], \
      int lda, DeviceLapackInfo* dev_lapack_info, int batch_size) {           \
    rocblas_stride stride = n;                                                \
    mutex_lock lock(handle_map_mutex);                                        \
    using HipScalar = typename HipComplexT<Scalar>::type;                     \
    ScratchSpace<uint8> dev_a = this->GetScratchSpace<uint8>(                 \
        sizeof(HipScalar*) * batch_size, "", /*on host */ false);             \
    if (!CopyHostToDevice(context_, dev_a.mutable_data(), host_a_dev_ptrs,    \
                          dev_a.bytes())) {                                   \
      return errors::Internal("PotrfBatched: Failed to copy ptrs to device"); \
    }                                                                         \
    int lwork;                                                                \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(potrfBatched, type_prefix)(         \
        hipsolver_handle_, uplo, n,                                           \
        reinterpret_cast<HipScalar**>(dev_a.mutable_data()), lda, &lwork,     \
        batch_size));                                                         \
    auto dev_work =                                                           \
        this->GetScratchSpace<Scalar>(lwork, "", /*on_host*/ false);          \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(potrfBatched, type_prefix)(          \
        hipsolver_handle_, uplo, n,                                           \
        reinterpret_cast<HipScalar**>(dev_a.mutable_data()), lda,             \
        AsHipComplex(dev_work.mutable_data()), lwork,                         \
        dev_lapack_info->mutable_data(), batch_size));                        \
    return Status::OK();                                                      \
  }

TF_CALL_LAPACK_TYPES(POTRF_BATCHED_INSTANCE);

#define HEEVD_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Heevd<Scalar>(                                             \
      hipsolverEigMode_t jobz, hipsolverFillMode_t uplo, int n, Scalar* dev_A, \
      int lda, typename Eigen::NumTraits<Scalar>::Real* dev_W,                 \
      int* dev_lapack_info) {                                                  \
    mutex_lock lock(handle_map_mutex);                                         \
    using EigenScalar = typename Eigen::NumTraits<Scalar>::Real;               \
    int lwork;                                                                 \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(heevd, type_prefix)(                 \
        hipsolver_handle_, jobz, uplo, n, AsHipComplex(dev_A), lda, dev_W,     \
        &lwork));                                                              \
    auto dev_workspace =                                                       \
        this->GetScratchSpace<Scalar>(lwork, "", /*on host */ false);          \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(heevd, type_prefix)(                  \
        hipsolver_handle_, jobz, uplo, n, AsHipComplex(dev_A), lda, dev_W,     \
        AsHipComplex(dev_workspace.mutable_data()), lwork, dev_lapack_info));  \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES_NO_REAL(HEEVD_INSTANCE);

#else
// Macro that specializes a solver method for all 4 standard
// numeric types.
// Macro to construct rocsolver method names.

#define GETRF_INSTANCE(Scalar, type_prefix)                                \
  template <>                                                              \
  Status GpuSolver::Getrf<Scalar>(int m, int n, Scalar* A, int lda,        \
                                  int* dev_pivots, int* dev_lapack_info) { \
    mutex_lock lock(handle_map_mutex);                                     \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrf, type_prefix)(              \
        rocm_blas_handle_, m, n, reinterpret_cast<ROCmScalar*>(A), lda,    \
        dev_pivots, dev_lapack_info));                                     \
    return Status::OK();                                                   \
  }

TF_CALL_LAPACK_TYPES(GETRF_INSTANCE);

#define GEQRF_INSTANCE(Scalar, type_prefix)                                 \
  template <>                                                               \
  Status GpuSolver::Geqrf(int m, int n, Scalar* dev_A, int lda,             \
                          Scalar* dev_tau, int* dev_lapack_info) {          \
    mutex_lock lock(handle_map_mutex);                                      \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                 \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(geqrf, type_prefix)(               \
        rocm_blas_handle_, m, n, reinterpret_cast<ROCmScalar*>(dev_A), lda, \
        reinterpret_cast<ROCmScalar*>(dev_tau)));                           \
    return Status::OK();                                                    \
  }

TF_CALL_LAPACK_TYPES(GEQRF_INSTANCE);

#define UMMQR_INSTANCE(Scalar, type_prefix)                                  \
  template <>                                                                \
  Status GpuSolver::Unmqr(rocblas_side side, rocblas_operation trans, int m, \
                          int n, int k, const Scalar* dev_a, int lda,        \
                          const Scalar* dev_tau, Scalar* dev_c, int ldc,     \
                          int* dev_lapack_info) {                            \
    mutex_lock lock(handle_map_mutex);                                       \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                  \
    ScratchSpace<uint8> dev_a_copy = this->GetScratchSpace<uint8>(           \
        sizeof(ROCmScalar*) * m * k, "", /*on host */ false);                \
    if (!CopyHostToDevice(context_, dev_a_copy.mutable_data(), dev_a,        \
                          dev_a_copy.bytes())) {                             \
      return errors::Internal("Unmqr: Failed to copy ptrs to device");       \
    }                                                                        \
    ScratchSpace<uint8> dev_tau_copy = this->GetScratchSpace<uint8>(         \
        sizeof(ROCmScalar*) * k * n, "", /*on host */ false);                \
    if (!CopyHostToDevice(context_, dev_tau_copy.mutable_data(), dev_tau,    \
                          dev_tau_copy.bytes())) {                           \
      return errors::Internal("Unmqr: Failed to copy ptrs to device");       \
    }                                                                        \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(unmqr, type_prefix)(                \
        rocm_blas_handle_, side, trans, m, n, k,                             \
        reinterpret_cast<ROCmScalar*>(dev_a_copy.mutable_data()), lda,       \
        reinterpret_cast<ROCmScalar*>(dev_tau_copy.mutable_data()),          \
        reinterpret_cast<ROCmScalar*>(dev_c), ldc));                         \
    return Status::OK();                                                     \
  }

TF_CALL_LAPACK_TYPES_NO_REAL(UMMQR_INSTANCE);

#define UNGQR_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Ungqr(int m, int n, int k, Scalar* dev_a, int lda,         \
                          const Scalar* dev_tau, int* dev_lapack_info) {       \
    mutex_lock lock(handle_map_mutex);                                         \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                    \
    ScratchSpace<uint8> dev_tau_copy = this->GetScratchSpace<uint8>(           \
        sizeof(ROCmScalar*) * k * n, "", /*on host */ false);                  \
    if (!CopyHostToDevice(context_, dev_tau_copy.mutable_data(), dev_tau,      \
                          dev_tau_copy.bytes())) {                             \
      return errors::Internal("Ungqr: Failed to copy ptrs to device");         \
    }                                                                          \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(ungqr, type_prefix)(                  \
        rocm_blas_handle_, m, n, k, reinterpret_cast<ROCmScalar*>(dev_a), lda, \
        reinterpret_cast<ROCmScalar*>(dev_tau_copy.mutable_data())));          \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES_NO_REAL(UNGQR_INSTANCE);

#define POTRF_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                  \
  Status GpuSolver::Potrf<Scalar>(rocblas_fill uplo, int n, Scalar* dev_A,     \
                                  int lda, int* dev_lapack_info) {             \
    mutex_lock lock(handle_map_mutex);                                         \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                    \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(potrf, type_prefix)(                  \
        rocm_blas_handle_, uplo, n, reinterpret_cast<ROCmScalar*>(dev_A), lda, \
        dev_lapack_info));                                                     \
    return Status::OK();                                                       \
  }

TF_CALL_LAPACK_TYPES(POTRF_INSTANCE);

#define GETRS_INSTANCE(Scalar, type_prefix)                                   \
  template <>                                                                 \
  Status GpuSolver::Getrs<Scalar>(rocblas_operation trans, int n, int nrhs,   \
                                  Scalar* A, int lda, const int* dev_pivots,  \
                                  Scalar* B, int ldb, int* dev_lapack_info) { \
    mutex_lock lock(handle_map_mutex);                                        \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                   \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrs, type_prefix)(                 \
        rocm_blas_handle_, trans, n, nrhs, reinterpret_cast<ROCmScalar*>(A),  \
        lda, dev_pivots, reinterpret_cast<ROCmScalar*>(B), ldb));             \
    return Status::OK();                                                      \
  }

TF_CALL_LAPACK_TYPES(GETRS_INSTANCE);

#define POTRF_BATCHED_INSTANCE(Scalar, type_prefix)                           \
  template <>                                                                 \
  Status GpuSolver::PotrfBatched<Scalar>(                                     \
      rocblas_fill uplo, int n, const Scalar* const host_a_dev_ptrs[],        \
      int lda, DeviceLapackInfo* dev_lapack_info, int batch_size) {           \
    rocblas_stride stride = n;                                                \
    mutex_lock lock(handle_map_mutex);                                        \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                   \
    ScratchSpace<uint8> dev_a = this->GetScratchSpace<uint8>(                 \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);            \
    if (!CopyHostToDevice(context_, dev_a.mutable_data(), host_a_dev_ptrs,    \
                          dev_a.bytes())) {                                   \
      return errors::Internal("PotrfBatched: Failed to copy ptrs to device"); \
    }                                                                         \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(potrf_batched, type_prefix)(         \
        rocm_blas_handle_, uplo, n,                                           \
        reinterpret_cast<ROCmScalar**>(dev_a.mutable_data()), lda,            \
        dev_lapack_info->mutable_data(), batch_size));                        \
    return Status::OK();                                                      \
  }

TF_CALL_LAPACK_TYPES(POTRF_BATCHED_INSTANCE);

#endif

#define GETRI_BATCHED_INSTANCE(Scalar, type_prefix)                           \
  template <>                                                                 \
  Status GpuSolver::GetriBatched<Scalar>(                                     \
      int n, const Scalar* const host_a_dev_ptrs[], int lda,                  \
      const int* dev_pivots, const Scalar* const host_a_inverse_dev_ptrs[],   \
      int ldainv, DeviceLapackInfo* dev_lapack_info, int batch_size) {        \
    mutex_lock lock(handle_map_mutex);                                        \
    rocblas_stride stride = n;                                                \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                   \
    ScratchSpace<uint8> dev_a = this->GetScratchSpace<uint8>(                 \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);            \
    if (!CopyHostToDevice(context_, dev_a.mutable_data(), host_a_dev_ptrs,    \
                          dev_a.bytes())) {                                   \
      return errors::Internal("GetriBatched: Failed to copy ptrs to device"); \
    }                                                                         \
    ScratchSpace<uint8> dev_a_inverse = this->GetScratchSpace<uint8>(         \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);            \
    if (!CopyHostToDevice(context_, dev_a_inverse.mutable_data(),             \
                          host_a_inverse_dev_ptrs, dev_a_inverse.bytes())) {  \
      return errors::Internal("GetriBatched: Failed to copy ptrs to device"); \
    }                                                                         \
    ScratchSpace<uint8> pivots = this->GetScratchSpace<uint8>(                \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);            \
    if (!CopyHostToDevice(context_, pivots.mutable_data(), dev_pivots,        \
                          pivots.bytes())) {                                  \
      return errors::Internal("GetriBatched: Failed to copy ptrs to device"); \
    }                                                                         \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getri_batched, type_prefix)(         \
        rocm_blas_handle_, n,                                                 \
        reinterpret_cast<ROCmScalar**>(dev_a.mutable_data()), lda,            \
        reinterpret_cast<int*>(pivots.mutable_data()), stride,                \
        dev_lapack_info->mutable_data(), batch_size));                        \
    return Status::OK();                                                      \
  }

TF_CALL_ROCSOLV_TYPES(GETRI_BATCHED_INSTANCE);

#define GETRF_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                  \
  Status GpuSolver::GetrfBatched<Scalar>(                                      \
      int n, Scalar** A, int lda, int* dev_pivots, DeviceLapackInfo* dev_info, \
      const int batch_size) {                                                  \
    mutex_lock lock(handle_map_mutex);                                         \
    rocblas_stride stride = n;                                                 \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                    \
    ScratchSpace<uint8> dev_a = this->GetScratchSpace<uint8>(                  \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);             \
    if (!CopyHostToDevice(context_, dev_a.mutable_data(), A, dev_a.bytes())) { \
      return errors::Internal("GetrfBatched: Failed to copy ptrs to device");  \
    }                                                                          \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrf_batched, type_prefix)(          \
        rocm_blas_handle_, n, n,                                               \
        reinterpret_cast<ROCmScalar**>(dev_a.mutable_data()), lda, dev_pivots, \
        stride, dev_info->mutable_data(), batch_size));                        \
    return Status::OK();                                                       \
  }

TF_CALL_ROCSOLV_TYPES(GETRF_BATCHED_INSTANCE);

#define GETRS_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                  \
  Status GpuSolver::GetrsBatched<Scalar>(                                      \
      const rocblas_operation trans, int n, int nrhs, Scalar** A, int lda,     \
      int* dev_pivots, Scalar** B, const int ldb, int* host_lapack_info,       \
      const int batch_size) {                                                  \
    rocblas_stride stride = n;                                                 \
    mutex_lock lock(handle_map_mutex);                                         \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                    \
    ScratchSpace<uint8> dev_a = this->GetScratchSpace<uint8>(                  \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);             \
    if (!CopyHostToDevice(context_, dev_a.mutable_data(), A, dev_a.bytes())) { \
      return errors::Internal("GetrfBatched: Failed to copy ptrs to device");  \
    }                                                                          \
    ScratchSpace<uint8> dev_b = this->GetScratchSpace<uint8>(                  \
        sizeof(ROCmScalar*) * batch_size, "", /*on host */ false);             \
    if (!CopyHostToDevice(context_, dev_b.mutable_data(), B, dev_b.bytes())) { \
      return errors::Internal("GetrfBatched: Failed to copy ptrs to device");  \
    }                                                                          \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(getrs_batched, type_prefix)(          \
        rocm_blas_handle_, trans, n, nrhs,                                     \
        reinterpret_cast<ROCmScalar**>(dev_a.mutable_data()), lda, dev_pivots, \
        stride, reinterpret_cast<ROCmScalar**>(dev_b.mutable_data()), ldb,     \
        batch_size));                                                          \
    return Status::OK();                                                       \
  }

TF_CALL_ROCSOLV_TYPES(GETRS_BATCHED_INSTANCE);

#define GESVD_INSTANCE(Scalar, type_prefix)                                   \
  template <>                                                                 \
  Status GpuSolver::Gesvd<Scalar>(                                            \
      signed char jobu, signed char jobvt, int m, int n, Scalar* dev_A,       \
      int lda, Scalar* dev_S, Scalar* dev_U, int ldu, Scalar* dev_VT,         \
      int ldvt, int* dev_lapack_info) {                                       \
    mutex_lock lock(handle_map_mutex);                                        \
    /* Get amount of workspace memory required. */                            \
    int lwork;                                                                \
    TF_RETURN_IF_ROCBLAS_ERROR(BUFSIZE_FN(gesvd, type_prefix)(                \
        hipsolver_handle_, jobu, jobvt, m, n, &lwork));                       \
    /* Allocate device memory for workspace. */                               \
    auto dev_workspace =                                                      \
        this->GetScratchSpace<Scalar>(lwork, "", /* on_host */ false);        \
    TF_RETURN_IF_ROCBLAS_ERROR(SOLVER_FN(gesvd, type_prefix)(                 \
        hipsolver_handle_, jobu, jobvt, m, n, ROCmComplex(dev_A), lda, dev_S, \
        ROCmComplex(dev_U), ldu, ROCmComplex(dev_VT), ldvt,                   \
        ROCmComplex(dev_workspace.mutable_data()), lwork, nullptr,            \
        dev_lapack_info));                                                    \
    return Status::OK();                                                      \
  }

TF_CALL_HIP_LAPACK_TYPES_NO_COMPLEX(GESVD_INSTANCE);

template <typename Scalar, typename SolverFnT>
Status MatInvBatchedImpl(GpuExecutor* gpu_executor, SolverFnT solver,
                         rocblas_handle rocm_blas_handle, int n,
                         const Scalar* const host_a_dev_ptrs[], int lda,
                         int* dev_pivots,
                         const Scalar* const host_a_inverse_dev_ptrs[],
                         int ldainv, DeviceLapackInfo* dev_lapack_info,
                         int batch_size) {
  mutex_lock lock(handle_map_mutex);
  using ROCmScalar = typename ROCmComplexT<Scalar>::type;
  ScopedActivateExecutorContext sac{gpu_executor};

  GetrfBatched(n, host_a_dev_ptrs, lda, dev_pivots, dev_lapack_info,
               batch_size);

  GetriBatched(n, host_a_dev_ptrs, lda, dev_pivots, host_a_inverse_dev_ptrs,
               ldainv, dev_lapack_info, batch_size);

  return Status::OK();
}

#define MATINVBATCHED_INSTANCE(Scalar, type_prefix)                           \
  template <>                                                                 \
  Status GpuSolver::MatInvBatched<Scalar>(                                    \
      int n, const Scalar* const host_a_dev_ptrs[], int lda,                  \
      const Scalar* const host_a_inverse_dev_ptrs[], int ldainv,              \
      DeviceLapackInfo* dev_lapack_info, int batch_size) {                    \
    GpuExecutor* gpu_executor = static_cast<GpuExecutor*>(                    \
        context_->op_device_context()->stream()->parent()->implementation()); \
    Tensor pivots;                                                            \
    context_->allocate_scoped_tensor(DataTypeToEnum<int>::value,              \
                                     TensorShape{batch_size, n}, &pivots);    \
    auto pivots_mat = pivots.template matrix<int>();                          \
    int* dev_pivots = pivots_mat.data();                                      \
    return MatInvBatchedImpl(                                                 \
        gpu_executor, BLAS_SOLVER_FN(matinvbatched, type_prefix),             \
        rocm_blas_handle_, n, host_a_dev_ptrs, lda, dev_pivots,               \
        host_a_inverse_dev_ptrs, ldainv, dev_lapack_info, batch_size);        \
  }

//=============================================================================
// Wrappers of rocBlas computational methods begin here.
//  Please check actual declarations here
//  https://github.com/ROCmSoftwarePlatform/rocBlas
//=============================================================================
#define TRSV_INSTANCE(Scalar, type_prefix)                               \
  template <>                                                            \
  Status GpuSolver::Trsv<Scalar>(                                        \
      rocblas_fill uplo, rocblas_operation trans, rocblas_diagonal diag, \
      int n, const Scalar* A, int lda, Scalar* x, int incx) {            \
    mutex_lock lock(handle_map_mutex);                                   \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;              \
    TF_RETURN_IF_ROCBLAS_ERROR(BLAS_SOLVER_FN(trsv, type_prefix)(        \
        rocm_blas_handle_, uplo, trans, diag, n, A, lda, x, incx));      \
    return Status::OK();                                                 \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(TRSV_INSTANCE);

template <typename Scalar, typename SolverFnT>
static inline Status TrsmImpl(GpuExecutor* gpu_executor, SolverFnT solver,
                              rocblas_handle rocm_blas_handle,
                              rocblas_side side, rocblas_fill uplo,
                              rocblas_operation trans, rocblas_diagonal diag,
                              int m, int n,
                              const Scalar* alpha, /* host or device pointer */
                              const Scalar* A, int lda, Scalar* B, int ldb) {
  mutex_lock lock(handle_map_mutex);
  using ROCmScalar = typename ROCmComplexT<Scalar>::type;

  ScopedActivateExecutorContext sac{gpu_executor};
  TF_RETURN_IF_ROCBLAS_ERROR(solver(rocm_blas_handle, side, uplo, trans, diag,
                                    m, n,
                                    reinterpret_cast<const ROCmScalar*>(alpha),
                                    reinterpret_cast<const ROCmScalar*>(A), lda,
                                    reinterpret_cast<ROCmScalar*>(B), ldb));

  return Status::OK();
}

#define TRSM_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                 \
  Status GpuSolver::Trsm<Scalar>(                                             \
      rocblas_side side, rocblas_fill uplo, rocblas_operation trans,          \
      rocblas_diagonal diag, int m, int n,                                    \
      const Scalar* alpha, /* host or device pointer */                       \
      const Scalar* A, int lda, Scalar* B, int ldb) {                         \
    GpuExecutor* gpu_executor = static_cast<GpuExecutor*>(                    \
        context_->op_device_context()->stream()->parent()->implementation()); \
    return TrsmImpl(gpu_executor, BLAS_SOLVER_FN(trsm, type_prefix),          \
                    rocm_blas_handle_, side, uplo, trans, diag, m, n, alpha,  \
                    A, lda, B, ldb);                                          \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(TRSM_INSTANCE);

#define TRSM_BATCHED_INSTANCE(Scalar, type_prefix)                            \
  template <>                                                                 \
  Status GpuSolver::TrsmBatched<Scalar>(                                      \
      rocblas_side side, rocblas_fill uplo, rocblas_operation trans,          \
      rocblas_diagonal diag, int m, int n, const Scalar* alpha,               \
      const Scalar* const dev_Aarray[], int lda, Scalar* dev_Barray[],        \
      int ldb, int batch_size) {                                              \
    mutex_lock lock(handle_map_mutex);                                        \
    using ROCmScalar = typename ROCmComplexT<Scalar>::type;                   \
    ScratchSpace<uint8> dev_a_dev_ptrs = this->GetScratchSpace<uint8>(        \
        sizeof(ROCmScalar*) * batch_size, "", /* on_host */ false);           \
    ScratchSpace<uint8> dev_b_dev_ptrs = this->GetScratchSpace<uint8>(        \
        sizeof(ROCmScalar*) * batch_size, "", /* on_host */ false);           \
    if (!CopyHostToDevice(context_, dev_a_dev_ptrs.mutable_data() /* dest */, \
                          dev_Aarray /* source */, dev_a_dev_ptrs.bytes())) { \
      return errors::Internal(                                                \
          "TrsmBatched: Failed to copy pointers to device");                  \
    }                                                                         \
    if (!CopyHostToDevice(context_, dev_b_dev_ptrs.mutable_data() /* dest */, \
                          dev_Barray /* source */,                          \ 
                            dev_b_dev_ptrs.bytes())) {                        \
      return errors::Internal(                                                \
          "TrsmBatched: Failed to copy pointers to device");                  \
    }                                                                         \
    TF_RETURN_IF_ROCBLAS_ERROR(BLAS_SOLVER_FN(trsm_batched, type_prefix)(     \
        rocm_blas_handle_, side, uplo, trans, diag, m, n, alpha,              \
        reinterpret_cast<ROCmScalar**>(dev_a_dev_ptrs.mutable_data()),      \ 
          lda,                                                                \
        reinterpret_cast<ROCmScalar**>(dev_b_dev_ptrs.mutable_data()), ldb,   \
        batch_size));                                                         \
    return Status::OK();                                                      \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(TRSM_BATCHED_INSTANCE);

template <typename Scalar, typename SolverFnT>
Status GeamImpl(GpuExecutor* gpu_executor, SolverFnT solver,
                rocblas_handle rocm_blas_handle, rocblas_operation transa,
                rocblas_operation transb, int m, int n, const Scalar* alpha,
                /* host or device pointer */ const Scalar* A, int lda,
                const Scalar* beta,
                /* host or device pointer */ const Scalar* B, int ldb,
                Scalar* C, int ldc) {
  mutex_lock lock(handle_map_mutex);
  using ROCmScalar = typename ROCmComplexT<Scalar>::type;

  ScopedActivateExecutorContext sac{gpu_executor};
  TF_RETURN_IF_ROCBLAS_ERROR(solver(rocm_blas_handle, transa, transb, m, n,
                                    reinterpret_cast<const ROCmScalar*>(alpha),
                                    reinterpret_cast<const ROCmScalar*>(A), lda,
                                    reinterpret_cast<const ROCmScalar*>(beta),
                                    reinterpret_cast<const ROCmScalar*>(B), ldb,
                                    reinterpret_cast<ROCmScalar*>(C), ldc));
  return Status::OK();
}

#define GEAM_INSTANCE(Scalar, type_prefix)                                    \
  template <>                                                                 \
  Status GpuSolver::Geam<Scalar>(                                             \
      rocblas_operation transa, rocblas_operation transb, int m, int n,       \
      const Scalar* alpha, const Scalar* A, int lda, const Scalar* beta,      \
      const Scalar* B, int ldb, Scalar* C, int ldc) {                         \
    GpuExecutor* gpu_executor = static_cast<GpuExecutor*>(                    \
        context_->op_device_context()->stream()->parent()->implementation()); \
    return GeamImpl(gpu_executor, BLAS_SOLVER_FN(geam, type_prefix),          \
                    rocm_blas_handle_, transa, transb, m, n, alpha, A, lda,   \
                    beta, B, ldb, C, ldc);                                    \
  }

TF_CALL_LAPACK_TYPES_NO_COMPLEX(GEAM_INSTANCE);
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
