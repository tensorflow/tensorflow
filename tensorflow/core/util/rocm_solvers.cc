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
#if TENSORFLOW_USE_ROCM
#include "tensorflow/core/util/rocm_solvers.h"

#include <complex>
#include <unordered_map>
#include <vector>

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
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/default/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/rocm/rocblas_wrapper.h"

namespace tensorflow {
namespace {

using stream_executor::gpu::GpuExecutor;
using stream_executor::gpu::ScopedActivateExecutorContext;

struct ROCmSolverHandles {
  explicit ROCmSolverHandles(GpuExecutor* parent, hipStream_t stream) {
    parent_ = parent;
    ScopedActivateExecutorContext sac{parent_};
    CHECK(wrap::rocblas_create_handle(&rocm_blas_handle) ==
          rocblas_status_success)
        << "Failed to create rocBlas instance.";
    CHECK(wrap::rocblas_set_stream(rocm_blas_handle, stream) ==
          rocblas_status_success)
        << "Failed to set rocBlas stream.";
  }

  ~ROCmSolverHandles() {
    ScopedActivateExecutorContext sac{parent_};
    CHECK(wrap::rocblas_destroy_handle(rocm_blas_handle) ==
          rocblas_status_success)
        << "Failed to destroy cuBlas instance.";
  }
  GpuExecutor* parent_;
  rocblas_handle rocm_blas_handle;
};

using HandleMap =
    std::unordered_map<hipStream_t, std::unique_ptr<ROCmSolverHandles>>;

// Returns a singleton map used for storing initialized handles for each unique
// gpu stream.
HandleMap* GetHandleMapSingleton() {
  static HandleMap* cm = new HandleMap;
  return cm;
}

static mutex handle_map_mutex(LINKER_INITIALIZED);

}  // namespace

ROCmSolver::ROCmSolver(OpKernelContext* context) : context_(context) {
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
    LOG(INFO) << "Creating ROCmSolver handles for stream " << hip_stream_;
    // Previously unseen Gpu stream. Initialize a set of Gpu solver library
    // handles for it.
    std::unique_ptr<ROCmSolverHandles> new_handles(
        new ROCmSolverHandles(gpu_executor, hip_stream_));
    it = handle_map->insert(std::make_pair(hip_stream_, std::move(new_handles)))
             .first;
  }
  rocm_blas_handle_ = it->second->rocm_blas_handle;
}

ROCmSolver::~ROCmSolver() {
  for (auto tensor_ref : scratch_tensor_refs_) {
    tensor_ref.Unref();
  }
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
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, s) m(double, d) m(std::complex<float>, c) m(std::complex<double>, z)
#define TF_CALL_LAPACK_TYPES_NO_COMPLEX(m) m(float, s) m(double, d)

#define BLAS_SOLVER_FN(method, type_prefix) \
  wrap::rocblas##_##type_prefix##method

// Allocates a temporary tensor. The ROCmSolver object maintains a
// TensorReference to the underlying Tensor to prevent it from being deallocated
// prematurely.
Status ROCmSolver::allocate_scoped_tensor(DataType type,
                                          const TensorShape& shape,
                                          Tensor* out_temp) {
  const Status status = context_->allocate_temp(type, shape, out_temp);
  if (status.ok()) {
    scratch_tensor_refs_.emplace_back(*out_temp);
  }
  return status;
}

Status ROCmSolver::forward_input_or_allocate_scoped_tensor(
    gtl::ArraySlice<int> candidate_input_indices, DataType type,
    const TensorShape& shape, Tensor* out_temp) {
  const Status status = context_->forward_input_or_allocate_temp(
      candidate_input_indices, type, shape, out_temp);
  if (status.ok()) {
    scratch_tensor_refs_.emplace_back(*out_temp);
  }
  return status;
}

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
  Status ROCmSolver::Trsm<Scalar>(                                            \
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

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
