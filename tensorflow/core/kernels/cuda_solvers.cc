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

#include "cuda/include/cublas_v2.h"
#include "cuda/include/cusolverDn.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace tensorflow {
namespace {

template <typename Scalar>
class ScratchSpace {
 public:
  explicit ScratchSpace(OpKernelContext* context, int size) {
    TF_CHECK_OK(context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                       TensorShape({size}), &scratch_tensor_));
  }
  Scalar* data() { return scratch_tensor_.template flat<Scalar>().data(); }

 private:
  Tensor scratch_tensor_;
};

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

// Converts values of std::complex<float/double> to values of
// cuComplex/cuDoubleComplex.
inline cuComplex CUDAComplexValue(std::complex<float> val) {
  return {val.real(), val.imag()};
}

inline cuDoubleComplex CUDAComplexValue(std::complex<double> val) {
  return {val.real(), val.imag()};
}
}  // namespace

#define TF_RETURN_IF_CUSOLVER_ERROR_MSG(expr, msg)             \
  do {                                                         \
    auto status = (expr);                                      \
    if (TF_PREDICT_FALSE(status != CUSOLVER_STATUS_SUCCESS)) { \
      return errors::Internal(msg);                            \
    }                                                          \
  } while (0)

#define TF_RETURN_IF_CUSOLVER_ERROR(expr) \
  TF_RETURN_IF_CUSOLVER_ERROR_MSG(expr, "cuSolverDN call failed.")

#define TF_RETURN_STATUS_FROM_INFO(method, device_info_ptr, info_ptr)     \
  do {                                                                    \
    int local_info;                                                       \
    TF_RETURN_IF_ERROR(GetInfo(device_info_ptr, &local_info));            \
    if (info_ptr != nullptr) *info_ptr = local_info;                      \
    if (TF_PREDICT_FALSE(local_info != 0)) {                              \
      return errors::Internal("cuSolverDN::" #method " returned info = ", \
                              local_info, ", expected info = 0");         \
    } else {                                                              \
      return Status::OK();                                                \
    }                                                                     \
  } while (0)

CudaSolverDN::CudaSolverDN(OpKernelContext* context) : context_(context) {
  const cudaStream_t* cu_stream_ptr = CHECK_NOTNULL(
      reinterpret_cast<const cudaStream_t*>(context->op_device_context()
                                                ->stream()
                                                ->implementation()
                                                ->CudaStreamMemberHack()));
  cuda_stream_ = *cu_stream_ptr;
  CHECK(cusolverDnCreate(&handle_) == CUSOLVER_STATUS_SUCCESS)
      << "Failed to create cuSolverDN instance.";
  CHECK(cusolverDnSetStream(handle_, cuda_stream_) == CUSOLVER_STATUS_SUCCESS)
      << "Failed to set cuSolverDN stream.";
}

CudaSolverDN::~CudaSolverDN() {
  CHECK(cusolverDnDestroy(handle_) == CUSOLVER_STATUS_SUCCESS)
      << "Failed to destroy cuSolverDN instance.";
}

Status CudaSolverDN::GetInfo(const int* dev_info, int* host_info) const {
  CHECK(dev_info != nullptr);
  CHECK(host_info != nullptr);
  auto stream = context_->op_device_context()->stream();
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<int*>(dev_info));
  if (!stream
           ->ThenMemcpy(host_info /* destination */, wrapped /* source */,
                        sizeof(int))
           .ok()) {
    return errors::Internal("Failed to copy dev_info to host.");
  }
  BlockingCounter barrier(1);
  context_->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      stream, [&barrier]() { barrier.DecrementCount(); });
  if (!barrier.WaitFor(std::chrono::minutes(1))) {
    return errors::Internal("Failed to copy dev_info to host within 1 minute.");
  }
  return Status::OK();
}

// Macro that specializes a solver method for all 4 standard
// numeric types.
#define TF_CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

// Macros to construct cusolver method names.
#define SOLVER_NAME(method, lapack_prefix) cusolverDn##lapack_prefix##method
#define BUFSIZE_NAME(method, lapack_prefix) \
  cusolverDn##lapack_prefix##method##_bufferSize

//=============================================================================
// Wrappers of cuSolverDN computational methods begin here.
//=============================================================================
#define POTRF_INSTANCE(Scalar, lapack_prefix)                                 \
  template <>                                                                 \
  Status CudaSolverDN::potrf<Scalar>(cublasFillMode_t uplo, int n, Scalar* A, \
                                     int lda, int* info) const {              \
    /* Get amount of workspace memory required. */                            \
    int lwork;                                                                \
    TF_RETURN_IF_CUSOLVER_ERROR(BUFSIZE_NAME(potrf, lapack_prefix)(           \
        handle_, uplo, n, CUDAComplex(A), lda, &lwork));                      \
                                                                              \
    /* Allocate device memory for workspace and info. */                      \
    ScratchSpace<Scalar> device_workspace(context_, lwork);                   \
    ScratchSpace<int> device_info(context_, 1);                               \
                                                                              \
    /* Launch the solver kernel. */                                           \
    TF_RETURN_IF_CUSOLVER_ERROR(SOLVER_NAME(potrf, lapack_prefix)(            \
        handle_, uplo, n, CUDAComplex(A), lda,                                \
        CUDAComplex(device_workspace.data()), lwork, device_info.data()));    \
                                                                              \
    /* Get info from device and return status. */                             \
    TF_RETURN_STATUS_FROM_INFO(potrf, device_info.data(), info);              \
    return Status::OK();                                                      \
  }

TF_CALL_LAPACK_TYPES(POTRF_INSTANCE);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
