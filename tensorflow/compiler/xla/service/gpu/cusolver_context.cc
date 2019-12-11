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

#include "tensorflow/compiler/xla/service/gpu/cusolver_context.h"

#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

namespace {

// Type traits to get CUDA complex types from std::complex<T>.
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

template <typename T>
inline typename CUDAComplexT<T>::type* ToDevicePointer(se::DeviceMemory<T> p) {
  return static_cast<typename CUDAComplexT<T>::type*>(p.opaque());
}

cublasFillMode_t CUDABlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case se::blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

// Converts a cuSolver status to a Status.
Status CusolverStatusToStatus(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return Status::OK();
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return FailedPrecondition("cuSolver has not been initialized");
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return ResourceExhausted("cuSolver allocation failed");
    case CUSOLVER_STATUS_INVALID_VALUE:
      return InvalidArgument("cuSolver invalid value error");
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return FailedPrecondition("cuSolver architecture mismatch error");
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return Unknown("cuSolver mapping error");
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return Unknown("cuSolver execution failed");
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return Internal("cuSolver internal error");
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return Unimplemented("cuSolver matrix type not supported error");
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return Unimplemented("cuSolver not supported error");
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return InvalidArgument("cuSolver zero pivot error");
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return FailedPrecondition("cuSolver invalid license error");
    default:
      return Unknown("Unknown cuSolver error");
  }
}

}  // namespace

StatusOr<CusolverContext> CusolverContext::Create(se::Stream* stream) {
  cusolverDnHandle_t handle;
  TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnCreate(&handle)));
  CusolverContext context(stream, handle);

  if (stream) {
    // StreamExecutor really should just expose the Cuda stream to clients...
    const cudaStream_t* cuda_stream =
        CHECK_NOTNULL(reinterpret_cast<const cudaStream_t*>(
            stream->implementation()->GpuStreamMemberHack()));
    TF_RETURN_IF_ERROR(
        CusolverStatusToStatus(cusolverDnSetStream(handle, *cuda_stream)));
  }

  return std::move(context);
}

CusolverContext::CusolverContext(se::Stream* stream, cusolverDnHandle_t handle)
    : stream_(stream), handle_(handle) {}

CusolverContext::CusolverContext(CusolverContext&& other) {
  handle_ = other.handle_;
  stream_ = other.stream_;
  other.handle_ = nullptr;
  other.stream_ = nullptr;
}

CusolverContext& CusolverContext::operator=(CusolverContext&& other) {
  std::swap(handle_, other.handle_);
  std::swap(stream_, other.stream_);
  return *this;
}

CusolverContext::~CusolverContext() {
  if (handle_) {
    Status status = CusolverStatusToStatus(cusolverDnDestroy(handle_));
    if (!status.ok()) {
      LOG(ERROR) << "cusolverDnDestroy failed: " << status;
    }
  }
}

#define CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

#define DN_SOLVER_FN(method, type_prefix) cusolverDn##type_prefix##method

// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
StatusOr<int64> CusolverContext::PotrfBufferSize(PrimitiveType type,
                                                 se::blas::UpperLower uplo,
                                                 int n, int lda) {
  int size = -1;
  switch (type) {
    case F32: {
      TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnSpotrf_bufferSize(
          handle(), CUDABlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case F64: {
      TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnDpotrf_bufferSize(
          handle(), CUDABlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case C64: {
      TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnCpotrf_bufferSize(
          handle(), CUDABlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    case C128: {
      TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnZpotrf_bufferSize(
          handle(), CUDABlasUpperLower(uplo), n, /*A=*/nullptr, lda, &size)));
      break;
    }
    default:
      return InvalidArgument("Invalid type for cholesky decomposition: %s",
                             PrimitiveType_Name(type));
  }
  return size;
}

#define POTRF_INSTANCE(T, type_prefix)                                    \
  template <>                                                             \
  Status CusolverContext::Potrf<T>(                                       \
      se::blas::UpperLower uplo, int n, se::DeviceMemory<T> A, int lda,   \
      se::DeviceMemory<int> lapack_info, se::DeviceMemory<T> workspace) { \
    return CusolverStatusToStatus(DN_SOLVER_FN(potrf, type_prefix)(       \
        handle(), CUDABlasUpperLower(uplo), n, ToDevicePointer(A), lda,   \
        ToDevicePointer(workspace), workspace.ElementCount(),             \
        ToDevicePointer(lapack_info)));                                   \
  }

CALL_LAPACK_TYPES(POTRF_INSTANCE);

}  // namespace gpu
}  // namespace xla
