/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_solver_context.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "rocm/include/hiprand/hiprand.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu_solver_context.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace stream_executor {

namespace {

// Type traits to get CUDA complex types from std::complex<T>.
template <typename T>
struct GpuComplexT {
  typedef T type;
};

// For ROCm, use hipsolver if the ROCm version >= 4.5 and
// rocblas/rocsolver if the ROCm version < 4.5.
using gpuDataType_t = hipDataType;

#if TF_ROCM_VERSION >= 40500
#define GPU_SOLVER_CONTEXT_PREFIX wrap::hipsolver
#define GPU_SOLVER_PREFIX wrap::hipsolver

template <>
struct GpuComplexT<std::complex<float>> {
  typedef hipFloatComplex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef hipDoubleComplex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef hipFloatComplex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef hipDoubleComplex* type;
};
#else
#define GPU_SOLVER_CONTEXT_PREFIX wrap::rocblas_
#define GPU_SOLVER_PREFIX wrap::rocsolver_

template <>
struct GpuComplexT<std::complex<float>> {
  typedef rocblas_float_complex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef rocblas_double_complex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef rocblas_float_complex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef rocblas_double_complex* type;
};
#endif  // TF_ROCM_VERSION >= 40500

template <typename T>
inline typename GpuComplexT<T>::type* ToDevicePointer(DeviceMemory<T> p) {
  return static_cast<typename GpuComplexT<T>::type*>(p.opaque());
}

#if TF_ROCM_VERSION >= 40500
hipsolverFillMode_t GpuBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return HIPSOLVER_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return HIPSOLVER_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower";
  }
}

absl::Status ConvertStatus(hipsolverStatus_t status) {
  switch (status) {
    case HIPSOLVER_STATUS_SUCCESS:
      return absl::OkStatus();
    case HIPSOLVER_STATUS_NOT_INITIALIZED:
      return xla::FailedPrecondition("hipsolver has not been initialized");
    case HIPSOLVER_STATUS_ALLOC_FAILED:
      return xla::ResourceExhausted("hipsolver allocation failed");
    case HIPSOLVER_STATUS_INVALID_VALUE:
      return xla::InvalidArgument("hipsolver invalid value error");
    case HIPSOLVER_STATUS_MAPPING_ERROR:
      return xla::Unknown("hipsolver mapping error");
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
      return xla::Unknown("hipsolver execution failed");
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
      return xla::Internal("hipsolver internal error");
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
      return xla::Unimplemented("hipsolver not supported error");
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
      return xla::FailedPrecondition("cuSolver architecture mismatch error");
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
      return xla::InvalidArgument("hipsolver handle is nullptr error");
    case HIPSOLVER_STATUS_INVALID_ENUM:
      return xla::InvalidArgument("hipsolver invalid enum error");
    case HIPSOLVER_STATUS_UNKNOWN:
      return xla::Unknown("hipsolver status unknown");
    default:
      return xla::Unknown("Unknown hipsolver error");
  }
}
#else  // TF_ROCM_VERSION < 40500
rocblas_fill GpuBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

absl::Status ConvertStatus(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return absl::OkStatus();
    case rocblas_status_invalid_handle:
      return xla::FailedPrecondition("handle not initialized, invalid or null");
    case rocblas_status_not_implemented:
      return xla::Internal("function is not implemented");
    case rocblas_status_invalid_pointer:
      return xla::InvalidArgument("invalid pointer argument");
    case rocblas_status_invalid_size:
      return xla::InvalidArgument("invalid size argument");
    case rocblas_status_memory_error:
      return xla::Internal(
          "failed internal memory allocation, copy or dealloc");
    case rocblas_status_internal_error:
      return xla::Internal("other internal library failure");
    case rocblas_status_perf_degraded:
      return xla::Internal("performance degraded due to low device memory");
    case rocblas_status_size_query_mismatch:
      return xla::Unknown("unmatched start/stop size query");
    case rocblas_status_size_increased:
      return xla::Unknown("queried device memory size increased");
    case rocblas_status_size_unchanged:
      return xla::Unknown("queried device memory size unchanged");
    case rocblas_status_invalid_value:
      return xla::InvalidArgument("passed argument not valid");
    case rocblas_status_continue:
      return xla::Unknown("nothing preventing function to proceed");
    default:
      return xla::Unknown("Unknown rocsolver error");
  }
}
#endif  // TF_ROCM_VERSION >= 40500

#define GPU_SOLVER_CAT_NX(A, B) A##B
#define GPU_SOLVER_CAT(A, B) GPU_SOLVER_CAT_NX(A, B)

#if TENSORFLOW_USE_HIPSOLVER
#define GpuSolverCreate GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, Create)
#define GpuSolverSetStream GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, SetStream)
#define GpuSolverDestroy GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, Destroy)
#else  // TENSORFLOW_USE_ROCSOLVER
#define GpuSolverCreate GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, create_handle)
#define GpuSolverSetStream GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, set_stream)
#define GpuSolverDestroy \
  GPU_SOLVER_CAT(GPU_SOLVER_CONTEXT_PREFIX, destroy_handle)
#endif
#define GpuSolverSpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Spotrf_bufferSize)
#define GpuSolverDpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Dpotrf_bufferSize)
#define GpuSolverCpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Cpotrf_bufferSize)
#define GpuSolverZpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Zpotrf_bufferSize)
#define GpuSolverDnXpotrf_bufferSize \
  GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Xpotrf_bufferSize)
#if TENSORFLOW_USE_HIPSOLVER
#define GpuSolverSpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Spotrf)
#define GpuSolverDpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Dpotrf)
#define GpuSolverCpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Cpotrf)
#define GpuSolverZpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Zpotrf)
#define GpuSolverSpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, SpotrfBatched)
#define GpuSolverDpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, DpotrfBatched)
#define GpuSolverCpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, CpotrfBatched)
#define GpuSolverZpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, ZpotrfBatched)
#define GpuSolverXpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, Xpotrf)
#else  // TENSORFLOW_USE_ROCSOLVER
#define GpuSolverSpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, spotrf)
#define GpuSolverDpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, dpotrf)
#define GpuSolverCpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, cpotrf)
#define GpuSolverZpotrf GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, zpotrf)
#define GpuSolverSpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, spotrf_batched)
#define GpuSolverDpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, dpotrf_batched)
#define GpuSolverCpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, cpotrf_batched)
#define GpuSolverZpotrfBatched GPU_SOLVER_CAT(GPU_SOLVER_PREFIX, zpotrf_batched)
#endif

}  // namespace

absl::StatusOr<std::unique_ptr<GpuSolverContext>> RocmSolverContext::Create() {
  gpusolverHandle_t handle;
  TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverCreate(&handle)));
  return absl::WrapUnique(new RocmSolverContext(handle));
}

absl::Status RocmSolverContext::SetStream(Stream* stream) {
  return ConvertStatus(GpuSolverSetStream(
      handle_,
      static_cast<hipStream_t>(stream->platform_specific_handle().stream)));
}

RocmSolverContext::RocmSolverContext(gpusolverHandle_t handle)
    : handle_(handle) {}

RocmSolverContext::~RocmSolverContext() {
  absl::Status status = ConvertStatus(GpuSolverDestroy(handle_));
  if (!status.ok()) {
    LOG(ERROR) << "GpuSolverDestroy failed: " << status;
  }
}

// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
absl::StatusOr<int64_t> RocmSolverContext::PotrfBufferSize(
    xla::PrimitiveType type, blas::UpperLower uplo, int n, int lda,
    int batch_size) {
  int size = -1;
  auto gpu_uplo = GpuBlasUpperLower(uplo);
#if TENSORFLOW_USE_HIPSOLVER
  switch (type) {
    case xla::F32: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverSpotrf_bufferSize(handle_, gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case xla::F64: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverDpotrf_bufferSize(handle_, gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case xla::C64: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverCpotrf_bufferSize(handle_, gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case xla::C128: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverZpotrf_bufferSize(handle_, gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    default:
      return xla::InvalidArgument("Invalid type for cholesky decomposition: %s",
                                  PrimitiveType_Name(type));
  }
#endif  // TENSORFLOW_USE_HIPSOLVER

#if TENSORFLOW_USE_HIPSOLVER
  // CUDA/HIP's potrfBatched needs space for the `as` array, which contains
  // batch_size pointers.  Divide by sizeof(type) because this function returns
  // not bytes but a number of elements of `type`.
  int64_t potrf_batched_scratch = xla::CeilOfRatio<int64_t>(
      batch_size * sizeof(void*), xla::primitive_util::ByteWidth(type));

  return std::max<int64_t>(size, potrf_batched_scratch);
#else  // not supported in rocsolver
  return 0;
#endif
}

absl::Status RocmSolverContext::PotrfBatched(blas::UpperLower uplo, int n,
                                             DeviceMemory<float*> as, int lda,
                                             DeviceMemory<int> lapack_info,
                                             int batch_size) {
  return ConvertStatus(GpuSolverSpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status RocmSolverContext::PotrfBatched(blas::UpperLower uplo, int n,
                                             DeviceMemory<double*> as, int lda,
                                             DeviceMemory<int> lapack_info,
                                             int batch_size) {
  return ConvertStatus(GpuSolverDpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status RocmSolverContext::PotrfBatched(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<float>*> as,
    int lda, DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(GpuSolverCpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status RocmSolverContext::PotrfBatched(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<double>*> as,
    int lda, DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(GpuSolverZpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

#if TENSORFLOW_USE_HIPSOLVER
absl::Status RocmSolverContext::Potrf(blas::UpperLower uplo, int n,
                                      DeviceMemory<double> a, int lda,
                                      DeviceMemory<int> lapack_info,
                                      DeviceMemory<double> workspace) {
  return ConvertStatus(GpuSolverDpotrf(handle_, GpuBlasUpperLower(uplo), n,
                                       ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status RocmSolverContext::Potrf(blas::UpperLower uplo, int n,
                                      DeviceMemory<float> a, int lda,
                                      DeviceMemory<int> lapack_info,
                                      DeviceMemory<float> workspace) {
  return ConvertStatus(GpuSolverSpotrf(handle_, GpuBlasUpperLower(uplo), n,
                                       ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status RocmSolverContext::Potrf(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<float>> a, int lda,
    DeviceMemory<int> lapack_info,
    DeviceMemory<std::complex<float>> workspace) {
  return ConvertStatus(GpuSolverCpotrf(handle_, GpuBlasUpperLower(uplo), n,
                                       ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status RocmSolverContext::Potrf(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<double>> a, int lda,
    DeviceMemory<int> lapack_info,
    DeviceMemory<std::complex<double>> workspace) {
  return ConvertStatus(GpuSolverZpotrf(handle_, GpuBlasUpperLower(uplo), n,
                                       ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}
#endif  // TENSORFLOW_USE_HIPSOLVER

}  // namespace stream_executor
