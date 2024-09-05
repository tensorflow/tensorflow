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

#include "xla/service/gpu/cusolver_context.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusolver_common.h"
#include "third_party/gpus/cuda/include/library_types.h"
#endif
#include "xla/primitive_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

// Type traits to get CUDA complex types from std::complex<T>.
template <typename T>
struct GpuComplexT {
  typedef T type;
};

// For ROCm, use hipsolver if the ROCm version >= 4.5 and
// rocblas/rocsolver if the ROCm version < 4.5.

#if GOOGLE_CUDA

#define GPU_SOLVER_CONTEXT_PREFIX cusolverDn
#define GPU_SOLVER_PREFIX cusolverDn

using gpuDataType_t = cudaDataType_t;

template <>
struct GpuComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct GpuComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};

template <>
struct GpuComplexT<std::complex<float>*> {
  typedef cuComplex* type;
};
template <>
struct GpuComplexT<std::complex<double>*> {
  typedef cuDoubleComplex* type;
};

#elif TENSORFLOW_USE_ROCM

using gpuDataType_t = hipDataType;

#if TF_ROCM_VERSION >= 40500
#define GPU_SOLVER_CONTEXT_PREFIX se::wrap::hipsolver
#define GPU_SOLVER_PREFIX se::wrap::hipsolver

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
#define GPU_SOLVER_CONTEXT_PREFIX se::wrap::rocblas_
#define GPU_SOLVER_PREFIX se::wrap::rocsolver_

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

#endif  // TENSORFLOW_USE_ROCM

template <typename T>
inline typename GpuComplexT<T>::type* ToDevicePointer(se::DeviceMemory<T> p) {
  return static_cast<typename GpuComplexT<T>::type*>(p.opaque());
}

#if GOOGLE_CUDA
cublasFillMode_t GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case se::blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

// Converts a cuSolver absl::Status to a absl::Status.
absl::Status ConvertStatus(cusolverStatus_t status) {
  switch (status) {
    case CUSOLVER_STATUS_SUCCESS:
      return absl::OkStatus();
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
#elif TENSORFLOW_USE_ROCM

#if TF_ROCM_VERSION >= 40500
hipsolverFillMode_t GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return HIPSOLVER_FILL_MODE_UPPER;
    case se::blas::UpperLower::kLower:
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
      return FailedPrecondition("hipsolver has not been initialized");
    case HIPSOLVER_STATUS_ALLOC_FAILED:
      return ResourceExhausted("hipsolver allocation failed");
    case HIPSOLVER_STATUS_INVALID_VALUE:
      return InvalidArgument("hipsolver invalid value error");
    case HIPSOLVER_STATUS_MAPPING_ERROR:
      return Unknown("hipsolver mapping error");
    case HIPSOLVER_STATUS_EXECUTION_FAILED:
      return Unknown("hipsolver execution failed");
    case HIPSOLVER_STATUS_INTERNAL_ERROR:
      return Internal("hipsolver internal error");
    case HIPSOLVER_STATUS_NOT_SUPPORTED:
      return Unimplemented("hipsolver not supported error");
    case HIPSOLVER_STATUS_ARCH_MISMATCH:
      return FailedPrecondition("cuSolver architecture mismatch error");
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR:
      return InvalidArgument("hipsolver handle is nullptr error");
    case HIPSOLVER_STATUS_INVALID_ENUM:
      return InvalidArgument("hipsolver invalid enum error");
    case HIPSOLVER_STATUS_UNKNOWN:
      return Unknown("hipsolver status unknown");
    default:
      return Unknown("Unknown hipsolver error");
  }
}
#else  // TF_ROCM_VERSION < 40500
rocblas_fill GpuBlasUpperLower(se::blas::UpperLower uplo) {
  switch (uplo) {
    case se::blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case se::blas::UpperLower::kLower:
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
      return FailedPrecondition("handle not initialized, invalid or null");
    case rocblas_status_not_implemented:
      return Internal("function is not implemented");
    case rocblas_status_invalid_pointer:
      return InvalidArgument("invalid pointer argument");
    case rocblas_status_invalid_size:
      return InvalidArgument("invalid size argument");
    case rocblas_status_memory_error:
      return Internal("failed internal memory allocation, copy or dealloc");
    case rocblas_status_internal_error:
      return Internal("other internal library failure");
    case rocblas_status_perf_degraded:
      return Internal("performance degraded due to low device memory");
    case rocblas_status_size_query_mismatch:
      return Unknown("unmatched start/stop size query");
    case rocblas_status_size_increased:
      return Unknown("queried device memory size increased");
    case rocblas_status_size_unchanged:
      return Unknown("queried device memory size unchanged");
    case rocblas_status_invalid_value:
      return InvalidArgument("passed argument not valid");
    case rocblas_status_continue:
      return Unknown("nothing preventing function to proceed");
    default:
      return Unknown("Unknown rocsolver error");
  }
}
#endif  // TF_ROCM_VERSION >= 40500
#endif  // TENSORFLOW_USE_ROCM

#define GPU_SOLVER_CAT_NX(A, B) A##B
#define GPU_SOLVER_CAT(A, B) GPU_SOLVER_CAT_NX(A, B)

#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
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
#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
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

absl::StatusOr<GpuSolverContext> GpuSolverContext::Create() {
  gpusolverHandle_t handle;
  TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverCreate(&handle)));
  return GpuSolverContext(handle);
}

absl::Status GpuSolverContext::SetStream(se::Stream* stream) {
  return ConvertStatus(
      GpuSolverSetStream(handle_.get(), se::gpu::AsGpuStreamValue(stream)));
}

GpuSolverContext::GpuSolverContext(gpusolverHandle_t handle)
    : handle_(handle) {}

void GpuSolverContext::Deleter::operator()(gpusolverHandle_t handle) {
  if (handle) {
    absl::Status status = ConvertStatus(GpuSolverDestroy(handle));
    if (!status.ok()) {
      LOG(ERROR) << "GpuSolverDestroy failed: " << status;
    }
  }
}

// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
absl::StatusOr<int64_t> GpuSolverContext::PotrfBufferSize(
    PrimitiveType type, se::blas::UpperLower uplo, int n, int lda,
    int batch_size) {
  int size = -1;
  auto gpu_uplo = GpuBlasUpperLower(uplo);
#if GOOGLE_CUDA
  size_t d_lwork = 0; /* size of workspace */
  size_t h_lwork = 0; /* size of workspace */

  gpuDataType_t cuda_data_type;
  switch (type) {
    case F32: {
      cuda_data_type = CUDA_R_32F;
      break;
    }
    case F64: {
      cuda_data_type = CUDA_R_64F;
      break;
    }
    case C64: {
      cuda_data_type = CUDA_C_32F;
      break;
    }
    case C128: {
      cuda_data_type = CUDA_C_64F;
      break;
    }
    default:
      return InvalidArgument("Invalid type for cholesky decomposition: %s",
                             PrimitiveType_Name(type));
  }
  TF_RETURN_IF_ERROR(ConvertStatus(GpuSolverDnXpotrf_bufferSize(
      handle_.get(), nullptr, gpu_uplo, n, cuda_data_type, nullptr, lda,
      cuda_data_type, &d_lwork, &h_lwork)));
  size = static_cast<int>(d_lwork);

#elif TENSORFLOW_USE_HIPSOLVER
  switch (type) {
    case F32: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverSpotrf_bufferSize(handle_.get(), gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case F64: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverDpotrf_bufferSize(handle_.get(), gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case C64: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverCpotrf_bufferSize(handle_.get(), gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    case C128: {
      TF_RETURN_IF_ERROR(
          ConvertStatus(GpuSolverZpotrf_bufferSize(handle_.get(), gpu_uplo, n,
                                                   /*A=*/nullptr, lda, &size)));
      break;
    }
    default:
      return InvalidArgument("Invalid type for cholesky decomposition: %s",
                             PrimitiveType_Name(type));
  }
#endif  // TENSORFLOW_USE_HIPSOLVER

#if TENSORFLOW_USE_CUSOLVER_OR_HIPSOLVER
  // CUDA/HIP's potrfBatched needs space for the `as` array, which contains
  // batch_size pointers.  Divide by sizeof(type) because this function returns
  // not bytes but a number of elements of `type`.
  int64_t potrf_batched_scratch = CeilOfRatio<int64_t>(
      batch_size * sizeof(void*), primitive_util::ByteWidth(type));

  return std::max<int64_t>(size, potrf_batched_scratch);
#else  // not supported in rocsolver
  return 0;
#endif
}

absl::Status GpuSolverContext::PotrfBatched(se::blas::UpperLower uplo, int n,
                                            se::DeviceMemory<float*> as,
                                            int lda,
                                            se::DeviceMemory<int> lapack_info,
                                            int batch_size) {
  return ConvertStatus(GpuSolverSpotrfBatched(
      handle_.get(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status GpuSolverContext::PotrfBatched(se::blas::UpperLower uplo, int n,
                                            se::DeviceMemory<double*> as,
                                            int lda,
                                            se::DeviceMemory<int> lapack_info,
                                            int batch_size) {
  return ConvertStatus(GpuSolverDpotrfBatched(
      handle_.get(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status GpuSolverContext::PotrfBatched(
    se::blas::UpperLower uplo, int n, se::DeviceMemory<std::complex<float>*> as,
    int lda, se::DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(GpuSolverCpotrfBatched(
      handle_.get(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status GpuSolverContext::PotrfBatched(
    se::blas::UpperLower uplo, int n,
    se::DeviceMemory<std::complex<double>*> as, int lda,
    se::DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(GpuSolverZpotrfBatched(
      handle_.get(), GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
#if TENSORFLOW_USE_HIPSOLVER
      nullptr, 0,
#endif
      ToDevicePointer(lapack_info), batch_size));
}

#if GOOGLE_CUDA
absl::Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                                     se::DeviceMemory<double> a, int lda,
                                     se::DeviceMemory<int> lapack_info,
                                     se::DeviceMemory<double> workspace) {
  absl::Status status = ConvertStatus(GpuSolverXpotrf(
      handle_.get(), nullptr, GpuBlasUpperLower(uplo), n, CUDA_R_64F,
      ToDevicePointer(a), lda, CUDA_R_64F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                                     se::DeviceMemory<float> a, int lda,
                                     se::DeviceMemory<int> lapack_info,
                                     se::DeviceMemory<float> workspace) {
  absl::Status status = ConvertStatus(GpuSolverXpotrf(
      handle_.get(), nullptr, GpuBlasUpperLower(uplo), n, CUDA_R_32F,
      ToDevicePointer(a), lda, CUDA_R_32F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status GpuSolverContext::Potrf(
    se::blas::UpperLower uplo, int n, se::DeviceMemory<std::complex<float>> a,
    int lda, se::DeviceMemory<int> lapack_info,
    se::DeviceMemory<std::complex<float>> workspace) {
  absl::Status status = ConvertStatus(GpuSolverXpotrf(
      handle_.get(), nullptr, GpuBlasUpperLower(uplo), n, CUDA_C_32F,
      ToDevicePointer(a), lda, CUDA_C_32F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status GpuSolverContext::Potrf(
    se::blas::UpperLower uplo, int n, se::DeviceMemory<std::complex<double>> a,
    int lda, se::DeviceMemory<int> lapack_info,
    se::DeviceMemory<std::complex<double>> workspace) {
  absl::Status status = ConvertStatus(GpuSolverXpotrf(
      handle_.get(), nullptr, GpuBlasUpperLower(uplo), n, CUDA_C_64F,
      ToDevicePointer(a), lda, CUDA_C_64F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}
#elif TENSORFLOW_USE_HIPSOLVER
absl::Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                                     se::DeviceMemory<double> a, int lda,
                                     se::DeviceMemory<int> lapack_info,
                                     se::DeviceMemory<double> workspace) {
  return ConvertStatus(GpuSolverDpotrf(handle_.get(), GpuBlasUpperLower(uplo),
                                       n, ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status GpuSolverContext::Potrf(se::blas::UpperLower uplo, int n,
                                     se::DeviceMemory<float> a, int lda,
                                     se::DeviceMemory<int> lapack_info,
                                     se::DeviceMemory<float> workspace) {
  return ConvertStatus(GpuSolverSpotrf(handle_.get(), GpuBlasUpperLower(uplo),
                                       n, ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status GpuSolverContext::Potrf(
    se::blas::UpperLower uplo, int n, se::DeviceMemory<std::complex<float>> a,
    int lda, se::DeviceMemory<int> lapack_info,
    se::DeviceMemory<std::complex<float>> workspace) {
  return ConvertStatus(GpuSolverCpotrf(handle_.get(), GpuBlasUpperLower(uplo),
                                       n, ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}

absl::Status GpuSolverContext::Potrf(
    se::blas::UpperLower uplo, int n, se::DeviceMemory<std::complex<double>> a,
    int lda, se::DeviceMemory<int> lapack_info,
    se::DeviceMemory<std::complex<double>> workspace) {
  return ConvertStatus(GpuSolverZpotrf(handle_.get(), GpuBlasUpperLower(uplo),
                                       n, ToDevicePointer(a), lda, nullptr, 0,
                                       ToDevicePointer(lapack_info)));
}
#endif  // TENSORFLOW_USE_HIPSOLVER

}  // namespace gpu
}  // namespace xla
