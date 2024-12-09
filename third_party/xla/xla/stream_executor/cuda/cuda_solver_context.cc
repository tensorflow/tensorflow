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

#include "xla/stream_executor/cuda/cuda_solver_context.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cusolverDn.h"
#include "third_party/gpus/cuda/include/cusolver_common.h"
#include "third_party/gpus/cuda/include/library_types.h"
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

template <typename T>
inline typename GpuComplexT<T>::type* ToDevicePointer(DeviceMemory<T> p) {
  return static_cast<typename GpuComplexT<T>::type*>(p.opaque());
}

cublasFillMode_t GpuBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
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
      return xla::FailedPrecondition("cuSolver has not been initialized");
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return xla::ResourceExhausted("cuSolver allocation failed");
    case CUSOLVER_STATUS_INVALID_VALUE:
      return xla::InvalidArgument("cuSolver invalid value error");
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return xla::FailedPrecondition("cuSolver architecture mismatch error");
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return xla::Unknown("cuSolver mapping error");
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return xla::Unknown("cuSolver execution failed");
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return xla::Internal("cuSolver internal error");
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return xla::Unimplemented("cuSolver matrix type not supported error");
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return xla::Unimplemented("cuSolver not supported error");
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return xla::InvalidArgument("cuSolver zero pivot error");
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return xla::FailedPrecondition("cuSolver invalid license error");
    default:
      return xla::Unknown("Unknown cuSolver error");
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<GpuSolverContext>> CudaSolverContext::Create() {
  cusolverDnHandle_t handle;
  TF_RETURN_IF_ERROR(ConvertStatus(cusolverDnCreate(&handle)));
  return absl::WrapUnique(new CudaSolverContext(handle));
}

absl::Status CudaSolverContext::SetStream(Stream* stream) {
  return ConvertStatus(cusolverDnSetStream(
      handle_,
      static_cast<cudaStream_t>(stream->platform_specific_handle().stream)));
}

CudaSolverContext::CudaSolverContext(cusolverDnHandle_t handle)
    : handle_(handle) {}

CudaSolverContext::~CudaSolverContext() {
  absl::Status status = ConvertStatus(cusolverDnDestroy(handle_));
  if (!status.ok()) {
    LOG(ERROR) << "GpuSolverDestroy failed: " << status;
  }
}

// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
absl::StatusOr<int64_t> CudaSolverContext::PotrfBufferSize(
    xla::PrimitiveType type, blas::UpperLower uplo, int n, int lda,
    int batch_size) {
  int size = -1;
  auto gpu_uplo = GpuBlasUpperLower(uplo);
  size_t d_lwork = 0; /* size of workspace */
  size_t h_lwork = 0; /* size of workspace */

  cudaDataType_t cuda_data_type;
  switch (type) {
    case xla::F32: {
      cuda_data_type = CUDA_R_32F;
      break;
    }
    case xla::F64: {
      cuda_data_type = CUDA_R_64F;
      break;
    }
    case xla::C64: {
      cuda_data_type = CUDA_C_32F;
      break;
    }
    case xla::C128: {
      cuda_data_type = CUDA_C_64F;
      break;
    }
    default:
      return xla::InvalidArgument("Invalid type for cholesky decomposition: %s",
                                  PrimitiveType_Name(type));
  }
  TF_RETURN_IF_ERROR(ConvertStatus(cusolverDnXpotrf_bufferSize(
      handle_, nullptr, gpu_uplo, n, cuda_data_type, nullptr, lda,
      cuda_data_type, &d_lwork, &h_lwork)));
  size = static_cast<int>(d_lwork);

  // CUDA's potrfBatched needs space for the `as` array, which contains
  // batch_size pointers.  Divide by sizeof(type) because this function returns
  // not bytes but a number of elements of `type`.
  int64_t potrf_batched_scratch = xla::CeilOfRatio<int64_t>(
      batch_size * sizeof(void*), xla::primitive_util::ByteWidth(type));

  return std::max<int64_t>(size, potrf_batched_scratch);
}

absl::Status CudaSolverContext::PotrfBatched(blas::UpperLower uplo, int n,
                                             DeviceMemory<float*> as, int lda,
                                             DeviceMemory<int> lapack_info,
                                             int batch_size) {
  return ConvertStatus(cusolverDnSpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status CudaSolverContext::PotrfBatched(blas::UpperLower uplo, int n,
                                             DeviceMemory<double*> as, int lda,
                                             DeviceMemory<int> lapack_info,
                                             int batch_size) {
  return ConvertStatus(cusolverDnDpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status CudaSolverContext::PotrfBatched(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<float>*> as,
    int lda, DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(cusolverDnCpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status CudaSolverContext::PotrfBatched(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<double>*> as,
    int lda, DeviceMemory<int> lapack_info, int batch_size) {
  return ConvertStatus(cusolverDnZpotrfBatched(
      handle_, GpuBlasUpperLower(uplo), n, ToDevicePointer(as), lda,
      ToDevicePointer(lapack_info), batch_size));
}

absl::Status CudaSolverContext::Potrf(blas::UpperLower uplo, int n,
                                      DeviceMemory<double> a, int lda,
                                      DeviceMemory<int> lapack_info,
                                      DeviceMemory<double> workspace) {
  absl::Status status = ConvertStatus(cusolverDnXpotrf(
      handle_, nullptr, GpuBlasUpperLower(uplo), n, CUDA_R_64F,
      ToDevicePointer(a), lda, CUDA_R_64F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status CudaSolverContext::Potrf(blas::UpperLower uplo, int n,
                                      DeviceMemory<float> a, int lda,
                                      DeviceMemory<int> lapack_info,
                                      DeviceMemory<float> workspace) {
  absl::Status status = ConvertStatus(cusolverDnXpotrf(
      handle_, nullptr, GpuBlasUpperLower(uplo), n, CUDA_R_32F,
      ToDevicePointer(a), lda, CUDA_R_32F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status CudaSolverContext::Potrf(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<float>> a, int lda,
    DeviceMemory<int> lapack_info,
    DeviceMemory<std::complex<float>> workspace) {
  absl::Status status = ConvertStatus(cusolverDnXpotrf(
      handle_, nullptr, GpuBlasUpperLower(uplo), n, CUDA_C_32F,
      ToDevicePointer(a), lda, CUDA_C_32F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

absl::Status CudaSolverContext::Potrf(
    blas::UpperLower uplo, int n, DeviceMemory<std::complex<double>> a, int lda,
    DeviceMemory<int> lapack_info,
    DeviceMemory<std::complex<double>> workspace) {
  absl::Status status = ConvertStatus(cusolverDnXpotrf(
      handle_, nullptr, GpuBlasUpperLower(uplo), n, CUDA_C_64F,
      ToDevicePointer(a), lda, CUDA_C_64F, ToDevicePointer(workspace),
      workspace.ElementCount(), nullptr, 0, ToDevicePointer(lapack_info)));
  return status;
}

}  // namespace stream_executor
