/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Include cuBLAS headers early, and then set EIGEN_HAS_CUDA_FP16
// if we have new enough CUDA (which we will only know after including
// cuda.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if CUDA has already defined one (and conversely, that we do
// not include <cuda_fp16.h> after Half.h has made its typedef).
#include "cuda/include/cuda.h"
#include "cuda/include/cublas_v2.h"

#if CUDA_VERSION >= 7050
#define EIGEN_HAS_CUDA_FP16
#endif

#if CUDA_VERSION >= 8000
#define SE_CUDA_DATA_HALF CUDA_R_16F
#else
#define SE_CUDA_DATA_HALF CUBLAS_DATA_HALF
#endif

#include "tensorflow/stream_executor/cuda/cuda_blas.h"

#include <complex>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/stream_executor/lib/stringprintf.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace perftools {
namespace gputools {
namespace cuda {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

namespace wrap {

#define PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)                      \
  struct WrapperShim__##__name {                                    \
    static const char *kName;                                       \
    template <typename... Args>                                     \
    cublasStatus_t operator()(CUDAExecutor *parent, Args... args) { \
      cuda::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;                                                         \
  const char *WrapperShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(__name)

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSnrm2)                    \
  __macro(cublasDnrm2)                    \
  __macro(cublasScnrm2)                   \
  __macro(cublasDznrm2)                   \
  __macro(cublasSdot)                     \
  __macro(cublasDdot)                     \
  __macro(cublasCdotu)                    \
  __macro(cublasCdotc)                    \
  __macro(cublasZdotu)                    \
  __macro(cublasZdotc)                    \
  __macro(cublasSscal)                    \
  __macro(cublasDscal)                    \
  __macro(cublasCscal)                    \
  __macro(cublasCsscal)                   \
  __macro(cublasZscal)                    \
  __macro(cublasZdscal)                   \
  __macro(cublasSaxpy)                    \
  __macro(cublasDaxpy)                    \
  __macro(cublasCaxpy)                    \
  __macro(cublasZaxpy)                    \
  __macro(cublasScopy)                    \
  __macro(cublasDcopy)                    \
  __macro(cublasCcopy)                    \
  __macro(cublasZcopy)                    \
  __macro(cublasSswap)                    \
  __macro(cublasDswap)                    \
  __macro(cublasCswap)                    \
  __macro(cublasZswap)                    \
  __macro(cublasIsamax)                   \
  __macro(cublasIdamax)                   \
  __macro(cublasIcamax)                   \
  __macro(cublasIzamax)                   \
  __macro(cublasIsamin)                   \
  __macro(cublasIdamin)                   \
  __macro(cublasIcamin)                   \
  __macro(cublasIzamin)                   \
  __macro(cublasSasum)                    \
  __macro(cublasDasum)                    \
  __macro(cublasScasum)                   \
  __macro(cublasDzasum)                   \
  __macro(cublasSrot)                     \
  __macro(cublasDrot)                     \
  __macro(cublasCrot)                     \
  __macro(cublasCsrot)                    \
  __macro(cublasZrot)                     \
  __macro(cublasZdrot)                    \
  __macro(cublasSrotg)                    \
  __macro(cublasDrotg)                    \
  __macro(cublasCrotg)                    \
  __macro(cublasZrotg)                    \
  __macro(cublasSrotm)                    \
  __macro(cublasDrotm)                    \
  __macro(cublasSrotmg)                   \
  __macro(cublasDrotmg)                   \
  __macro(cublasSgemv)                    \
  __macro(cublasDgemv)                    \
  __macro(cublasCgemv)                    \
  __macro(cublasZgemv)                    \
  __macro(cublasSgbmv)                    \
  __macro(cublasDgbmv)                    \
  __macro(cublasCgbmv)                    \
  __macro(cublasZgbmv)                    \
  __macro(cublasStrmv)                    \
  __macro(cublasDtrmv)                    \
  __macro(cublasCtrmv)                    \
  __macro(cublasZtrmv)                    \
  __macro(cublasStbmv)                    \
  __macro(cublasDtbmv)                    \
  __macro(cublasCtbmv)                    \
  __macro(cublasZtbmv)                    \
  __macro(cublasStpmv)                    \
  __macro(cublasDtpmv)                    \
  __macro(cublasCtpmv)                    \
  __macro(cublasZtpmv)                    \
  __macro(cublasStrsv)                    \
  __macro(cublasDtrsv)                    \
  __macro(cublasCtrsv)                    \
  __macro(cublasZtrsv)                    \
  __macro(cublasStpsv)                    \
  __macro(cublasDtpsv)                    \
  __macro(cublasCtpsv)                    \
  __macro(cublasZtpsv)                    \
  __macro(cublasStbsv)                    \
  __macro(cublasDtbsv)                    \
  __macro(cublasCtbsv)                    \
  __macro(cublasZtbsv)                    \
  __macro(cublasSsymv)                    \
  __macro(cublasDsymv)                    \
  __macro(cublasCsymv)                    \
  __macro(cublasZsymv)                    \
  __macro(cublasChemv)                    \
  __macro(cublasZhemv)                    \
  __macro(cublasSsbmv)                    \
  __macro(cublasDsbmv)                    \
  __macro(cublasChbmv)                    \
  __macro(cublasZhbmv)                    \
  __macro(cublasSspmv)                    \
  __macro(cublasDspmv)                    \
  __macro(cublasChpmv)                    \
  __macro(cublasZhpmv)                    \
  __macro(cublasSger)                     \
  __macro(cublasDger)                     \
  __macro(cublasCgeru)                    \
  __macro(cublasCgerc)                    \
  __macro(cublasZgeru)                    \
  __macro(cublasZgerc)                    \
  __macro(cublasSsyr)                     \
  __macro(cublasDsyr)                     \
  __macro(cublasCsyr)                     \
  __macro(cublasZsyr)                     \
  __macro(cublasCher)                     \
  __macro(cublasZher)                     \
  __macro(cublasSspr)                     \
  __macro(cublasDspr)                     \
  __macro(cublasChpr)                     \
  __macro(cublasZhpr)                     \
  __macro(cublasSsyr2)                    \
  __macro(cublasDsyr2)                    \
  __macro(cublasCsyr2)                    \
  __macro(cublasZsyr2)                    \
  __macro(cublasCher2)                    \
  __macro(cublasZher2)                    \
  __macro(cublasSspr2)                    \
  __macro(cublasDspr2)                    \
  __macro(cublasChpr2)                    \
  __macro(cublasZhpr2)                    \
  __macro(cublasSgemm)                    \
  __macro(cublasDgemm)                    \
  __macro(cublasCgemm)                    \
  __macro(cublasZgemm)                    \
  __macro(cublasSsyrk)                    \
  __macro(cublasDsyrk)                    \
  __macro(cublasCsyrk)                    \
  __macro(cublasZsyrk)                    \
  __macro(cublasCherk)                    \
  __macro(cublasZherk)                    \
  __macro(cublasSsyr2k)                   \
  __macro(cublasDsyr2k)                   \
  __macro(cublasCsyr2k)                   \
  __macro(cublasZsyr2k)                   \
  __macro(cublasCher2k)                   \
  __macro(cublasZher2k)                   \
  __macro(cublasSsyrkx)                   \
  __macro(cublasDsyrkx)                   \
  __macro(cublasCsyrkx)                   \
  __macro(cublasZsyrkx)                   \
  __macro(cublasCherkx)                   \
  __macro(cublasZherkx)                   \
  __macro(cublasSsymm)                    \
  __macro(cublasDsymm)                    \
  __macro(cublasCsymm)                    \
  __macro(cublasZsymm)                    \
  __macro(cublasChemm)                    \
  __macro(cublasZhemm)                    \
  __macro(cublasStrsm)                    \
  __macro(cublasDtrsm)                    \
  __macro(cublasCtrsm)                    \
  __macro(cublasZtrsm)                    \
  __macro(cublasStrmm)                    \
  __macro(cublasDtrmm)                    \
  __macro(cublasCtrmm)                    \
  __macro(cublasZtrmm)                    \
  __macro(cublasSgeam)                    \
  __macro(cublasDgeam)                    \
  __macro(cublasCgeam)                    \
  __macro(cublasZgeam)                    \
  __macro(cublasSdgmm)                    \
  __macro(cublasDdgmm)                    \
  __macro(cublasCdgmm)                    \
  __macro(cublasZdgmm)

PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasCreate)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasDestroy)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetStream)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasSetPointerMode)
PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP(cublasGetPointerMode)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasDgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasCgemmBatched)
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasZgemmBatched)
CUBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_CUBLAS_V2_WRAP)

#if CUDA_VERSION >= 7050
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasSgemmEx)
#endif

#if CUDA_VERSION >= 8000
PERFTOOLS_GPUTOOLS_CUBLAS_WRAP(cublasGemmEx)
#endif

}  // namespace wrap

static string ToString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 8000
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return port::StrCat("<invalid cublas status: ", status, ">");
  }
}

// cuBLAS has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with cublasSetPointerMode.
//
// This helper sets the cuBLAS pointer mode to a desired value for a cuBLAS call
// you are about to perform in a given scope.
//
// The prior cuBLAS pointer mode is retained and restored when this object goes
// out of scope.
class ScopedCublasPointerMode {
 public:
  // Note that, because the setting of the cublas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the pointer mode.
  explicit ScopedCublasPointerMode(CUDAExecutor *parent, cublasHandle_t handle)
      : parent_(parent), handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasPointerMode_t new_mode) {
    cublasStatus_t ret =
        wrap::cublasGetPointerMode(parent_, handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = wrap::cublasSetPointerMode(parent_, handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasPointerMode() {
    if (ok_) {
      cublasStatus_t ret =
          wrap::cublasSetPointerMode(parent_, handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  CUDAExecutor *parent_;   // Executor establishing this pointer mode for.
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
  cublasPointerMode_t old_mode_;  // Prior cuBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};

bool CUDABlas::Init() {
  cublasStatus_t ret = wrap::cublasCreate(parent_, &blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    return false;
  }

  return true;
}

CUDABlas::CUDABlas(cuda::CUDAExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

CUDABlas::~CUDABlas() {
  if (blas_ != nullptr) {
    wrap::cublasDestroy(parent_, blas_);
  }
}

bool CUDABlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsCUDAStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  cublasStatus_t ret =
      wrap::cublasSetStream(parent_, blas_, AsCUDAStreamValue(stream));
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into cuBLAS arguments.

cublasOperation_t CUDABlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return CUBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return CUBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return CUBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

cublasFillMode_t CUDABlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return CUBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return CUBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

cublasDiagType_t CUDABlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return CUBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return CUBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

cublasSideMode_t CUDABlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return CUBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return CUBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

// CUDADataType<T>::type translates from a C++ type (e.g. float) to a
// cudaDataType_t (e.g. CUDA_R_32F).  CUDAComputationType(ty) translates from a
// blas::ComputationType to a cudaDataType_t.
//
// These are used to build the argument type and computation type args to
// cublasGemmEx.  cublasGemmEx and cudaDataType_t are available only on
// CUDA >= 8.0.
#if CUDA_VERSION >= 8000
template <typename T>
struct CUDADataType;

template <>
struct CUDADataType<Eigen::half> {
  static constexpr cudaDataType_t type = SE_CUDA_DATA_HALF;
};

template <>
struct CUDADataType<std::complex<Eigen::half>> {
  static constexpr cudaDataType_t type = CUDA_C_16F;
};

template <>
struct CUDADataType<float> {
  static constexpr cudaDataType_t type = CUDA_R_32F;
};

template <>
struct CUDADataType<std::complex<float>> {
  static constexpr cudaDataType_t type = CUDA_C_32F;
};

template <>
struct CUDADataType<double> {
  static constexpr cudaDataType_t type = CUDA_R_64F;
};

template <>
struct CUDADataType<std::complex<double>> {
  static constexpr cudaDataType_t type = CUDA_C_64F;
};

template <>
struct CUDADataType<int8> {
  static constexpr cudaDataType_t type = CUDA_R_8I;
};

template <>
struct CUDADataType<std::complex<int8>> {
  static constexpr cudaDataType_t type = CUDA_C_8I;
};

template <>
struct CUDADataType<uint8> {
  static constexpr cudaDataType_t type = CUDA_R_8U;
};

template <>
struct CUDADataType<std::complex<uint8>> {
  static constexpr cudaDataType_t type = CUDA_C_8U;
};

cudaDataType_t CUDAComputationType(blas::ComputationType ty) {
  switch (ty) {
    case blas::ComputationType::kF16:
      return CUDA_R_16F;
    case blas::ComputationType::kF32:
      return CUDA_R_32F;
    case blas::ComputationType::kF64:
      return CUDA_R_64F;
    case blas::ComputationType::kComplexF32:
      return CUDA_C_32F;
    case blas::ComputationType::kComplexF64:
      return CUDA_C_64F;
  }
}
#endif

}  // namespace

template <typename FuncT, typename... Args>
bool CUDABlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  ScopedCublasPointerMode pointer_mode{parent_, blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }

  cublasStatus_t ret = cublas_func(parent_, blas_, args...);
  if (err_on_failure && ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cuBLAS routine " << cublas_func.kName << ": "
               << ToString(ret);
  }
  return ret == CUBLAS_STATUS_SUCCESS;
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::cublasSasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::cublasDasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasScasum, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDzasum, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCaxpy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZaxpy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasScopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemory(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemory(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasSdot, stream, false /* = pointer_mode_host */, elem_count,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDdot, stream, false /* = pointer_mode_host */, elem_count,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(
      wrap::cublasCdotc, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(
      wrap::cublasZdotc, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(
      wrap::cublasCdotu, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(
      wrap::cublasZdotu, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(result)));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::cublasSnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::cublasDnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::cublasScnrm2, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::cublasDznrm2, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return DoBlasInternal(
      wrap::cublasSrot, stream, true /* = pointer_mode_host */, elem_count,
      CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return DoBlasInternal(
      wrap::cublasDrot, stream, true /* = pointer_mode_host */, elem_count,
      CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return DoBlasInternal(wrap::cublasCsrot, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemoryMutable(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return DoBlasInternal(wrap::cublasZdrot, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemoryMutable(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return DoBlasInternal(wrap::cublasSrotg, stream,
                        false /* = pointer_mode_host */, CUDAMemoryMutable(a),
                        CUDAMemoryMutable(b), CUDAMemoryMutable(c),
                        CUDAMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return DoBlasInternal(wrap::cublasDrotg, stream,
                        false /* = pointer_mode_host */,
                        CUDAComplex(CUDAMemoryMutable(a)), CUDAMemoryMutable(b),
                        CUDAMemoryMutable(c), CUDAMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return DoBlasInternal(
      wrap::cublasCrotg, stream, false /* = pointer_mode_host */,
      CUDAComplex(CUDAMemoryMutable(a)), CUDAComplex(CUDAMemoryMutable(b)),
      CUDAComplex(CUDAMemoryMutable(c)), CUDAComplex(CUDAMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return DoBlasInternal(
      wrap::cublasZrotg, stream, false /* = pointer_mode_host */,
      CUDAComplex(CUDAMemoryMutable(a)), CUDAComplex(CUDAMemoryMutable(b)),
      CUDAComplex(CUDAMemoryMutable(c)), CUDAComplex(CUDAMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return DoBlasInternal(wrap::cublasSrotm, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy,
                        CUDAMemory(param));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return DoBlasInternal(wrap::cublasDrotm, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy,
                        CUDAMemory(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return DoBlasInternal(wrap::cublasSrotmg, stream,
                        false /* = pointer_mode_host */, CUDAMemoryMutable(d1),
                        CUDAMemoryMutable(d2), CUDAMemoryMutable(x1),
                        CUDAMemory(y1), CUDAMemoryMutable(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return DoBlasInternal(wrap::cublasDrotmg, stream,
                        false /* = pointer_mode_host */, CUDAMemoryMutable(d1),
                        CUDAMemoryMutable(d2), CUDAMemoryMutable(x1),
                        CUDAMemory(y1), CUDAMemoryMutable(param));
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasSscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasCsscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasZdscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasCscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasZscal, stream, true /* = pointer_mode_host */, elem_count,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAMemoryMutable(x), incx, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::cublasCswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemoryMutable(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::cublasZswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        CUDAComplex(CUDAMemoryMutable(x)), incx,
                        CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::cublasIsamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::cublasIdamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        CUDAMemory(x), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIcamax, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIzamax, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIsamin, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIdamin, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIcamin, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::cublasIzamin, stream, false /* = pointer_mode_host */, elem_count,
      CUDAComplex(CUDAMemory(x)), incx, CUDAMemoryMutable(result));
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSgbmv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, kl, ku, &alpha, CUDAMemory(a), lda,
      CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDgbmv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, kl, ku, &alpha, CUDAMemory(a), lda,
      CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasCgbmv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, kl, ku, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZgbmv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, kl, ku, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasCgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZgemv, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(trans), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasDger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      CUDAMemory(x), incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCgerc, stream, true /* = pointer_mode_host */, m, n,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZgerc, stream, true /* = pointer_mode_host */, m, n,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCgeru, stream, true /* = pointer_mode_host */, m, n,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZgeru, stream, true /* = pointer_mode_host */, m, n,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemory(y)), incy, CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, k, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, k, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChemv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhemv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCher, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, &alpha, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZher, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, &alpha, CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasCher2, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(
      wrap::cublasZher2, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasChpmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(ap)), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasZhpmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(ap)), CUDAComplex(CUDAMemory(x)), incx,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return DoBlasInternal(
      wrap::cublasChpr, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return DoBlasInternal(
      wrap::cublasZhpr, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  return DoBlasInternal(
      wrap::cublasChpr2, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  return DoBlasInternal(
      wrap::cublasZhpr2, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(x)), incx, CUDAComplex(CUDAMemory(y)), incy,
      CUDAComplex(CUDAMemoryMutable(ap)));
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasSsbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, k, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::cublasDsbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), n, k, &alpha, CUDAMemory(a), lda, CUDAMemory(x),
      incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSspmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(ap),
                        CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDspmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(ap),
                        CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return DoBlasInternal(wrap::cublasSspr, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return DoBlasInternal(wrap::cublasDspr, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return DoBlasInternal(wrap::cublasSspr2, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemory(y), incy, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return DoBlasInternal(wrap::cublasDspr2, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemory(y), incy, CUDAMemoryMutable(ap));
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::cublasSsymv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(a), lda,
                        CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::cublasDsymv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(a), lda,
                        CUDAMemory(x), incx, &beta, CUDAMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::cublasSsyr, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::cublasDsyr, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::cublasSsyr2, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::cublasDsyr2, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, CUDAMemory(x),
                        incx, CUDAMemory(y), incy, CUDAMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStbmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtbmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasCtbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasZtbmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStbsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtbsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasCtbsv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasZtbsv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, k, CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasStpmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasDtpmv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtpmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtpmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(
      wrap::cublasStpsv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(
      wrap::cublasDtpsv, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
      CUDABlasDiagonal(diag), n, CUDAMemory(ap), CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtpsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtpsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(ap)),
                        CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStrmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtrmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtrmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtrmv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::cublasStrsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::cublasDtrsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAMemory(a), lda,
                        CUDAMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::cublasCtrsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::cublasZtrsv, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, CUDAComplex(CUDAMemory(a)),
                        lda, CUDAComplex(CUDAMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if CUDA_VERSION >= 7050
  VLOG(1) << port::Printf(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  // TODO(sesse): Consider supporting the Hgemm interface, which uses half
  // calculations internally (faster on newer devices, such as Pascal and TX1,
  // but less precise).
  return DoBlasInternal(
      wrap::cublasSgemmEx, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), SE_CUDA_DATA_HALF, lda, CUDAMemory(b), SE_CUDA_DATA_HALF,
      ldb, &beta, CUDAMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);
#else
  LOG(ERROR) << "fp16 sgemm is not implemented in this cuBLAS version "
             << "(need at least CUDA 7.5)";
  return false;
#endif
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  VLOG(1) << port::Printf(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
      "k=%llu alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
  if (transa == blas::Transpose::kNoTranspose) {
    if (lda < static_cast<int64>(m)) {
      LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                      "precondition violation";
    }
  } else {
    if (lda < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                   << ") (transpose case); precondition violation";
    }
  }
  if (transb == blas::Transpose::kNoTranspose) {
    if (ldb < static_cast<int64>(k)) {
      LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                   << ") (no transpose case); precondition violation";
    }
  } else {
    if (ldb < static_cast<int64>(n)) {
      LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                      "precondition violation";
    }
  }
  return DoBlasInternal(
      wrap::cublasSgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZgemm, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
      CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

template <typename T>
bool CUDABlas::DoBlasGemmWithAlgorithmImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const T &alpha, const DeviceMemory<T> &a, int lda,
    const DeviceMemory<T> &b, int ldb, const T &beta, DeviceMemory<T> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
// CUDA < version 8 and GPUs < sm_50 don't support cublasGemmEx.
#if CUDA_VERSION < 8000
  return false;
#else
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().cuda_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major < 5) {
    return false;
  }

  struct TimerDeleter {
    void operator()(CUDATimer *t) {
      t->Destroy();
      delete t;
    }
  };
  std::unique_ptr<CUDATimer, TimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new CUDATimer(parent_));
    if (!timer->Init() || !timer->Start(AsCUDAStream(stream))) {
      return false;
    }
  }

  cudaDataType_t data_type = CUDADataType<T>::type;
  bool result = DoBlasInternalFailureOK(
      wrap::cublasGemmEx, stream, /* pointer_mode_host = */ true,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      CUDAMemory(a), data_type, lda, CUDAMemory(b), data_type, ldb, &beta,
      CUDAMemoryMutable(c), data_type, ldc,
      CUDAComputationType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm));

  if (timer != nullptr && result) {
    // CUDATimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsCUDAStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
#endif
}

bool CUDABlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
// cublasGemmAlgo_t (and the function that accepts this type, cublasGemmEx)
// were first introduced in CUDA 8.
#if CUDA_VERSION >= 8000
  for (cublasGemmAlgo_t algo :
       {CUBLAS_GEMM_DFALT, CUBLAS_GEMM_ALGO0, CUBLAS_GEMM_ALGO1,
        CUBLAS_GEMM_ALGO2, CUBLAS_GEMM_ALGO3, CUBLAS_GEMM_ALGO4,
        CUBLAS_GEMM_ALGO5, CUBLAS_GEMM_ALGO6, CUBLAS_GEMM_ALGO7}) {
    out_algorithms->push_back(algo);
  }
#endif
  return true;
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const Eigen::half &alpha,
    const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, const Eigen::half &beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

template <typename T, typename FuncT>
port::Status CUDABlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename CUDAComplexT<T>::type CUDA_T;

  const size_t size = batch_count * sizeof(CUDA_T *);

  // Device-side copy of pointers to matrices.
  DeviceMemory<CUDA_T *> a;
  DeviceMemory<CUDA_T *> b;
  DeviceMemory<CUDA_T *> c;

  // If temporary space is allocated for device-side copies of pointers to
  // matrices, that temporary space should not be freed until this function
  // returns. Although the values for these unique_ptrs are not set here, they
  // are declared at this scope so they will be destroyed when the function
  // returns.
  //
  // If a scratch allocator is provided, these pointers will not be used at all.
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> a_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> b_temporary;
  std::unique_ptr<TemporaryDeviceMemory<CUDA_T *>> c_temporary;

  // Decide how to allocate device-side copy of pointers to matrices based on
  // whether a scratch allocator was passed.
  if (scratch_allocator != nullptr) {
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
                        scratch_allocator->AllocateBytes(stream, size));
    a = DeviceMemory<CUDA_T *>(a_bytes);
    b = DeviceMemory<CUDA_T *>(b_bytes);
    c = DeviceMemory<CUDA_T *>(c_bytes);
  } else {
    SE_ASSIGN_OR_RETURN(a_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(b_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    SE_ASSIGN_OR_RETURN(c_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    a = DeviceMemory<CUDA_T *>(*a_temporary->mutable_device_memory());
    b = DeviceMemory<CUDA_T *>(*b_temporary->mutable_device_memory());
    c = DeviceMemory<CUDA_T *>(*c_temporary->mutable_device_memory());
  }

  if (!stream->ThenMemcpy(&a, a_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&b, b_raw_ptrs.data(), size).ok() ||
      !stream->ThenMemcpy(&c, c_raw_ptrs.data(), size).ok()) {
    return port::Status(port::error::INTERNAL,
                        "failed to copy memory from host to device in "
                        "CUDABlas::DoBlasGemmBatched");
  }

  bool ok = DoBlasInternal(
      cublas_func, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      CUDAComplex(&alpha), const_cast<const CUDA_T **>(CUDAMemory(a)), lda,
      const_cast<const CUDA_T **>(CUDAMemory(b)), ldb, CUDAComplex(&beta),
      const_cast<CUDA_T **>(CUDAMemory(c)), ldc, batch_count);

  if (ok) {
    return port::Status::OK();
  }
  return port::Status(port::error::INTERNAL,
                      "failed BLAS call, see log for details");
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
      lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasChemm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZhemm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return DoBlasInternal(wrap::cublasCherk, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        &beta, CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return DoBlasInternal(wrap::cublasZherk, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        &beta, CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  return DoBlasInternal(wrap::cublasCher2k, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        CUDAComplex(CUDAMemory(b)), ldb, &beta,
                        CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  return DoBlasInternal(wrap::cublasZher2k, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        CUDAComplex(CUDAMemory(b)), ldb, &beta,
                        CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsymm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, &alpha, CUDAMemory(a),
      lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsymm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, &alpha, CUDAMemory(a),
      lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCsymm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZsymm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemory(b)), ldb,
      CUDAComplex(&beta), CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsyrk, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      CUDAMemory(a), lda, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsyrk, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      CUDAMemory(a), lda, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasCsyrk, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasZsyrk, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k,
      CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(&beta),
      CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasSsyr2k, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::cublasDsyr2k, stream, true /* = pointer_mode_host */,
      CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n, k, &alpha,
      CUDAMemory(a), lda, CUDAMemory(b), ldb, &beta, CUDAMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  return DoBlasInternal(wrap::cublasCsyr2k, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
                        CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  return DoBlasInternal(wrap::cublasZsyr2k, stream,
                        true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, CUDAComplex(&alpha), CUDAComplex(CUDAMemory(a)), lda,
                        CUDAComplex(CUDAMemory(b)), ldb, CUDAComplex(&beta),
                        CUDAComplex(CUDAMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasStrmm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a), lda,
      CUDAMemoryMutable(b), ldb, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasDtrmm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a), lda,
      CUDAMemoryMutable(b), ldb, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasCtrmm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb,
      CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasZtrmm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb,
      CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(wrap::cublasStrsm, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a),
                        lda, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(wrap::cublasDtrsm, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, CUDAMemory(a),
                        lda, CUDAMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasCtrsm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(
      wrap::cublasZtrsm, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), CUDABlasTranspose(transa),
      CUDABlasDiagonal(diag), m, n, CUDAComplex(&alpha),
      CUDAComplex(CUDAMemory(a)), lda, CUDAComplex(CUDAMemoryMutable(b)), ldb);
}

}  // namespace cuda

namespace gpu = ::perftools::gputools;

void initialize_cublas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::cuda::kCudaPlatformId, gpu::cuda::kCuBlasPlugin, "cuBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::cuda::CUDAExecutor *cuda_executor =
                    dynamic_cast<gpu::cuda::CUDAExecutor *>(parent);
                if (cuda_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the cuBLAS "
                      << "support library with a non-CUDA StreamExecutor";
                  return nullptr;
                }

                gpu::cuda::CUDABlas *blas =
                    new gpu::cuda::CUDABlas(cuda_executor);
                if (!blas->Init()) {
                  // Note: Init() will log a more specific error.
                  delete blas;
                  return nullptr;
                }
                return blas;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuBLAS factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::cuda::kCudaPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::cuda::kCuBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { perftools::gputools::initialize_cublas(); });
