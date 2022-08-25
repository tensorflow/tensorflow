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

#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"

#define SE_CUDA_DATA_HALF CUDA_R_16F

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas.h"

// Both Eigen Half.h and CUDA cuda_fp16.h provide similar typedef for __half. As
// such, there are two ways to get the typedef for __half:
//
// (1) Includes cuda_fp16.h and defines EIGEN_HAS_CUDA_FP16.
// (2) Neither includes cuda_fp16.h nor defines EIGEN_HAS_CUDA_FP16.
//
// Due to issue b/73793421, when the first approach is used and NVCC is used to
// compile this file, NVCC will complain duplicated definition for
// EIGEN_HAS_CUDA_FP16. On the other hand, when the second approach is used and
// clang is used to compile this file, clang will not understand __half
// due to missing the definition and macro EIGEN_HAS_CUDA_FP16.
//
// Because this file may be compiled with CLANG but will never be compiled with
// NVCC, we choose the first approach for CUDA < 9.0. For CUDA >= 9.0, we have
// to use the second approach because the data member in the __half defined
// by CUDA > 9.0 is `__x` while Eigen expects it to be `x`.
//
// TODO(b/73793421): Remove the following code block to switch to the second
// approach when the issue is fixed.
#if CUDA_VERSION < 9000
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#define EIGEN_HAS_CUDA_FP16
#endif

#include <complex>
#include <cstdint>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_utils.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_timer.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/stream_executor/lib/initialize.h"
#include "tensorflow/compiler/xla/stream_executor/lib/status.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"

namespace stream_executor {
namespace cuda {

using gpu::AsGpuStream;
using gpu::AsGpuStreamValue;
using gpu::GpuComplex;
using gpu::GpuComplexT;
using gpu::GpuComplexType;
using gpu::GpuComplexValue;
using gpu::GpuDoubleComplexType;
using gpu::GpuExecutor;
using gpu::GpuMemory;
using gpu::GpuMemoryMutable;
using gpu::GpuTimer;
using gpu::GpuTimerDeleter;

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

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
  explicit ScopedCublasPointerMode(cublasHandle_t handle)
      : handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasPointerMode_t new_mode) {
    cublasStatus_t ret = cublasGetPointerMode(handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = cublasSetPointerMode(handle_, new_mode);
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
      cublasStatus_t ret = cublasSetPointerMode(handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  cublasHandle_t handle_;         // Handle to the cuBLAS instance of interest.
  cublasPointerMode_t old_mode_;  // Prior cuBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};

#if CUDA_VERSION >= 9000
// cuBLAS has interfaces that permit computations to use the Volta hardware.
// This must be enabled via the cublasGet/SetMathMode APIs.
//
// This helper sets the cuBLAS math mode to a desired value for a cuBLAS call
// you are about to perform in a given scope.
//
// The prior cuBLAS math mode is retained and restored when this object goes
// out of scope.
class ScopedCublasMathMode {
 public:
  // Note that, because the setting of the cublas math mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The cublas library handle to act upon in setting the math mode.
  explicit ScopedCublasMathMode(cublasHandle_t handle)
      : handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped math mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(cublasMath_t new_mode) {
    cublasStatus_t ret = cublasGetMathMode(handle_, &old_mode_);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old cublas math mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = cublasSetMathMode(handle_, new_mode);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new cublas math mode: " << ToString(ret);
      return ok_ = false;
    }
    return ok_ = true;
  }

  // Switches back to the prior math mode, if the switch operation was
  // successful in the first place.
  ~ScopedCublasMathMode() {
    if (ok_) {
      cublasStatus_t ret = cublasSetMathMode(handle_, old_mode_);
      if (ret != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former cublas math mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
  cublasMath_t old_mode_;  // Prior cuBLAS math mode, to be restored.
  bool ok_;                // Whether the change was successful.
};
#endif  // CUDA_VERSION >= 9000

static const char *const kCublasNotInitializedExplanation =
    "Failure to initialize cublas may be due to OOM (cublas needs some free "
    "memory when you initialize it, and your deep-learning framework may have "
    "preallocated more than its fair share), or may be because this binary was "
    "not built with support for the GPU in your machine.";

bool CUDABlas::Init() {
  gpu::ScopedActivateExecutorContext sac{parent_};
  cublasStatus_t ret = cublasCreate(&blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    if (ret == CUBLAS_STATUS_NOT_INITIALIZED) {
      LOG(ERROR) << kCublasNotInitializedExplanation;
    }
    return false;
  }

#if CUDA_VERSION >= 11000
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << kCublasNotInitializedExplanation;
    return false;
  }
#endif  // CUDA_VERSION >= 11000

  return true;
}

CUDABlas::CUDABlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)),
      blas_(nullptr)
#if CUDA_VERSION >= 11000
      ,
      blas_lt_(parent)
#endif
{
}

CUDABlas::~CUDABlas() {
  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    cublasDestroy(blas_);
  }
}

bool CUDABlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  cublasStatus_t ret = cublasSetStream(blas_, AsGpuStreamValue(stream));
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

cudaStream_t CUDABlas::CUDAStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  return AsGpuStreamValue(stream);
}

namespace {

// Helper functions transforming blas arguments into cuBLAS arguments.

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
// cudaDataType_t (e.g. CUDA_R_32F).
//
// These are used to build the argument type and computation type args to
// cublasGemmEx.
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
struct CUDADataType<int> {
  static constexpr cudaDataType_t type = CUDA_R_32I;
};

template <>
struct CUDADataType<int8_t> {
  static constexpr cudaDataType_t type = CUDA_R_8I;
};

template <>
struct CUDADataType<std::complex<int8_t>> {
  static constexpr cudaDataType_t type = CUDA_C_8I;
};

template <>
struct CUDADataType<uint8_t> {
  static constexpr cudaDataType_t type = CUDA_R_8U;
};

template <>
struct CUDADataType<std::complex<uint8_t>> {
  static constexpr cudaDataType_t type = CUDA_C_8U;
};

}  // namespace

template <typename FuncT, typename... Args>
port::Status CUDABlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                          bool pointer_mode_host,
                                          cublasMath_t math_type,
                                          Args... args) {
  absl::MutexLock lock(&mu_);

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return port::InternalError("Failed setting stream");
  }

#if CUDA_VERSION >= 9000
  ScopedCublasMathMode math_mode{blas_};
#if CUBLAS_VER_MAJOR >= 11
  if (math_type == CUBLAS_TF32_TENSOR_OP_MATH &&
      tensorflow::tensor_float_32_execution_enabled()) {
#else
  if (math_type == CUBLAS_TENSOR_OP_MATH) {
#endif
    if (!math_mode.Init(math_type)) {
      return port::InternalError("Failed initializing math mode");
    }
  }
#endif

  gpu::ScopedActivateExecutorContext sac{parent_};
  ScopedCublasPointerMode pointer_mode{blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return port::InternalError("Failed setting error mode");
  }
  cublasStatus_t ret = cublas_func(blas_, args...);
  if (ret == CUBLAS_STATUS_SUCCESS) {
    return ::tensorflow::OkStatus();
  }
  return port::InternalError(ToString(ret));
}

// cublas_func may be overloaded, so we need to figure out which one we really
// need to call based on the args. One way to do it is to wrap it in lambda.
#define AS_LAMBDA(func)                                            \
  [](auto &&...args) -> decltype(func(                             \
                         std::forward<decltype(args)>(args)...)) { \
    return func(std::forward<decltype(args)>(args)...);            \
  }

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasScopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(cublasCcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(cublasZcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasSscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCsscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZdscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

port::Status CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                  blas::Transpose transb, uint64_t m, uint64 n,
                                  uint64_t k, blas::DataType dtype,
                                  const void *alpha, const DeviceMemoryBase &a,
                                  int lda, const DeviceMemoryBase &b, int ldb,
                                  const void *beta, DeviceMemoryBase *c,
                                  int ldc, blas::ComputePrecision precision) {
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;

#if CUDA_VERSION < 11000
  if (dtype == blas::DataType::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == blas::DataType::kFloat) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
    if (stream->GetCudaComputeCapability().IsAtLeast(
            CudaComputeCapability::AMPERE)) {
      // TODO(reedwm): Remove or make this VLOG(1) once TensorFloat-32 is more
      // well tested.
      if (tensorflow::tensor_float_32_execution_enabled()) {
        LOG_FIRST_N(INFO, 1) << "TensorFloat-32 will be used for the matrix "
                                "multiplication. This will only be logged "
                                "once.";
      }
    }
    if (precision > blas::kDefaultComputePrecision) {
      math_type = CUBLAS_DEFAULT_MATH;
    }
  }
#endif

  // TODO(cheshire): Return an error instead.
  // TODO(cheshire): Why are these checked only for `half` and `float`?
  if (dtype == blas::DataType::kHalf || dtype == blas::DataType::kFloat) {
    if (transa == blas::Transpose::kNoTranspose) {
      if (lda < static_cast<int64_t>(m)) {
        LOG(WARNING) << "GEMM lda was smaller than m (no transpose case); "
                        "precondition violation";
      }
    } else {
      if (lda < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM lda (" << lda << ") was smaller than k (" << k
                     << ") (transpose case); precondition violation";
      }
    }
    if (transb == blas::Transpose::kNoTranspose) {
      if (ldb < static_cast<int64_t>(k)) {
        LOG(WARNING) << "GEMM ldb (" << ldb << ") was smaller than k (" << k
                     << ") (no transpose case); precondition violation";
      }
    } else {
      if (ldb < static_cast<int64_t>(n)) {
        LOG(WARNING) << "GEMM ldb was smaller than n (transpose case); "
                        "precondition violation";
      }
    }
  }

  VLOG(1) << absl::StrFormat(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%u n=%u "
      "k=%u alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);

  switch (dtype) {
    case blas::DataType::kHalf: {
#if CUDA_VERSION < 7050
      return port::InternalError(
          "fp16 sgemm is not implemented in this cuBLAS version "
          "(need at least CUDA 7.5)");
#endif

      return DoBlasInternalImpl(
          cublasSgemmEx, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          static_cast<const float *>(alpha), a.opaque(), SE_CUDA_DATA_HALF, lda,
          b.opaque(), SE_CUDA_DATA_HALF, ldb, static_cast<const float *>(beta),
          c->opaque(), SE_CUDA_DATA_HALF, ldc);
    }
#if CUDA_VERSION > 11000
    case blas::DataType::kBF16: {
      return DoBlasInternalImpl(
          cublasSgemmEx, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          static_cast<const float *>(alpha), a.opaque(), CUDA_R_16BF, lda,
          b.opaque(), CUDA_R_16BF, ldb, static_cast<const float *>(beta),
          c->opaque(), CUDA_R_16BF, ldc);
    }
#endif
    case dnn::kFloat:
      return DoBlasInternalImpl(
          cublasSgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda,
          static_cast<const float *>(b.opaque()), ldb,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc);
    case dnn::kDouble:
      return DoBlasInternalImpl(
          cublasDgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda,
          static_cast<const double *>(b.opaque()), ldb,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc);
    case dnn::kComplexFloat: {
      GpuComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<float> *>(alpha));
      GpuComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          &cb_alpha, static_cast<const GpuComplexType *>(a.opaque()), lda,
          static_cast<const GpuComplexType *>(b.opaque()), ldb, &cb_beta,
          static_cast<GpuComplexType *>(c->opaque()), ldc);
    }
    case dnn::kComplexDouble: {
      GpuDoubleComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<double> *>(alpha));
      GpuDoubleComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          &cb_alpha, static_cast<const GpuDoubleComplexType *>(a.opaque()), lda,
          static_cast<const GpuDoubleComplexType *>(b.opaque()), ldb, &cb_beta,
          static_cast<GpuDoubleComplexType *>(c->opaque()), ldc);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

template <typename T>
bool CUDABlas::DoBlasGemvWithProfilingImpl(
    Stream *stream, blas::Transpose trans, uint64_t m, uint64 n, const T &alpha,
    const DeviceMemory<T> &a, int lda, const DeviceMemory<T> &x, int incx,
    const T &beta, DeviceMemory<T> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  // Call blasGemm
  bool result =
      DoBlasGemv(stream, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

  if (timer != nullptr && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(blas::kDefaultBlasGemv);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

template <typename T, typename ParamType>
bool CUDABlas::DoBlasGemmWithProfilingImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
    int lda, const DeviceMemory<T> &b, int ldb, const ParamType &beta,
    DeviceMemory<T> *c, int ldc, blas::ProfileResult *output_profile_result) {
  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  // Call blasGemm
  bool result = DoBlasGemm(stream, transa, transb, m, n, k,
                           blas::ToDataType<T>::value, &alpha, a, lda, b, ldb,
                           &beta, c, ldc, blas::kDefaultComputePrecision)
                    .ok();

  if (timer != nullptr && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(blas::kDefaultBlasGemm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

static bool UsesTensorOps(blas::AlgorithmType algo) {
#if CUDA_VERSION >= 9000
  cublasGemmAlgo_t cublas_algo = static_cast<cublasGemmAlgo_t>(algo);
  return cublas_algo >= CUBLAS_GEMM_DEFAULT_TENSOR_OP;
#else
  return false;
#endif
}

static port::StatusOr<cublasMath_t> GetMathTypeForGemmEx(
    Stream *stream, blas::AlgorithmType algorithm, blas::DataType type_a,
    blas::DataType type_b, blas::ComputePrecision precision) {
  if (type_a != type_b) {
    return port::InternalError("Types of inputs mismatch");
  }

  // GPUs < sm_50 don't support cublasGemmEx.
  CudaComputeCapability cc = stream->GetCudaComputeCapability();
  if (cc.major < 5) {
    return port::InternalError(absl::StrCat(
        "sm_", cc.major, " does not support explicit gemm algorithms."));
  }

  bool algo_uses_tensor_ops = UsesTensorOps(algorithm);
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
  if (algo_uses_tensor_ops) {
    if (cc.major < 7) {
      return port::InternalError(absl::StrCat(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available in sm", cc.major,
          "X devices."));
    } else if (type_a == blas::DataType::kFloat) {
#if CUDA_VERSION < 11000
      return port::InternalError(absl::StrCat(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available for fp32"));
#else
      if (cc.major < 8) {
        return port::InternalError(absl::StrCat(
            "Algorithm ", algorithm,
            " uses tensor ops, but tensor ops are not available in sm",
            cc.major, "X devices for float input types."));
      } else if (!tensorflow::tensor_float_32_execution_enabled()) {
        return port::InternalError(absl::StrCat(
            "Algorithm ", algorithm,
            " uses tensor ops, but tensor ops are disabled for fp32 inputs"));
      }
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
#endif
    } else if (type_a == blas::DataType::kHalf) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#endif
    } else {
      return port::InternalError(
          absl::StrCat("Algorithm ", algorithm,
                       " uses tensor ops which are not supported for input"));
    }
  }
  if (precision > blas::kDefaultComputePrecision) {
    math_type = CUBLAS_DEFAULT_MATH;
  }

  // Return false if we might be hitting a cuBLAS bug that produces the wrong
  // result. See nvbugs/2156201, b/79126339.
#if CUDA_VERSION >= 9000 && CUDA_VERSION < 9020
  if ((algorithm == CUBLAS_GEMM_DEFAULT || algorithm >= CUBLAS_GEMM_ALGO13) &&
      std::max({m, n, k}) >= 2097153 && cc_major < 7) {
    return port::InternalError(
        "DoBlasGemmWithAlgorithm returning false to work around cudnn "
        "<9.2 bug with m, n, or k >= 2097153.  See b/79126339.");
  }
#endif
  return math_type;
}

static port::StatusOr<std::unique_ptr<GpuTimer, GpuTimerDeleter>>
StartGpuTimerForProfile(Stream *stream, GpuExecutor *executor,
                        blas::ProfileResult *output_profile_result) {
  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result) {
    timer.reset(new GpuTimer(executor));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return port::InternalError(
          "output_profile_result given, but unable to create a GpuTimer");
    }
  }
  return timer;
}

static port::Status PopulateProfileFromTimer(
    GpuTimer *timer, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result, Stream *stream) {
  if (timer) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return port::InternalError("unable to stop GpuTimer.");
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return ::tensorflow::OkStatus();
}

port::Status CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ComputePrecision precision,
    blas::ProfileResult *output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      cublasMath_t math_type,
      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b, precision));

  TF_ASSIGN_OR_RETURN(auto timer, StartGpuTimerForProfile(
                                      stream, parent_, output_profile_result));

  // Since we are converting 'algorithm' to cublasGemmAlgo_t by static_cast,
  // we do the following compile-time check on the default value:
  static_assert(blas::kDefaultGemmAlgo == CUBLAS_GEMM_DFALT, "");

  TF_RETURN_IF_ERROR(DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true, math_type,
      AsCublasOperation(transa), AsCublasOperation(transb), m, n, k, alpha,
      a.opaque(), AsCudaDataType(type_a), lda, b.opaque(),
      AsCudaDataType(type_b), ldb, beta, c->opaque(), AsCudaDataType(type_c),
      ldc, AsCublasComputeType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm)));
  TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                              output_profile_result, stream));
  return ::tensorflow::OkStatus();
}

port::Status CUDABlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ComputePrecision precision,
    blas::ProfileResult *output_profile_result) {
  TF_ASSIGN_OR_RETURN(
      cublasMath_t math_type,
      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b, precision));
  TF_ASSIGN_OR_RETURN(auto timer, StartGpuTimerForProfile(
                                      stream, parent_, output_profile_result));

  cudaDataType_t cuda_in_type = AsCudaDataType(type_a);

#if CUDA_VERSION >= 11000
  // Workaround CUDA bug where batched GEMM is erroneously marked as
  // unsupported by manually unbatching it on Pascal.
  if (cuda_in_type == CUDA_R_16BF &&
      !stream->GetCudaComputeCapability().IsAtLeast(7)) {
    for (int batch = 0; batch < batch_count; ++batch) {
      const auto *a_matrix = reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<const Eigen::bfloat16 *>(a.opaque()) + batch * stride_a);
      const auto *b_matrix = reinterpret_cast<const __nv_bfloat16 *>(
          static_cast<const Eigen::bfloat16 *>(b.opaque()) + batch * stride_b);
      auto *c_matrix = reinterpret_cast<__nv_bfloat16 *>(
          static_cast<Eigen::bfloat16 *>(c->opaque()) + batch * stride_c);
      TF_RETURN_IF_ERROR(DoBlasInternalImpl(
          AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
          b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
          c_matrix, CUDA_R_16BF, ldc, AsCublasComputeType(computation_type),
          static_cast<cublasGemmAlgo_t>(algorithm)));
    }
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                                output_profile_result, stream));
    return port::Status::OK();
  }
#endif

  TF_RETURN_IF_ERROR(DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmStridedBatchedEx), stream, /*pointer_mode_host=*/true,
      math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
      alpha, a.opaque(), cuda_in_type, lda, stride_a, b.opaque(), cuda_in_type,
      ldb, stride_b, beta, c->opaque(), AsCudaDataType(type_c), ldc, stride_c,
      batch_count, AsCublasComputeType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm)));
  TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                              output_profile_result, stream));
  return ::tensorflow::OkStatus();
}

bool CUDABlas::GetBlasGemmAlgorithms(
    Stream *stream, std::vector<blas::AlgorithmType> *out_algorithms) {
  // cublasGemmAlgo_t (and the function that accepts this type, cublasGemmEx)
  // were first introduced in CUDA 8.
  //
  // Note that when CUDA version and compute capability is not sufficient, we
  // still return the out_algorithms. Caller needs to make sure that in this
  // case, the returned vector is empty.
  if (stream->GetCudaComputeCapability().IsAtLeast(
          CudaComputeCapability::AMPERE)) {
    // Note: for NVIDIA Ampere Architecture GPUs and beyond, i.e. SM version >=
    // 80, the numbered algorithm options are equivalent to CUBLAS_GEMM_DEFAULT
    // or CUBLAS_GEMM_DEFAULT_TENSOR_OP respectively.
    *out_algorithms = {
        CUBLAS_GEMM_DFALT,
        CUBLAS_GEMM_DFALT_TENSOR_OP,
    };
  } else {
    *out_algorithms = {
      CUBLAS_GEMM_DFALT,
      CUBLAS_GEMM_ALGO0,
      CUBLAS_GEMM_ALGO1,
      CUBLAS_GEMM_ALGO2,
      CUBLAS_GEMM_ALGO3,
      CUBLAS_GEMM_ALGO4,
      CUBLAS_GEMM_ALGO5,
      CUBLAS_GEMM_ALGO6,
      CUBLAS_GEMM_ALGO7,
#if CUDA_VERSION >= 9000
      CUBLAS_GEMM_ALGO8,
      CUBLAS_GEMM_ALGO9,
      CUBLAS_GEMM_ALGO10,
      CUBLAS_GEMM_ALGO11,
      CUBLAS_GEMM_ALGO12,
      CUBLAS_GEMM_ALGO13,
      CUBLAS_GEMM_ALGO14,
      CUBLAS_GEMM_ALGO15,
      CUBLAS_GEMM_ALGO16,
      CUBLAS_GEMM_ALGO17,
      CUBLAS_GEMM_DFALT_TENSOR_OP,
      CUBLAS_GEMM_ALGO0_TENSOR_OP,
      CUBLAS_GEMM_ALGO1_TENSOR_OP,
      CUBLAS_GEMM_ALGO2_TENSOR_OP,
      CUBLAS_GEMM_ALGO3_TENSOR_OP,
      CUBLAS_GEMM_ALGO4_TENSOR_OP,
#endif
#if CUDA_VERSION >= 9020
      CUBLAS_GEMM_ALGO18,
      CUBLAS_GEMM_ALGO19,
      CUBLAS_GEMM_ALGO20,
      CUBLAS_GEMM_ALGO21,
      CUBLAS_GEMM_ALGO22,
      CUBLAS_GEMM_ALGO23,
      CUBLAS_GEMM_ALGO5_TENSOR_OP,
      CUBLAS_GEMM_ALGO6_TENSOR_OP,
      CUBLAS_GEMM_ALGO7_TENSOR_OP,
      CUBLAS_GEMM_ALGO8_TENSOR_OP,
      CUBLAS_GEMM_ALGO9_TENSOR_OP,
      CUBLAS_GEMM_ALGO10_TENSOR_OP,
      CUBLAS_GEMM_ALGO11_TENSOR_OP,
      CUBLAS_GEMM_ALGO12_TENSOR_OP,
      CUBLAS_GEMM_ALGO13_TENSOR_OP,
      CUBLAS_GEMM_ALGO14_TENSOR_OP,
      CUBLAS_GEMM_ALGO15_TENSOR_OP,
#endif
    };
  }
  return true;
}

template <typename T>
struct HalfAsFloat {
  typedef T type;
};

template <>
struct HalfAsFloat<Eigen::half> {
  typedef float type;
};

namespace {
// pass-through for non-complex types that don't need conversion to
// cublas-specific type.
template <typename T>
T inline GpuComplexValue(T v) {
  return v;
}
}  // namespace

template <typename T, typename Scalar, typename FuncT>
port::Status CUDABlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64 n, uint64 k, Scalar alpha,
    const DeviceMemorySlice<T> &a_ptrs_to_wrappers, int lda,
    const DeviceMemorySlice<T> &b_ptrs_to_wrappers, int ldb, Scalar beta,
    const DeviceMemorySlice<T> &c_ptrs_to_wrappers, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename HalfAsFloat<typename GpuComplexT<T>::type>::type CUDA_T;

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
    TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> a_bytes,
                        scratch_allocator->AllocateBytes(size));
    TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> b_bytes,
                        scratch_allocator->AllocateBytes(size));
    TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> c_bytes,
                        scratch_allocator->AllocateBytes(size));
    a = DeviceMemory<CUDA_T *>(a_bytes);
    b = DeviceMemory<CUDA_T *>(b_bytes);
    c = DeviceMemory<CUDA_T *>(c_bytes);
  } else {
    TF_ASSIGN_OR_RETURN(a_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    TF_ASSIGN_OR_RETURN(b_temporary,
                        stream->AllocateTemporaryArray<CUDA_T *>(batch_count));
    TF_ASSIGN_OR_RETURN(c_temporary,
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

  cudaDataType_t data_type = CUDADataType<T>::type;

#if CUDA_VERSION >= 9010
  if (stream->GetCudaComputeCapability().IsAtLeast(5)) {
    cublasMath_t math_type;
    cublasGemmAlgo_t algo;
    if (data_type == CUDA_R_16F) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#else
      math_type = CUBLAS_DEFAULT_MATH;
#endif
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#if CUBLAS_VER_MAJOR >= 11
    } else if (data_type == CUDA_R_32F) {
      // DoBlassInternalImpl will switch math_type back to CUBLAS_DEFAULT_MATH
      // if TensorFloat-32 is disabled.
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
      algo = tensorflow::tensor_float_32_execution_enabled()
                 ? CUBLAS_GEMM_DFALT_TENSOR_OP
                 : CUBLAS_GEMM_DFALT;
#endif
    } else {
      math_type = CUBLAS_DEFAULT_MATH;
      algo = CUBLAS_GEMM_DFALT;
    }
    cudaDataType_t compute_type =
        (data_type == CUDA_R_16F ? CUDA_R_32F : data_type);
    const void **a_void_ptrs = reinterpret_cast<const void **>(
        const_cast<const CUDA_T **>(GpuMemory(a)));
    const void **b_void_ptrs = reinterpret_cast<const void **>(
        const_cast<const CUDA_T **>(GpuMemory(b)));
    void **c_void_ptrs =
        reinterpret_cast<void **>(const_cast<CUDA_T **>(GpuMemory(c)));
    return DoBlasInternalImpl(
        AS_LAMBDA(cublasGemmBatchedEx), stream, true /* = pointer_mode_host */,
        math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
        k, &alpha, a_void_ptrs, data_type, lda, b_void_ptrs, data_type, ldb,
        &beta, c_void_ptrs, data_type, ldc, batch_count, compute_type, algo);
  }
#endif
  // either CUDA_VERSION < 9.1 or SM < 5.0
  if (data_type != CUDA_R_16F) {
    auto cb_alpha = GpuComplexValue(alpha);
    auto cb_beta = GpuComplexValue(beta);
    bool ok = DoBlasInternal(
        cublas_func, stream, true /* = pointer_mode_host */,
        AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
        GpuComplex(&cb_alpha), const_cast<const CUDA_T **>(GpuMemory(a)), lda,
        const_cast<const CUDA_T **>(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
        const_cast<CUDA_T **>(GpuMemory(c)), ldc, batch_count);
    if (ok) {
      return ::tensorflow::OkStatus();
    }
    return port::Status(port::error::INTERNAL,
                        "failed BLAS call, see log for details");
  } else {
    // Fall back to a loop for fp16
    for (int b = 0; b < batch_count; ++b) {
      const DeviceMemory<T> &a_matrix = *a_ptrs_to_wrappers[b];
      const DeviceMemory<T> &b_matrix = *b_ptrs_to_wrappers[b];
      DeviceMemory<T> *c_matrix = c_ptrs_to_wrappers[b];
      TF_RETURN_IF_ERROR(DoBlasGemm(
          stream, transa, transb, m, n, k, blas::ToDataType<T>::value, &alpha,
          a_matrix, lda, b_matrix, ldb, &beta, c_matrix, ldc,
          blas::kDefaultComputePrecision));
    }
    return ::tensorflow::OkStatus();
  }
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    const DeviceMemorySlice<Eigen::half> &a_array, int lda,
    const DeviceMemorySlice<Eigen::half> &b_array, int ldb, float beta,
    const DeviceMemorySlice<Eigen::half> &c_array, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  // Note: The func passed here (cublasSgemmBatched) is not actually called,
  // due to special handling of fp16 inside DoBlasGemmBatchedInternal.
  port::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha, const DeviceMemorySlice<float> &a_array,
    int lda, const DeviceMemorySlice<float> &b_array, int ldb, float beta,
    const DeviceMemorySlice<float> &c_array, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, double alpha,
    const DeviceMemorySlice<double> &a_array, int lda,
    const DeviceMemorySlice<double> &b_array, int ldb, double beta,
    const DeviceMemorySlice<double> &c_array, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    const DeviceMemorySlice<std::complex<float>> &a_array, int lda,
    const DeviceMemorySlice<std::complex<float>> &b_array, int ldb,
    std::complex<float> beta,
    const DeviceMemorySlice<std::complex<float>> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    const DeviceMemorySlice<std::complex<double>> &a_array, int lda,
    const DeviceMemorySlice<std::complex<double>> &b_array, int ldb,
    std::complex<double> beta,
    const DeviceMemorySlice<std::complex<double>> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

port::Status CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputePrecision precision) {
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#if CUDA_VERSION < 11000
  if (dtype == dnn::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == dnn::kFloat) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
  }
  if (precision > blas::kDefaultComputePrecision) {
    math_type = CUBLAS_DEFAULT_MATH;
  }
#endif

  switch (dtype) {
#if CUDA_VERSION >= 11000
    case dnn::kBF16: {
      CudaComputeCapability cc = stream->GetCudaComputeCapability();
      if (cc.IsAtLeast(7)) {
        cublasGemmAlgo_t algo =
            (cc.major >= 7 ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
        return DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmStridedBatchedEx), stream,
            true /* = pointer_mode_host */, math_type,
            AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
            alpha, a.opaque(), CUDA_R_16BF, lda, stride_a, b.opaque(),
            CUDA_R_16BF, ldb, stride_b, beta, c->opaque(), CUDA_R_16BF, ldc,
            stride_c, batch_count,
            /*compute_type=*/CUDA_R_32F, algo);
      }
      // Fall back to a loop.
      for (int batch = 0; batch < batch_count; ++batch) {
        const auto *a_matrix = reinterpret_cast<const __nv_bfloat16 *>(
            static_cast<const Eigen::bfloat16 *>(a.opaque()) +
            batch * stride_a);
        const auto *b_matrix = reinterpret_cast<const __nv_bfloat16 *>(
            static_cast<const Eigen::bfloat16 *>(b.opaque()) +
            batch * stride_b);
        auto *c_matrix = reinterpret_cast<__nv_bfloat16 *>(
            static_cast<Eigen::bfloat16 *>(c->opaque()) + batch * stride_c);
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            cublasSgemmEx, stream, true /* = pointer_mode_host */,
            CUBLAS_DEFAULT_MATH, AsCublasOperation(transa),
            AsCublasOperation(transb), m, n, k,
            static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
            b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
            c_matrix, CUDA_R_16BF, ldc));
      }
      return port::Status::OK();
    }
#endif
    case dnn::kHalf: {
#if CUDA_VERSION >= 9010
      CudaComputeCapability cc = stream->GetCudaComputeCapability();
      if (cc.major >= 5) {
        cublasGemmAlgo_t algo =
            (cc.major >= 7 ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
        return DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmStridedBatchedEx), stream,
            true /* = pointer_mode_host */, math_type,
            AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
            alpha, a.opaque(), CUDA_R_16F, lda, stride_a, b.opaque(),
            CUDA_R_16F, ldb, stride_b, beta, c->opaque(), CUDA_R_16F, ldc,
            stride_c, batch_count, CUDA_R_32F, algo);
      }
#endif
      // Either CUDA_VERSION < 9.1 or SM < 5.0. Fall back to a loop.
      for (int batch = 0; batch < batch_count; ++batch) {
        const auto *a_matrix = reinterpret_cast<const __half *>(
            static_cast<const Eigen::half *>(a.opaque()) + batch * stride_a);
        const auto *b_matrix = reinterpret_cast<const __half *>(
            static_cast<const Eigen::half *>(b.opaque()) + batch * stride_b);
        auto *c_matrix = reinterpret_cast<__half *>(
            static_cast<Eigen::half *>(c->opaque()) + batch * stride_c);
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            cublasSgemmEx, stream, true /* = pointer_mode_host */,
            CUBLAS_DEFAULT_MATH, AsCublasOperation(transa),
            AsCublasOperation(transb), m, n, k,
            static_cast<const float *>(alpha), a_matrix, SE_CUDA_DATA_HALF, lda,
            b_matrix, SE_CUDA_DATA_HALF, ldb, static_cast<const float *>(beta),
            c_matrix, SE_CUDA_DATA_HALF, ldc));
      }
      return ::tensorflow::OkStatus();
    }
    case dnn::kFloat: {
      return DoBlasInternalImpl(
          cublasSgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda, stride_a,
          static_cast<const float *>(b.opaque()), ldb, stride_b,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc, stride_c, batch_count);
    }
    case dnn::kDouble:
      return DoBlasInternalImpl(
          cublasDgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda, stride_a,
          static_cast<const double *>(b.opaque()), ldb, stride_b,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc, stride_c, batch_count);
    case dnn::kComplexFloat: {
      GpuComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<float> *>(alpha));
      GpuComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, GpuComplex(&cb_alpha),
          static_cast<const GpuComplexType *>(a.opaque()), lda, stride_a,
          static_cast<const GpuComplexType *>(b.opaque()), ldb, stride_b,
          GpuComplex(&cb_beta), static_cast<GpuComplexType *>(c->opaque()), ldc,
          stride_c, batch_count);
    }
    case dnn::kComplexDouble: {
      GpuDoubleComplexType cb_alpha =
          GpuComplexValue(*static_cast<const std::complex<double> *>(alpha));
      GpuDoubleComplexType cb_beta =
          GpuComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, GpuComplex(&cb_alpha),
          static_cast<const GpuDoubleComplexType *>(a.opaque()), lda, stride_a,
          static_cast<const GpuDoubleComplexType *>(b.opaque()), ldb, stride_b,
          GpuComplex(&cb_beta),
          static_cast<GpuDoubleComplexType *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    default:
      return port::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(cublasStrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(cublasDtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(cublasStrsmBatched, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), AsCublasOperation(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(cublasDtrsmBatched, stream,
                        true /* = pointer_mode_host */, CUDABlasSide(side),
                        CUDABlasUpperLower(uplo), AsCublasOperation(transa),
                        CUDABlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(
      cublasCtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), AsCublasOperation(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<float2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<float2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(
      cublasZtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), AsCublasOperation(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<double2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<double2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

port::Status CUDABlas::GetVersion(std::string *version) {
  absl::MutexLock lock(&mu_);

  int v;
  auto status = cublasGetVersion(blas_, &v);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::InternalError(ToString(status));
  }
  *version = std::to_string(v);
  return ::tensorflow::OkStatus();
}

void initialize_cublas() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          kCudaPlatformId, kCuBlasPlugin, "cuBLAS",
          [](::stream_executor::internal::StreamExecutorInterface *parent)
              -> blas::BlasSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuBLAS "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            CUDABlas *blas = new CUDABlas(cuda_executor);
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

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kBlas, kCuBlasPlugin);
}

}  // namespace cuda
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { stream_executor::cuda::initialize_cublas(); });
