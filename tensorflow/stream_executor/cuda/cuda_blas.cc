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

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"

#define SE_CUDA_DATA_HALF CUDA_R_16F

#include "tensorflow/stream_executor/cuda/cuda_blas.h"

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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
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
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuBlasPlugin);

static std::string ToString(cublasStatus_t status) {
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
      return absl::StrCat("<invalid cublas status: ", status, ">");
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
  cublasHandle_t handle_;  // Handle to the cuBLAS instance of interest.
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

bool CUDABlas::Init() {
  gpu::ScopedActivateExecutorContext sac{parent_};
  cublasStatus_t ret = cublasCreate(&blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    return false;
  }

#if CUDA_VERSION >= 11000
  ret = cublasLtCreate(&blasLt_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublasLt handle: " << ToString(ret);
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
      blasLt_(nullptr)
#endif
{
}

CUDABlas::~CUDABlas() {
  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    cublasDestroy(blas_);
  }
#if CUDA_VERSION >= 11000
  if (blasLt_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    cublasLtDestroy(blasLt_);
  }
#endif
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
    case blas::ComputationType::kI32:
      return CUDA_R_32I;
    case blas::ComputationType::kComplexF32:
      return CUDA_C_32F;
    case blas::ComputationType::kComplexF64:
      return CUDA_C_64F;
    case blas::ComputationType::kTF32AsF32:  // fall-through
    case blas::ComputationType::kBF16AsF32:
      // These cases are currently only supported in the blasLt routines, which
      // use CUBLASComputationType() instead.
      LOG(FATAL) << "Invalid value of blas::ComputationType.";
  }
}

#if CUDA_VERSION >= 11000
cublasComputeType_t CUBLASComputationType(blas::ComputationType ty) {
  switch (ty) {
    case blas::ComputationType::kF16:
      return CUBLAS_COMPUTE_16F;
    case blas::ComputationType::kF32:  // fall-through
    case blas::ComputationType::kComplexF32:
      return CUBLAS_COMPUTE_32F;
    case blas::ComputationType::kF64:  // fall-through
    case blas::ComputationType::kComplexF64:
      return CUBLAS_COMPUTE_64F;
    case blas::ComputationType::kI32:
      return CUBLAS_COMPUTE_32I;
    case blas::ComputationType::kTF32AsF32:
      return CUBLAS_COMPUTE_32F_FAST_TF32;
    case blas::ComputationType::kBF16AsF32:
      return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
}
#endif  // CUDA_VERSION >= 11000

blas::DataType GetScaleType(blas::DataType data_type,
                            blas::ComputationType compute_type) {
  bool is_complex = data_type == blas::DataType::kComplexFloat ||
                    data_type == blas::DataType::kComplexDouble;
  switch (compute_type) {
    case blas::ComputationType::kF16:
      return blas::DataType::kHalf;
    case blas::ComputationType::kF32:         // fall-through
    case blas::ComputationType::kComplexF32:  // fall-through
    case blas::ComputationType::kTF32AsF32:   // fall-through
    case blas::ComputationType::kBF16AsF32:
      return is_complex ? blas::DataType::kComplexFloat
                        : blas::DataType::kFloat;
    case blas::ComputationType::kF64:  // fall-through
    case blas::ComputationType::kComplexF64:
      return is_complex ? blas::DataType::kComplexDouble
                        : blas::DataType::kDouble;
    case blas::ComputationType::kI32:
      return blas::DataType::kInt32;
  }
}

#if CUDA_VERSION >= 11000
cublasLtPointerMode_t CUBLASPointerMode(blas::PointerMode pointer_mode) {
  switch (pointer_mode) {
    case blas::PointerMode::kHost:
      return CUBLASLT_POINTER_MODE_HOST;
    case blas::PointerMode::kDevice:
      return CUBLASLT_POINTER_MODE_DEVICE;
  }
}
cublasLtEpilogue_t CUBLASEpilogue(blas::Epilogue epilogue) {
  switch (epilogue) {
    case blas::Epilogue::kDefault:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case blas::Epilogue::kReLU:
      return CUBLASLT_EPILOGUE_RELU;
    case blas::Epilogue::kBias:
      return CUBLASLT_EPILOGUE_BIAS;
    case blas::Epilogue::kBiasThenReLU:
      return CUBLASLT_EPILOGUE_RELU_BIAS;
  }
}
#endif  // CUDA_VERSION >= 11000

cudaDataType_t GetCUDADataType(blas::DataType ty) {
  switch (ty) {
    case blas::DataType::kHalf:
      return CUDA_R_16F;
    case blas::DataType::kFloat:
      return CUDA_R_32F;
    case blas::DataType::kDouble:
      return CUDA_R_64F;
    case blas::DataType::kInt8:
      return CUDA_R_8I;
    case blas::DataType::kInt32:
      return CUDA_R_32I;
    case blas::DataType::kComplexFloat:
      return CUDA_C_32F;
    case blas::DataType::kComplexDouble:
      return CUDA_C_64F;
    default:
      LOG(FATAL) << "Invalid value of blas::DataType in GetCUDADataType";
  }
}

int GetDataTypeSizeBytes(blas::DataType ty) {
  switch (ty) {
    case blas::DataType::kHalf:
      return 2;
    case blas::DataType::kFloat:
      return 4;
    case blas::DataType::kDouble:
      return 8;
    case blas::DataType::kInt8:
      return 1;
    case blas::DataType::kInt32:
      return 4;
    case blas::DataType::kComplexFloat:
      return 8;
    case blas::DataType::kComplexDouble:
      return 16;
    default:
      LOG(FATAL) << "Invalid value of blas::DataType in GetDataTypeSizeBytes";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool CUDABlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  cublasMath_t math_type, Args... args) {
  absl::MutexLock lock(&mu_);

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
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
      return false;
    }
  }
#endif

  gpu::ScopedActivateExecutorContext sac{parent_};
  ScopedCublasPointerMode pointer_mode{blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }
  cublasStatus_t ret = cublas_func(blas_, args...);
  if ((err_on_failure || VLOG_IS_ON(3)) && ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run cuBLAS routine: " << ToString(ret);
  }
  return ret == CUBLAS_STATUS_SUCCESS;
}

// cublas_func may be overloaded, so we need to figure out which one we really
// need to call based on the args. One way to do it is to wrap it in lambda.
#define AS_LAMBDA(func)                                                  \
  [](auto &&... args) -> decltype(                                       \
                          func(std::forward<decltype(args)>(args)...)) { \
    return func(std::forward<decltype(args)>(args)...);                  \
  }

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(cublasSasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(cublasDasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(cublasScasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(cublasDzasum, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZaxpy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasScopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(cublasCcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(cublasZcopy, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(cublasSdot, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(cublasDdot, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(cublasCdotc, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(cublasZdotc, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return DoBlasInternal(cublasCdotu, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return DoBlasInternal(cublasZdotu, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(result)));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(cublasSnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(cublasDnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(cublasScnrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(cublasDznrm2, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return DoBlasInternal(cublasSrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return DoBlasInternal(cublasDrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return DoBlasInternal(cublasCsrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return DoBlasInternal(cublasZdrot, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy, &c, &s);
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return DoBlasInternal(cublasSrotg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(a), GpuMemoryMutable(b),
                        GpuMemoryMutable(c), GpuMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return DoBlasInternal(cublasDrotg, stream, false /* = pointer_mode_host */,
                        GpuComplex(GpuMemoryMutable(a)), GpuMemoryMutable(b),
                        GpuMemoryMutable(c), GpuMemoryMutable(s));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return DoBlasInternal(
      cublasCrotg, stream, false /* = pointer_mode_host */,
      GpuComplex(GpuMemoryMutable(a)), GpuComplex(GpuMemoryMutable(b)),
      GpuComplex(GpuMemoryMutable(c)), GpuComplex(GpuMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return DoBlasInternal(
      cublasZrotg, stream, false /* = pointer_mode_host */,
      GpuComplex(GpuMemoryMutable(a)), GpuComplex(GpuMemoryMutable(b)),
      GpuComplex(GpuMemoryMutable(c)), GpuComplex(GpuMemoryMutable(s)));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return DoBlasInternal(cublasSrotm, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool CUDABlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return DoBlasInternal(cublasDrotm, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy, GpuMemory(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return DoBlasInternal(cublasSrotmg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(d1), GpuMemoryMutable(d2),
                        GpuMemoryMutable(x1), GpuMemory(y1),
                        GpuMemoryMutable(param));
}

bool CUDABlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return DoBlasInternal(cublasDrotmg, stream, false /* = pointer_mode_host */,
                        GpuMemoryMutable(d1), GpuMemoryMutable(d2),
                        GpuMemoryMutable(x1), GpuMemory(y1),
                        GpuMemoryMutable(param));
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasSscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCsscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZdscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, GpuComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZscal, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuMemoryMutable(x), incx,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(cublasCswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(cublasZswap, stream, true /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemoryMutable(x)), incx,
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIsamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIdamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuMemory(x), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIcamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIzamax, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIsamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIdamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIcamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return DoBlasInternal(cublasIzamin, stream, false /* = pointer_mode_host */,
                        elem_count, GpuComplex(GpuMemory(x)), incx,
                        GpuMemoryMutable(result));
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku, &alpha,
                        GpuMemory(a), lda, GpuMemory(x), incx, &beta,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku, &alpha,
                        GpuMemory(a), lda, GpuMemory(x), incx, &beta,
                        GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(x)), incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, kl, ku,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(x)), incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgemv, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(trans), m, n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(cublasSger, stream, true /* = pointer_mode_host */, m,
                        n, &alpha, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(cublasDger, stream, true /* = pointer_mode_host */, m,
                        n, &alpha, GpuMemory(x), incx, GpuMemory(y), incy,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCgerc, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZgerc, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCgeru, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZgeru, stream, true /* = pointer_mode_host */, m,
                        n, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChemv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhemv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(a)), lda, GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return DoBlasInternal(cublasCher, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return DoBlasInternal(cublasZher, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCher2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZher2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(a)), lda);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(ap)), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(ap)), GpuComplex(GpuMemory(x)),
                        incx, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return DoBlasInternal(cublasChpr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return DoBlasInternal(cublasZhpr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha,
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasChpr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZhpr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, GpuComplex(&cb_alpha),
                        GpuComplex(GpuMemory(x)), incx,
                        GpuComplex(GpuMemory(y)), incy,
                        GpuComplex(GpuMemoryMutable(ap)));
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDsbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, k, &alpha, GpuMemory(a),
                        lda, GpuMemory(x), incx, &beta, GpuMemoryMutable(y),
                        incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSspmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDspmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(ap),
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return DoBlasInternal(cublasSspr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return DoBlasInternal(cublasDspr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return DoBlasInternal(cublasSspr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return DoBlasInternal(cublasDspr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(ap));
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(cublasSsymv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(cublasDsymv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(a), lda,
                        GpuMemory(x), incx, &beta, GpuMemoryMutable(y), incy);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(cublasSsyr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(cublasDsyr, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(cublasSsyr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(cublasDsyr2, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), n, &alpha, GpuMemory(x), incx,
                        GpuMemory(y), incy, GpuMemoryMutable(a), lda);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasStbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(cublasCtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(cublasZtbmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasStbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return DoBlasInternal(cublasCtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return DoBlasInternal(cublasZtbsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, k, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(cublasStpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZtpmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return DoBlasInternal(cublasStpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(ap),
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZtpsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(ap)),
                        GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasStrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZtrmv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(cublasStrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(cublasDtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuMemory(a), lda,
                        GpuMemoryMutable(x), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(cublasCtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZtrsv, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans),
                        CUDABlasDiagonal(diag), n, GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if CUDA_VERSION >= 7050
  VLOG(1) << absl::StrFormat(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%u n=%u "
      "k=%u alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
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

#if CUDA_VERSION < 11000
  cublasMath_t math_type = CUBLAS_TENSOR_OP_MATH;
#else
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#endif

  return DoBlasInternalImpl(
      cublasSgemmEx, stream, true /* = pointer_mode_host */,
      true /* = err_on_failure= */, math_type, CUDABlasTranspose(transa),
      CUDABlasTranspose(transb), m, n, k, &alpha, GpuMemory(a),
      SE_CUDA_DATA_HALF, lda, GpuMemory(b), SE_CUDA_DATA_HALF, ldb, &beta,
      GpuMemoryMutable(c), SE_CUDA_DATA_HALF, ldc);

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
  VLOG(1) << absl::StrFormat(
      "doing cuBLAS SGEMM: at=%d bt=%d m=%u n=%u "
      "k=%u alpha=%f a=%p lda=%d b=%p ldb=%d beta=%f "
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

#if CUDA_VERSION < 11000
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#else
  cublasMath_t math_type = CUBLAS_TF32_TENSOR_OP_MATH;
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().cuda_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major >= 8) {
    // TODO(reedwm): Remove or make this VLOG(1) once TensorFloat-32 is more
    // well tested.
    LOG_FIRST_N(INFO, 1) << "TensorFloat-32 will be used for the matrix "
                            "multiplication. This will only be logged once.";
  }
#endif

  return DoBlasInternalImpl(
      cublasSgemm, stream, true /* = pointer_mode_host */,
      true /* = err_on_failure */, math_type, CUDABlasTranspose(transa),
      CUDABlasTranspose(transb), m, n, k, &alpha, GpuMemory(a), lda,
      GpuMemory(b), ldb, &beta, GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(cublasDgemm, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m,
                        n, k, &alpha, GpuMemory(a), lda, GpuMemory(b), ldb,
                        &beta, GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCgemm, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m,
                        n, k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemory(b)), ldb,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZgemm, stream, true /* = pointer_mode_host */,
                        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m,
                        n, k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)),
                        lda, GpuComplex(GpuMemory(b)), ldb,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool CUDABlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
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
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, const T &alpha,
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
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
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
  bool result = DoBlasGemm(stream, transa, transb, m, n, k, alpha, a, lda, b,
                           ldb, beta, c, ldc);

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

template <typename InT, typename OutT, typename CompT>
bool CUDABlas::DoBlasGemmWithAlgorithmImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<CompT> &alpha,
    const DeviceMemory<InT> &a, int lda, const DeviceMemory<InT> &b, int ldb,
    const HostOrDeviceScalar<CompT> &beta, DeviceMemory<OutT> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  // GPUs < sm_50 don't support cublasGemmEx.
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().cuda_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major < 5) {
    VLOG(2) << "DoBlasGemmWithAlgorithm returning false because sm" << cc_major
            << cc_minor << " devices don't support explicit gemm algorithms.";
    return false;
  }

  bool algo_uses_tensor_ops = UsesTensorOps(algorithm);
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
  if (algo_uses_tensor_ops) {
    if (cc_major < 7) {
      VLOG(2) << "DoBlasGemmWithAlgorithm returning false because algorithm "
              << algorithm
              << " uses tensor ops, but tensor ops are not available in sm"
              << cc_major << "X devices.";
      return false;
    } else if (std::is_same<InT, float>::value) {
#if CUDA_VERSION < 11000
      VLOG(2) << "DoBlasGemmWithAlgorithm returning false because algorithm "
              << algorithm
              << " uses tensor ops, but tensor ops are not available for fp32"
              << " inputs.";
      return false;
#else
      if (cc_major < 8) {
        VLOG(2) << "DoBlasGemmWithAlgorithm returning false because algorithm "
                << algorithm
                << " uses tensor ops, but tensor ops are not available in sm"
                << cc_major << "X devices for float input types.";
        return false;
      } else if (!tensorflow::tensor_float_32_execution_enabled()) {
        VLOG(2) << "DoBlasGemmWithAlgorithm returning false because algorithm "
                << algorithm
                << " uses tensor ops, but tensor ops are disabled for fp32"
                << " inputs.";
        return false;
      }
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
#endif
    } else if (std::is_same<InT, Eigen::half>::value) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#endif
    } else {
      VLOG(2) << "DoBlasGemmWithAlgorithm returning false because algorithm "
              << algorithm
              << " uses tensor ops, which are not supported for InT.";
      return false;
    }
  }

  // Either both 'alpha' and 'beta' need to be pointers to device memory, or
  // they need to be both host scalars.
  if (alpha.is_pointer() != beta.is_pointer()) {
    VLOG(2) << "DoBlasGemmWithAlgorithm returning false because one of `alpha` "
               "and `beta` is a pointer, but the other is not.";
    return false;
  }

  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      VLOG(2) << "DoBlasGemmWithAlgorithm returning false because "
                 "output_profile_result was given, but we were unable to "
                 "create a GpuTimer.";
      return false;
    }
  }

  // Return false if we might be hitting a cuBLAS bug that produces the wrong
  // result. See nvbugs/2156201, b/79126339.
#if CUDA_VERSION >= 9000 && CUDA_VERSION < 9020
  if ((algorithm == CUBLAS_GEMM_DEFAULT || algorithm >= CUBLAS_GEMM_ALGO13) &&
      std::max({m, n, k}) >= 2097153 && cc_major < 7) {
    VLOG(2) << "DoBlasGemmWithAlgorithm returning false to work around cudnn "
               "<9.2 bug with m, n, or k >= 2097153.  See b/79126339.";
    return false;
  }
#endif

  cudaDataType_t cuda_in_type = CUDADataType<InT>::type;
  // Since we are converting 'algorithm' to cublasGemmAlgo_t by static_cast,
  // we do the following compile-time check on the default value:
  static_assert(blas::kDefaultGemmAlgo == CUBLAS_GEMM_DFALT, "");
  // If 'alpha' and 'beta' are host scalars and CompT is Eigen::half, we
  // essentially reinterpet_cast to __half, which is safe because Eigen::half
  // inherits from __half.
  bool result = DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmEx), stream,
      /* pointer_mode_host = */ !alpha.is_pointer(), /*err_on_failure=*/false,
      math_type, CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      alpha.is_pointer() ? GpuMemory(alpha.pointer()) : &alpha.value(),
      GpuMemory(a), cuda_in_type, lda, GpuMemory(b), cuda_in_type, ldb,
      beta.is_pointer() ? GpuMemory(beta.pointer()) : &beta.value(),
      GpuMemoryMutable(c), CUDADataType<OutT>::type, ldc,
      CUDAComputationType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm));

  if (timer != nullptr && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      VLOG(2) << "DoBlasGemmWithAlgorithm returning false; unable to stop "
                 "GpuTimer.";
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

bool CUDABlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
  // cublasGemmAlgo_t (and the function that accepts this type, cublasGemmEx)
  // were first introduced in CUDA 8.
  //
  // Note that when CUDA version and compute capability is not sufficient, we
  // still return the out_algorithms. Caller needs to make sure that in this
  // case, the returned vector is empty.
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
  return true;
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<int> &alpha,
    const DeviceMemory<int8> &a, int lda, const DeviceMemory<int8> &b, int ldb,
    const HostOrDeviceScalar<int> &beta, DeviceMemory<int> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<Eigen::half> &alpha,
    const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb,
    const HostOrDeviceScalar<Eigen::half> &beta, DeviceMemory<Eigen::half> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  if (computation_type == blas::ComputationType::kF32) {
    if (alpha.is_pointer() || beta.is_pointer()) {
      // We cannot easily convert a pointer to f16 memory to a pointer to f32
      // memory from here, so we don't support this for now.
      // TODO(akuegel): Investigate whether we can do the conversion before
      // calling DoBlasGemmWithAlgorithm.
      return false;
    }
    HostOrDeviceScalar<float> float_alpha(static_cast<float>(alpha.value()));
    HostOrDeviceScalar<float> float_beta(static_cast<float>(beta.value()));
    return DoBlasGemmWithAlgorithmImpl(
        stream, transa, transb, m, n, k, float_alpha, a, lda, b, ldb,
        float_beta, c, ldc, computation_type, algorithm, output_profile_result);
  }

  CHECK_EQ(computation_type, blas::ComputationType::kF16);
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<float> &alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &b,
    int ldb, const HostOrDeviceScalar<float> &beta, DeviceMemory<float> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<double> &alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &b,
    int ldb, const HostOrDeviceScalar<double> &beta, DeviceMemory<double> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<std::complex<float>> &alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    const HostOrDeviceScalar<std::complex<float>> &beta,
    DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
}

bool CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<std::complex<double>> &alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    const HostOrDeviceScalar<std::complex<double>> &beta,
    DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithAlgorithmImpl(
      stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
      computation_type, algorithm, output_profile_result);
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
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, Scalar alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    Scalar beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
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
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> a_bytes,
                        scratch_allocator->AllocateBytes(size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> b_bytes,
                        scratch_allocator->AllocateBytes(size));
    SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> c_bytes,
                        scratch_allocator->AllocateBytes(size));
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

  cudaDataType_t data_type = CUDADataType<T>::type;

#if CUDA_VERSION >= 9010
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().cuda_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major >= 5) {
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
    bool ok;
    ok = DoBlasInternalImpl(
        AS_LAMBDA(cublasGemmBatchedEx), stream, true /* = pointer_mode_host */,
        true /* = err_on_failure */, math_type, CUDABlasTranspose(transa),
        CUDABlasTranspose(transb), m, n, k, &alpha, a_void_ptrs, data_type, lda,
        b_void_ptrs, data_type, ldb, &beta, c_void_ptrs, data_type, ldc,
        batch_count, compute_type, algo);
    if (ok) {
      return port::Status::OK();
    }
    return port::Status(port::error::INTERNAL,
                        "failed BLAS call, see log for details");
  }
#endif
  // either CUDA_VERSION < 9.1 or SM < 5.0
  if (data_type != CUDA_R_16F) {
    auto cb_alpha = GpuComplexValue(alpha);
    auto cb_beta = GpuComplexValue(beta);
    bool ok = DoBlasInternal(
        cublas_func, stream, true /* = pointer_mode_host */,
        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
        GpuComplex(&cb_alpha), const_cast<const CUDA_T **>(GpuMemory(a)), lda,
        const_cast<const CUDA_T **>(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
        const_cast<CUDA_T **>(GpuMemory(c)), ldc, batch_count);
    if (ok) {
      return port::Status::OK();
    }
    return port::Status(port::error::INTERNAL,
                        "failed BLAS call, see log for details");
  } else {
    // Fall back to a loop for fp16
    for (int b = 0; b < batch_count; ++b) {
      const DeviceMemory<T> &a_matrix = *a_ptrs_to_wrappers[b];
      const DeviceMemory<T> &b_matrix = *b_ptrs_to_wrappers[b];
      DeviceMemory<T> *c_matrix = c_ptrs_to_wrappers[b];
      bool ok = DoBlasGemm(stream, transa, transb, m, n, k, alpha, a_matrix,
                           lda, b_matrix, ldb, beta, c_matrix, ldc);
      if (!ok) {
        return port::Status(port::error::INTERNAL,
                            "failed BLAS call, see log for details");
      }
    }
    return port::Status::OK();
  }
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b_array, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<Eigen::half> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
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
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
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
      cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
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
      cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
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
      cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, int64 stride_a, const DeviceMemory<Eigen::half> &b, int ldb,
    int64 stride_b, float beta, DeviceMemory<Eigen::half> *c, int ldc,
    int64 stride_c, int batch_count) {
#if CUDA_VERSION >= 9010
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().cuda_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major >= 5) {
    cublasGemmAlgo_t algo =
        (cc_major >= 7 ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
#if CUDA_VERSION < 11000
    cublasMath_t math_type = CUBLAS_TENSOR_OP_MATH;
#else
    cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#endif
    bool ok = DoBlasInternalImpl(
        AS_LAMBDA(cublasGemmStridedBatchedEx), stream,
        true /* = pointer_mode_host */, true /* = err_on_failure */, math_type,
        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
        GpuMemory(a), CUDA_R_16F, lda, stride_a, GpuMemory(b), CUDA_R_16F, ldb,
        stride_b, &beta, GpuMemoryMutable(c), CUDA_R_16F, ldc, stride_c,
        batch_count, CUDA_R_32F, algo);
    if (ok) {
      return true;
    }
    LOG(ERROR) << "failed BLAS call, see log for details";
    return false;
  }
#endif
  // Either CUDA_VERSION < 9.1 or SM < 5.0. Fall back to a loop.
  for (int batch = 0; batch < batch_count; ++batch) {
    const auto *a_matrix =
        reinterpret_cast<const __half *>(GpuMemory(a) + batch * stride_a);
    const auto *b_matrix =
        reinterpret_cast<const __half *>(GpuMemory(b) + batch * stride_b);
    auto *c_matrix =
        reinterpret_cast<__half *>(GpuMemoryMutable(c) + batch * stride_c);
    bool ok = DoBlasInternalImpl(
        cublasSgemmEx, stream, true /* = pointer_mode_host */,
        true /* = err_on_failure= */, CUBLAS_DEFAULT_MATH,
        CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
        a_matrix, SE_CUDA_DATA_HALF, lda, b_matrix, SE_CUDA_DATA_HALF, ldb,
        &beta, c_matrix, SE_CUDA_DATA_HALF, ldc);
    if (!ok) {
      LOG(ERROR) << "failed BLAS call, see log for details";
      return false;
    }
  }
  return true;
}

bool CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    int64 stride_a, const DeviceMemory<float> &b, int ldb, int64 stride_b,
    float beta, DeviceMemory<float> *c, int ldc, int64 stride_c,
    int batch_count) {
#if CUDA_VERSION < 11000
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#else
  cublasMath_t math_type = CUBLAS_TF32_TENSOR_OP_MATH;
#endif
  return DoBlasInternalImpl(
      cublasSgemmStridedBatched, stream, true /* = pointer_mode_host */,
      true /* = err_on_failure */, math_type, CUDABlasTranspose(transa),
      CUDABlasTranspose(transb), m, n, k, &alpha, GpuMemory(a), lda, stride_a,
      GpuMemory(b), ldb, stride_b, &beta, GpuMemoryMutable(c), ldc, stride_c,
      batch_count);
}

bool CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    int64 stride_a, const DeviceMemory<double> &b, int ldb, int64 stride_b,
    double beta, DeviceMemory<double> *c, int ldc, int64 stride_c,
    int batch_count) {
  return DoBlasInternal(
      cublasDgemmStridedBatched, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k, &alpha,
      GpuMemory(a), lda, stride_a, GpuMemory(b), ldb, stride_b, &beta,
      GpuMemoryMutable(c), ldc, stride_c, batch_count);
}

bool CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda, int64 stride_a,
    const DeviceMemory<std::complex<float>> &b, int ldb, int64 stride_b,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    int64 stride_c, int batch_count) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(
      cublasCgemmStridedBatched, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda, stride_a,
      GpuComplex(GpuMemory(b)), ldb, stride_b, GpuComplex(&cb_beta),
      GpuComplex(GpuMemoryMutable(c)), ldc, stride_c, batch_count);
}

bool CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda, int64 stride_a,
    const DeviceMemory<std::complex<double>> &b, int ldb, int64 stride_b,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    int64 stride_c, int batch_count) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(
      cublasZgemmStridedBatched, stream, true /* = pointer_mode_host */,
      CUDABlasTranspose(transa), CUDABlasTranspose(transb), m, n, k,
      GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda, stride_a,
      GpuComplex(GpuMemory(b)), ldb, stride_b, GpuComplex(&cb_beta),
      GpuComplex(GpuMemoryMutable(c)), ldc, stride_c, batch_count);
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasChemm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZhemm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return DoBlasInternal(cublasCherk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuComplex(GpuMemory(a)), lda, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return DoBlasInternal(cublasZherk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuComplex(GpuMemory(a)), lda, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCher2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZher2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, &beta,
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(cublasSsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(cublasDsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsymm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(cublasSsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(cublasDsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsyrk, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(&cb_beta), GpuComplex(GpuMemoryMutable(c)),
                        ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return DoBlasInternal(cublasSsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(cublasDsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, &alpha, GpuMemory(a), lda, GpuMemory(b), ldb, &beta,
                        GpuMemoryMutable(c), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasCsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  auto cb_alpha = GpuComplexValue(alpha);
  auto cb_beta = GpuComplexValue(beta);
  return DoBlasInternal(cublasZsyr2k, stream, true /* = pointer_mode_host */,
                        CUDABlasUpperLower(uplo), CUDABlasTranspose(trans), n,
                        k, GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemory(b)), ldb, GpuComplex(&cb_beta),
                        GpuComplex(GpuMemoryMutable(c)), ldc);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(cublasStrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb,
                        GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(cublasDtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb,
                        GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZtrmm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(cublasStrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(cublasDtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasCtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  auto cb_alpha = GpuComplexValue(alpha);
  return DoBlasInternal(cublasZtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        CUDABlasTranspose(transa), CUDABlasDiagonal(diag), m, n,
                        GpuComplex(&cb_alpha), GpuComplex(GpuMemory(a)), lda,
                        GpuComplex(GpuMemoryMutable(b)), ldb);
}

// We only use cublasLt from CUDA 11.0 onward.
#if CUDA_VERSION >= 11000

namespace {

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatrixLayout_t handle,
                                    cublasLtMatrixLayoutAttribute_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatrixLayoutSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulAlgo_t *handle,
                                    cublasLtMatmulAlgoConfigAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulAlgoConfigSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulAlgoConfigSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulPreference_t handle,
                                    cublasLtMatmulPreferenceAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulPreferenceSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulPreferenceSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline bool GetCublasLtAttr(const cublasLtMatmulAlgo_t *handle,
                            cublasLtMatmulAlgoConfigAttributes_t attr,
                            T *value) {
  auto mutable_handle = const_cast<cublasLtMatmulAlgo_t *>(handle);
  size_t bytes_written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(mutable_handle, attr, value,
                                              sizeof(T), &bytes_written) ==
             CUBLAS_STATUS_SUCCESS &&
         bytes_written == sizeof(T);
}

template <typename T>
inline const T &ValueForStrCat(const T &value) {
  return value;
}
template <typename T>
inline absl::Hex ValueForStrCat(T *ptr) {
  return absl::Hex(reinterpret_cast<uintptr_t>(ptr));
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulDesc_t handle,
                                    cublasLtMatmulDescAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulDescSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescSetAttribute(attr=", attr, ", value=",
                     ValueForStrCat(value), ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

struct MatmulDescDestroyer {
  void operator()(cublasLtMatmulDesc_t matmul_desc) const {
    cublasLtMatmulDescDestroy(matmul_desc);
  }
};
struct LayoutDestroyer {
  void operator()(cublasLtMatrixLayout_t layout) const {
    cublasLtMatrixLayoutDestroy(layout);
  }
};
struct MatmulPreferenceDestroyer {
  void operator()(cublasLtMatmulPreference_t matmul_pref) const {
    cublasLtMatmulPreferenceDestroy(matmul_pref);
  }
};
using UniqueOpDesc =
    std::unique_ptr<std::remove_pointer<cublasLtMatmulDesc_t>::type,
                    MatmulDescDestroyer>;
using UniqueLayoutDesc =
    std::unique_ptr<std::remove_pointer<cublasLtMatrixLayout_t>::type,
                    LayoutDestroyer>;
using UniqueMatmulPreference =
    std::unique_ptr<std::remove_pointer<cublasLtMatmulPreference_t>::type,
                    MatmulPreferenceDestroyer>;

port::StatusOr<UniqueOpDesc> CreateCublasLtOperationDesc(
    blas::ComputationType computation_type, blas::DataType scale_type,
    blas::PointerMode pointer_mode, blas::Epilogue epilogue,
    blas::Transpose transa, blas::Transpose transb) {
  cublasLtMatmulDesc_t desc;
  cublasComputeType_t cublas_compute_type =
      CUBLASComputationType(computation_type);
  cudaDataType_t cuda_scale_type = GetCUDADataType(scale_type);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&desc, cublas_compute_type, cuda_scale_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescCreate(computation_type=",
                     computation_type, ") failed: ", ToString(status)));
  }
  UniqueOpDesc unique_desc(desc);
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                     CUBLASPointerMode(pointer_mode)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     CUBLASEpilogue(epilogue)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     CUDABlasTranspose(transa)));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     CUDABlasTranspose(transb)));
  return unique_desc;
}

port::StatusOr<UniqueLayoutDesc> CreateCublasLtLayoutDesc(
    blas::DataType data_type, uint64 rows, uint64 cols, int64 ld, int64 stride,
    int batch_count) {
  cublasLtMatrixLayout_t desc;
  cublasStatus_t status = cublasLtMatrixLayoutCreate(
      &desc, GetCUDADataType(data_type), rows, cols, ld);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutCreate failed: ", ToString(status)));
  }
  UniqueLayoutDesc unique_desc(desc);
  SE_RETURN_IF_ERROR(
      SetCublasLtAttr(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_count));
  SE_RETURN_IF_ERROR(SetCublasLtAttr(
      desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride));
  return unique_desc;
}

// Helper function to allocate workspace.
port::Status AllocateWorkspace(void **workspace,
                               ScratchAllocator *scratch_allocator,
                               size_t num_bytes) {
  SE_ASSIGN_OR_RETURN(DeviceMemory<uint8> workspace_bytes,
                      scratch_allocator->AllocateBytes(num_bytes));
  *workspace = (void *)GpuMemoryMutable(&workspace_bytes);
  return port::Status::OK();
}

template <typename T>
blas::ComputationType ToComputationType();
template <>
blas::ComputationType ToComputationType<Eigen::half>() {
  return blas::ComputationType::kF16;
}
template <>
blas::ComputationType ToComputationType<float>() {
  return blas::ComputationType::kF32;
}
template <>
blas::ComputationType ToComputationType<double>() {
  return blas::ComputationType::kF64;
}
template <>
blas::ComputationType ToComputationType<std::complex<float>>() {
  return blas::ComputationType::kComplexF32;
}
template <>
blas::ComputationType ToComputationType<std::complex<double>>() {
  return blas::ComputationType::kComplexF64;
}

class CUDABlasLtMatmulPlan final : public blas::IBlasLtMatmulPlan {
 public:
  port::Status init(const blas::BlasLtMatmulPlanParams &p) {
    params_ = p;
    scale_type_ = GetScaleType(p.c_type, p.computation_type);
    SE_ASSIGN_OR_RETURN(
        op_desc_,
        CreateCublasLtOperationDesc(
            p.computation_type, GetScaleType(p.c_type, p.computation_type),
            p.pointer_mode, p.epilogue, p.transa, p.transb));
    uint64 rows_a = p.transa == blas::Transpose::kNoTranspose ? p.m : p.k;
    uint64 cols_a = p.transa == blas::Transpose::kNoTranspose ? p.k : p.m;
    uint64 rows_b = p.transb == blas::Transpose::kNoTranspose ? p.k : p.n;
    uint64 cols_b = p.transb == blas::Transpose::kNoTranspose ? p.n : p.k;
    SE_ASSIGN_OR_RETURN(
        a_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda,
                                          p.stride_a, capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        b_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb,
                                          p.stride_b, capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        c_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                          capped_batch_count()));
    SE_ASSIGN_OR_RETURN(
        d_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                          capped_batch_count()));
    remainder_batch_count_ =
        p.batch_count > kMaxBatchCount ? p.batch_count % kMaxBatchCount : 0;
    if (remainder_batch_count_) {
      SE_ASSIGN_OR_RETURN(
          a_remainder_desc_,
          CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda, p.stride_a,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          b_remainder_desc_,
          CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb, p.stride_b,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          c_remainder_desc_,
          CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                   remainder_batch_count_));
      SE_ASSIGN_OR_RETURN(
          d_remainder_desc_,
          CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                   remainder_batch_count_));
    }
    return port::Status::OK();
  }

  cublasLtMatmulDesc_t op_desc() const { return op_desc_.get(); }
  cublasLtMatrixLayout_t a_desc() const { return a_desc_.get(); }
  cublasLtMatrixLayout_t b_desc() const { return b_desc_.get(); }
  cublasLtMatrixLayout_t c_desc() const { return c_desc_.get(); }
  cublasLtMatrixLayout_t d_desc() const { return d_desc_.get(); }
  cublasLtMatrixLayout_t a_remainder_desc() const {
    return a_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t b_remainder_desc() const {
    return b_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t c_remainder_desc() const {
    return c_remainder_desc_.get();
  }
  cublasLtMatrixLayout_t d_remainder_desc() const {
    return d_remainder_desc_.get();
  }

  const blas::BlasLtMatmulPlanParams &params() const { return params_; }
  blas::DataType scale_type() const { return scale_type_; }
  blas::DataType ab_type() const override { return params_.ab_type; }
  blas::DataType c_type() const override { return params_.c_type; }
  int capped_batch_count() const {
    return std::min(params_.batch_count, kMaxBatchCount);
  }
  int remainder_batch_count() const { return remainder_batch_count_; }

  // Note: Must be const to satisfy API. This is always called before the plan
  // is executed, so the state change is not observed in subsequent executions.
  bool SetBiasPointer(const void *bias) const;

 private:
  // In some cases cublasLt does not support large batch sizes, so we need to
  // split up such cases into multiple calls.
  static constexpr int kMaxBatchCount = 65535;
  blas::BlasLtMatmulPlanParams params_;
  blas::DataType scale_type_;
  UniqueOpDesc op_desc_;
  // These have batch count set to capped_batch_count().
  UniqueLayoutDesc a_desc_;
  UniqueLayoutDesc b_desc_;
  UniqueLayoutDesc c_desc_;
  UniqueLayoutDesc d_desc_;
  int remainder_batch_count_;
  // These have batch count set to remainder_batch_count_, and are only created
  // if params_.batch_count > kMaxBatchSize.
  UniqueLayoutDesc a_remainder_desc_;
  UniqueLayoutDesc b_remainder_desc_;
  UniqueLayoutDesc c_remainder_desc_;
  UniqueLayoutDesc d_remainder_desc_;
};

/*static*/ constexpr int CUDABlasLtMatmulPlan::kMaxBatchCount;

bool CUDABlasLtMatmulPlan::SetBiasPointer(const void *bias) const {
  return SetCublasLtAttr(op_desc_.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                         bias)
      .ok();
}

class CUDABlasLtMatmulAlgorithm final : public blas::IBlasLtMatmulAlgorithm {
 public:
  CUDABlasLtMatmulAlgorithm(blas::AlgorithmType index,
                            cublasLtMatmulAlgo_t algo, size_t workspace_size)
      : index_(index), algo_(algo), workspace_size_(workspace_size) {}

  blas::AlgorithmType index() const override { return index_; }

  size_t workspace_size() const override { return workspace_size_; }

  const cublasLtMatmulAlgo_t *algo() const { return &algo_; }

  int algo_id() const {
    int id;
    GetCublasLtAttr(&algo_, CUBLASLT_ALGO_CONFIG_ID, &id);
    return id;
  }

 private:
  blas::AlgorithmType index_;
  cublasLtMatmulAlgo_t algo_;
  size_t workspace_size_;
};

port::StatusOr<UniqueMatmulPreference> CreateCublasLtMatmulPreference(
    const blas::IBlasLtMatmulPlan *plan, size_t max_workspace_bytes) {
  cublasLtMatmulPreference_t preference;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("cublasLtMatmulPreferenceCreate failed: ",
                                     ToString(status)));
  }
  UniqueMatmulPreference unique_preference(preference);
  SE_RETURN_IF_ERROR(SetCublasLtAttr(preference,
                                     CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     max_workspace_bytes));

  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  if (cuda_plan.params().batch_count == 0) {
    return unique_preference;
  }
  // This is a workaround for a known issue in cuBlasLt where the heuristic may
  // in rare cases select an algo that does not support the specified stride.
  // Specifying the alignment requirements manually like this avoids the issue.
  auto get_alignment_bytes = [](int64 stride, blas::DataType dtype) {
    return (stride & -stride) * GetDataTypeSizeBytes(dtype);
  };
  if (cuda_plan.params().stride_a) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_a,
                                    cuda_plan.params().ab_type)));
  }
  if (cuda_plan.params().stride_b) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_b,
                                    cuda_plan.params().ab_type)));
  }
  if (cuda_plan.params().stride_c) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_c,
                                    cuda_plan.params().c_type)));
  }
  if (cuda_plan.params().stride_c) {
    SE_RETURN_IF_ERROR(SetCublasLtAttr(
        preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        (uint32)get_alignment_bytes(cuda_plan.params().stride_c,
                                    cuda_plan.params().c_type)));
  }
  return unique_preference;
}

}  // namespace

#endif  // CUDA_VERSION >= 11000

port::StatusOr<std::unique_ptr<blas::IBlasLtMatmulPlan>>
CUDABlas::CreateBlasLtMatmulPlan(const blas::BlasLtMatmulPlanParams &p) {
#if CUDA_VERSION >= 11000
  auto cuda_plan = std::make_unique<CUDABlasLtMatmulPlan>();
  SE_RETURN_IF_ERROR(cuda_plan->init(p));
  return static_cast<std::unique_ptr<blas::IBlasLtMatmulPlan>>(
      std::move(cuda_plan));
#else
  return port::Status(
      port::error::UNIMPLEMENTED,
      "CreateBlasLtMatmulPlan is not supported with this version of CUDA");
#endif
}

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
CUDABlas::GetBlasLtMatmulAlgorithmsInternal(const blas::IBlasLtMatmulPlan *plan,
                                            size_t max_workspace_size,
                                            int max_algorithm_count,
                                            bool for_remainder_batch) {
#if CUDA_VERSION >= 11000
  SE_ASSIGN_OR_RETURN(UniqueMatmulPreference preference,
                      CreateCublasLtMatmulPreference(plan, max_workspace_size));

  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    absl::MutexLock lock(&mu_);

    CHECK(blasLt_ != nullptr);

    gpu::ScopedActivateExecutorContext sac{parent_};

    int found_algorithm_count = 0;
    const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
    const auto &a_desc =
        for_remainder_batch ? cuda_plan.a_remainder_desc() : cuda_plan.a_desc();
    const auto &b_desc =
        for_remainder_batch ? cuda_plan.b_remainder_desc() : cuda_plan.b_desc();
    const auto &c_desc =
        for_remainder_batch ? cuda_plan.c_remainder_desc() : cuda_plan.c_desc();
    const auto &d_desc =
        for_remainder_batch ? cuda_plan.d_remainder_desc() : cuda_plan.d_desc();
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        blasLt_, cuda_plan.op_desc(), a_desc, b_desc, c_desc, d_desc,
        preference.get(), max_algorithm_count, results.data(),
        &found_algorithm_count);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return port::Status(
          port::error::INTERNAL,
          absl::StrCat("cublasLtMatmulAlgoGetHeuristic failed: ",
                       ToString(status)));
    }
    results.resize(found_algorithm_count);
  }

  std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>> out_algorithms;
  out_algorithms.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    if (result.state != CUBLAS_STATUS_SUCCESS) continue;  // Skip failed algos
    out_algorithms.emplace_back(std::make_unique<CUDABlasLtMatmulAlgorithm>(
        i, result.algo, result.workspaceSize));
  }
  return out_algorithms;
#else  // if CUDA_VERSION < 11000
  return port::Status(
      port::error::UNIMPLEMENTED,
      "GetBlasLtMatmulAlgorithms is not supported with this version of CUDA");
#endif
}

port::StatusOr<std::vector<std::unique_ptr<blas::IBlasLtMatmulAlgorithm>>>
CUDABlas::GetBlasLtMatmulAlgorithms(const blas::IBlasLtMatmulPlan *plan,
                                    size_t max_workspace_size,
                                    int max_algorithm_count) {
  return GetBlasLtMatmulAlgorithmsInternal(plan, max_workspace_size,
                                           max_algorithm_count);
}

#if CUDA_VERSION >= 11000
bool CUDABlas::DoBlasLtMatmulInternal(
    Stream *stream, bool err_on_failure, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, DeviceMemoryBase d, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias) {
  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  const auto &cuda_algo =
      *static_cast<const CUDABlasLtMatmulAlgorithm *>(algorithm);

  if (alpha.data_type() != cuda_plan.scale_type() ||
      beta.data_type() != cuda_plan.scale_type()) {
    VLOG(2) << "DoBlasLtMatmul returning false because alpha and beta types do "
               "not match plan: expected "
            << cuda_plan.c_type() << ", got alpha=" << alpha.data_type()
            << " beta=" << beta.data_type();
    return false;
  }
  if (alpha.is_pointer() != beta.is_pointer()) {
    VLOG(2) << "DoBlasLtMatmul returning false because one of `alpha` "
               "and `beta` is a pointer, but the other is not.";
    return false;
  }
  bool is_pointer_mode_host = !alpha.is_pointer();
  if ((cuda_plan.params().pointer_mode == blas::PointerMode::kHost) !=
      is_pointer_mode_host) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "pointer_mode for the given alpha/beta.";
    return false;
  }
  if ((cuda_plan.params().epilogue == blas::Epilogue::kBias ||
       cuda_plan.params().epilogue == blas::Epilogue::kBiasThenReLU) !=
      (bias != nullptr)) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "epilogue for the given bias pointer.";
    return false;
  }
  const void *alpha_ptr = alpha.is_pointer() ? alpha.opaque_pointer().opaque()
                                             : alpha.opaque_value();
  const void *beta_ptr =
      beta.is_pointer() ? beta.opaque_pointer().opaque() : beta.opaque_value();

  void *workspace = nullptr;
  if (cuda_algo.workspace_size()) {
    port::Status allocation_status = AllocateWorkspace(
        &workspace, scratch_allocator, cuda_algo.workspace_size());
    if (!allocation_status.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR)
            << "Failed to allocate workspace for cublasLtMatmul algo with id: "
            << cuda_algo.algo_id() << " requiring "
            << cuda_algo.workspace_size() << " bytes of workspace";
      }
      return false;
    }
  }

  // This is only used when batch_count > kMaxBatchCount.
  std::unique_ptr<blas::IBlasLtMatmulAlgorithm> unique_remainder_algo;
  if (cuda_plan.remainder_batch_count()) {
    // There is no easy way to get the user-specified max workspace size here,
    // so we just allow a very small amount and don't worry too much about
    // performance because this is only used in rare cases. The same reasoning
    // applies to selection of the algorithm.
    size_t max_workspace_size = 4 * 1024 * 1024;  // 4 MiB
    auto status_or_algorithms =
        GetBlasLtMatmulAlgorithmsInternal(plan, max_workspace_size,
                                          /* max_algorithm_count = */ 1,
                                          /* for_remainder_batch = */ true);
    if (!status_or_algorithms.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "Failed to get algorithms for blasLt remainder batch.";
      }
      return false;
    }
    auto algorithms = status_or_algorithms.ConsumeValueOrDie();
    unique_remainder_algo = std::move(algorithms.front());
  }

  cudaStream_t cuda_stream = CUDAStream(stream);

  absl::MutexLock lock(&mu_);

  if (bias != nullptr) {
    if (!cuda_plan.SetBiasPointer(bias.opaque())) {
      VLOG(2) << "DoBlasLtMatmul returning false because setting the bias "
                 "pointer failed.";
      return false;
    }
  }

  CHECK(blasLt_ != nullptr);

  gpu::ScopedActivateExecutorContext sac{parent_};

  // Plan execution is broken down into repeat calls with capped_batch_count,
  // followed by a final call with remainder_batch_count.
  // Cases where batch_count <= kMaxBatchCount require only a single call (a
  // single loop iteration and no remainder).
  int ab_type_size = GetDataTypeSizeBytes(cuda_plan.params().ab_type);
  int c_type_size = GetDataTypeSizeBytes(cuda_plan.params().c_type);
  const char *a_ptr = static_cast<const char *>(a.opaque());
  const char *b_ptr = static_cast<const char *>(b.opaque());
  const char *c_ptr = static_cast<const char *>(c.opaque());
  char *d_ptr = static_cast<char *>(d.opaque());
  int capped_batch_count = cuda_plan.capped_batch_count();
  for (int batch = 0;
       batch + capped_batch_count <= cuda_plan.params().batch_count;
       batch += capped_batch_count) {
    cublasStatus_t ret = cublasLtMatmul(
        blasLt_, cuda_plan.op_desc(), alpha_ptr, a_ptr, cuda_plan.a_desc(),
        b_ptr, cuda_plan.b_desc(), beta_ptr, c_ptr, cuda_plan.c_desc(), d_ptr,
        cuda_plan.d_desc(), cuda_algo.algo(), workspace,
        cuda_algo.workspace_size(), cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run cublasLtMatmul routine: " << ToString(ret);
      }
      return false;
    }
    a_ptr += capped_batch_count * cuda_plan.params().stride_a * ab_type_size;
    b_ptr += capped_batch_count * cuda_plan.params().stride_b * ab_type_size;
    c_ptr += capped_batch_count * cuda_plan.params().stride_c * c_type_size;
    d_ptr += capped_batch_count * cuda_plan.params().stride_c * c_type_size;
  }
  // This is only used when batch_count > kMaxBatchCount.
  if (cuda_plan.remainder_batch_count()) {
    const auto &remainder_algo =
        *static_cast<const CUDABlasLtMatmulAlgorithm *>(
            unique_remainder_algo.get());
    if (remainder_algo.workspace_size()) {
      port::Status allocation_status = AllocateWorkspace(
          &workspace, scratch_allocator, remainder_algo.workspace_size());
      if (!allocation_status.ok()) {
        if (err_on_failure || VLOG_IS_ON(3)) {
          LOG(ERROR) << "Failed to allocate workspace for cublasLtMatmul algo "
                        "with id: "
                     << remainder_algo.algo_id() << " requiring "
                     << remainder_algo.workspace_size()
                     << " bytes of workspace";
        }
        return false;
      }
    }
    cublasStatus_t ret = cublasLtMatmul(
        blasLt_, cuda_plan.op_desc(), alpha_ptr, a_ptr,
        cuda_plan.a_remainder_desc(), b_ptr, cuda_plan.b_remainder_desc(),
        beta_ptr, c_ptr, cuda_plan.c_remainder_desc(), d_ptr,
        cuda_plan.d_remainder_desc(), remainder_algo.algo(), workspace,
        remainder_algo.workspace_size(), cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run remainder cublasLtMatmul routine: "
                   << ToString(ret);
      }
      return false;
    }
  }
  return true;
}
#endif  // CUDA_VERSION >= 11000

bool CUDABlas::DoBlasLtMatmul(
    Stream *stream, const blas::IBlasLtMatmulPlan *plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
    const blas::IBlasLtMatmulAlgorithm *algorithm, DeviceMemoryBase bias,
    blas::ProfileResult *output_profile_result) {
#if CUDA_VERSION >= 11000
  const auto &cuda_plan = *static_cast<const CUDABlasLtMatmulPlan *>(plan);
  HostOrDeviceScalar<void> alpha_cast = alpha;
  HostOrDeviceScalar<void> beta_cast = beta;
  if (cuda_plan.c_type() == blas::DataType::kHalf &&
      cuda_plan.scale_type() == blas::DataType::kFloat) {
    // The given alpha and beta types are F16 (they always match c), but F32*
    // computation type requires that they be F32, so we must cast them.
    if (alpha.is_pointer() || beta.is_pointer()) {
      // We cannot easily convert a pointer to f16 memory to a pointer to f32
      // memory from here, so we don't support this for now.
      return false;
    }
    alpha_cast = HostOrDeviceScalar<void>(
        static_cast<float>(alpha.value<Eigen::half>()));
    beta_cast =
        HostOrDeviceScalar<void>(static_cast<float>(beta.value<Eigen::half>()));
  }

  std::unique_ptr<GpuTimer, GpuTimerDeleter> timer;
  if (output_profile_result) {
    timer.reset(new GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(AsGpuStream(stream))) {
      return false;
    }
  }

  bool err_on_failure = timer != nullptr;
  bool result = DoBlasLtMatmulInternal(stream, err_on_failure, plan, alpha_cast,
                                       a, b, beta_cast, c, c, scratch_allocator,
                                       algorithm, bias);

  if (timer && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm->index());
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
#else  // if CUDA_VERSION < 11000
  return false;
#endif
}

port::Status CUDABlas::GetVersion(std::string *version) {
  absl::MutexLock lock(&mu_);

  int v;
  auto status = cublasGetVersion(blas_, &v);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::InternalError(ToString(status));
  }
  *version = std::to_string(v);
  return port::Status::OK();
}

}  // namespace gpu

void initialize_cublas() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          cuda::kCudaPlatformId, gpu::kCuBlasPlugin, "cuBLAS",
          [](internal::StreamExecutorInterface *parent) -> blas::BlasSupport * {
            gpu::GpuExecutor *cuda_executor =
                dynamic_cast<gpu::GpuExecutor *>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR)
                  << "Attempting to initialize an instance of the cuBLAS "
                  << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::CUDABlas *blas = new gpu::CUDABlas(cuda_executor);
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
      cuda::kCudaPlatformId, PluginKind::kBlas, gpu::kCuBlasPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cublas,
                            { stream_executor::initialize_cublas(); });
