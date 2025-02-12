/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cuda_blas.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "Eigen/Core"
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/gpus/cuda/include/library_types.h"
#include "third_party/gpus/cuda/include/vector_types.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/cuda/cuda_blas_utils.h"
#include "xla/stream_executor/cuda/cuda_helpers.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/protobuf/dnn.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"

namespace stream_executor {
namespace cuda {

using gpu::GpuMemory;
using gpu::GpuMemoryMutable;

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

static const char *const kCublasNotInitializedExplanation =
    "Failure to initialize cublas may be due to OOM (cublas needs some free "
    "memory when you initialize it, and your deep-learning framework may have "
    "preallocated more than its fair share), or may be because this binary was "
    "not built with support for the GPU in your machine.";

bool CUDABlas::Init() {
  absl::MutexLock lock(&mu_);

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  cublasStatus_t ret = cublasCreate(&blas_);
  if (ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create cublas handle: " << ToString(ret);
    if (ret == CUBLAS_STATUS_NOT_INITIALIZED ||
        ret == CUBLAS_STATUS_ALLOC_FAILED) {
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

CUDABlas::CUDABlas(StreamExecutor *parent)
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
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    cublasDestroy(blas_);
  }
}

bool CUDABlas::SetStream(Stream *stream) {
  CHECK(blas_ != nullptr);
  std::unique_ptr<ActivateContext> activation = parent_->Activate();

  auto handle = (stream != nullptr) ? gpu::AsGpuStreamValue(stream) : nullptr;
  if (auto ret = cublasSetStream(blas_, handle); ret != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for cuBLAS calls: " << ToString(ret);
    return false;
  }
  return true;
}

absl::StatusOr<bool> CUDABlas::IsMainStreamSet() const {
  absl::MutexLock lock{&mu_};
  CHECK(blas_ != nullptr);
  CUstream handle{};
  if (auto ret = cublasGetStream(blas_, &handle);
      ret != CUBLAS_STATUS_SUCCESS) {
    return absl::InternalError("failed to get the current stream value");
  }
  return (handle == nullptr);
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
  static constexpr cudaDataType_t type = CUDA_R_16F;  // NOLINT
};

#if CUDA_VERSION >= 11000
template <>
struct CUDADataType<Eigen::bfloat16> {
  static constexpr cudaDataType_t type = CUDA_R_16BF;  // NOLINT
};
#endif  // CUDA_VERSION >= 11000

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
absl::Status CUDABlas::DoBlasInternalImpl(FuncT cublas_func, Stream *stream,
                                          bool pointer_mode_host,
                                          cublasMath_t math_type,
                                          Args... args) {
  absl::MutexLock lock(&mu_);

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return absl::InternalError("Failed setting stream");
  }

  // Set workspace to a user-owned buffer, otherwise cuBlas will use its own
  // memory pool, and it's not compatible with CUDA graphs.
  if (auto *workspace = GetWorkspace();
      workspace && workspace->opaque() && workspace->size() > 0) {
    cublasStatus_t ret =
        cublasSetWorkspace(blas_, workspace->opaque(), workspace->size());
    if (ret != CUBLAS_STATUS_SUCCESS) {
      return absl::InternalError(
          absl::StrCat("Failed setting cuBlas workspace: ", ToString(ret)));
    }
  }

  ScopedCublasMathMode math_mode{blas_};
#if CUBLAS_VER_MAJOR >= 11
  if (math_type == CUBLAS_TF32_TENSOR_OP_MATH &&
      tsl::tensor_float_32_execution_enabled()) {
#else
  if (math_type == CUBLAS_TENSOR_OP_MATH) {
#endif
    if (!math_mode.Init(math_type)) {
      return absl::InternalError("Failed initializing math mode");
    }
  }

  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  ScopedCublasPointerMode pointer_mode{blas_};
  if (!pointer_mode.Init(pointer_mode_host ? CUBLAS_POINTER_MODE_HOST
                                           : CUBLAS_POINTER_MODE_DEVICE)) {
    return absl::InternalError("Failed setting error mode");
  }
  cublasStatus_t ret = cublas_func(blas_, args...);
  if (ret == CUBLAS_STATUS_SUCCESS) {
    return absl::OkStatus();
  }
  return absl::InternalError(ToString(ret));
}

// cublas_func may be overloaded, so we need to figure out which one we really
// need to call based on the args. One way to do it is to wrap it in lambda.
#define AS_LAMBDA(func)                                            \
  [](auto &&...args) -> decltype(func(                             \
                         std::forward<decltype(args)>(args)...)) { \
    return func(std::forward<decltype(args)>(args)...);            \
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
                        elem_count, &alpha, CUDAComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(cublasZdscal, stream, true /* = pointer_mode_host */,
                        elem_count, &alpha, CUDAComplex(GpuMemoryMutable(x)),
                        incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(cublasCscal, stream, true /* = pointer_mode_host */,
                        elem_count, CUDAComplex(&cb_alpha),
                        CUDAComplex(GpuMemoryMutable(x)), incx);
}

bool CUDABlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(cublasZscal, stream, true /* = pointer_mode_host */,
                        elem_count, CUDAComplex(&cb_alpha),
                        CUDAComplex(GpuMemoryMutable(x)), incx);
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
  auto cb_alpha = CUDAComplexValue(alpha);
  auto cb_beta = CUDAComplexValue(beta);
  return DoBlasInternal(cublasCgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, CUDAComplex(&cb_alpha),
                        CUDAComplex(GpuMemory(a)), lda,
                        CUDAComplex(GpuMemory(x)), incx, CUDAComplex(&cb_beta),
                        CUDAComplex(GpuMemoryMutable(y)), incy);
}

bool CUDABlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  auto cb_alpha = CUDAComplexValue(alpha);
  auto cb_beta = CUDAComplexValue(beta);
  return DoBlasInternal(cublasZgemv, stream, true /* = pointer_mode_host */,
                        AsCublasOperation(trans), m, n, CUDAComplex(&cb_alpha),
                        CUDAComplex(GpuMemory(a)), lda,
                        CUDAComplex(GpuMemory(x)), incx, CUDAComplex(&cb_beta),
                        CUDAComplex(GpuMemoryMutable(y)), incy);
}

absl::Status CUDABlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, const DeviceMemoryBase &b, int ldb,
    const void *beta, DeviceMemoryBase *c, int ldc,
    const NumericOptions &numeric_options, blas::CallContext context) {
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;

#if CUDA_VERSION < 11000
  if (dtype == blas::DataType::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == blas::DataType::kFloat) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
    if (!numeric_options.allow_tf32) {
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
      return DoBlasInternalImpl(
          cublasSgemmEx, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          static_cast<const float *>(alpha), a.opaque(), CUDA_R_16F, lda,
          b.opaque(), CUDA_R_16F, ldb, static_cast<const float *>(beta),
          c->opaque(), CUDA_R_16F, ldc);
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
      cuComplex cb_alpha =
          CUDAComplexValue(*static_cast<const std::complex<float> *>(alpha));
      cuComplex cb_beta =
          CUDAComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          &cb_alpha, static_cast<const cuComplex *>(a.opaque()), lda,
          static_cast<const cuComplex *>(b.opaque()), ldb, &cb_beta,
          static_cast<cuComplex *>(c->opaque()), ldc);
    }
    case dnn::kComplexDouble: {
      cuDoubleComplex cb_alpha =
          CUDAComplexValue(*static_cast<const std::complex<double> *>(alpha));
      cuDoubleComplex cb_beta =
          CUDAComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemm, stream, true /* = pointer_mode_host */, math_type,
          AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
          &cb_alpha, static_cast<const cuDoubleComplex *>(a.opaque()), lda,
          static_cast<const cuDoubleComplex *>(b.opaque()), ldb, &cb_beta,
          static_cast<cuDoubleComplex *>(c->opaque()), ldc);
    }
    default:
      return absl::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

static bool UsesTensorOps(blas::AlgorithmType algo) {
  cublasGemmAlgo_t cublas_algo = static_cast<cublasGemmAlgo_t>(algo);
  return cublas_algo >= CUBLAS_GEMM_DEFAULT_TENSOR_OP;
}

static absl::StatusOr<cublasMath_t> GetMathTypeForGemmEx(
    Stream *stream, blas::AlgorithmType algorithm, blas::DataType type_a,
    blas::DataType type_b, const NumericOptions &numeric_options) {
  if (type_a != type_b) {
    return absl::InternalError("Types of inputs mismatch");
  }

  // GPUs < sm_50 don't support cublasGemmEx.
  CudaComputeCapability cc = stream->GetCudaComputeCapability();
  if (cc.major < 5) {
    return absl::InternalError(absl::StrCat(
        "sm_", cc.major, " does not support explicit gemm algorithms."));
  }

  bool algo_uses_tensor_ops = UsesTensorOps(algorithm);
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
  if (algo_uses_tensor_ops) {
    if (cc.major < 7) {
      return absl::InternalError(absl::StrCat(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available in sm", cc.major,
          "X devices."));
    } else if (type_a == blas::DataType::kFloat) {
#if CUDA_VERSION < 11000
      return absl::InternalError(
          "Algorithm ", algorithm,
          " uses tensor ops, but tensor ops are not available for fp32");
#else
      if (cc.major < 8) {
        return absl::InternalError(absl::StrCat(
            "Algorithm ", algorithm,
            " uses tensor ops, but tensor ops are not available in sm",
            cc.major, "X devices for float input types."));
      }
      math_type = CUBLAS_TF32_TENSOR_OP_MATH;
#endif
    } else if (type_a == blas::DataType::kHalf) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#endif
    } else {
      return absl::InternalError(
          absl::StrCat("Algorithm ", algorithm,
                       " uses tensor ops which are not supported for input"));
    }
  }
  if (!numeric_options.allow_tf32) {
    math_type = CUBLAS_DEFAULT_MATH;
  }

  return math_type;
}

static absl::Status PopulateProfileFromTimer(
    EventBasedTimer *timer, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  if (output_profile_result) {
    TF_ASSIGN_OR_RETURN(absl::Duration duration, timer->GetElapsedDuration());
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        absl::ToDoubleMilliseconds(duration));
  }
  return absl::OkStatus();
}

absl::Status CUDABlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result, blas::CallContext context) {
  TF_ASSIGN_OR_RETURN(
      cublasMath_t math_type,
      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b, numeric_options));

  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer,
                        stream->CreateEventBasedTimer(
                            output_profile_result->warmup_run_executed()));
  }

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
  TF_RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  return absl::OkStatus();
}

absl::Status CUDABlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *output_profile_result, blas::CallContext context) {
  TF_ASSIGN_OR_RETURN(
      cublasMath_t math_type,
      GetMathTypeForGemmEx(stream, algorithm, type_a, type_b, numeric_options));
  std::unique_ptr<EventBasedTimer> timer;
  if (output_profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer,
                        stream->CreateEventBasedTimer(
                            output_profile_result->warmup_run_executed()));
  }
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

      if (AsCudaDataType(type_c) == CUDA_R_16BF) {
        auto *c_matrix = reinterpret_cast<__nv_bfloat16 *>(
            static_cast<Eigen::bfloat16 *>(c->opaque()) + batch * stride_c);
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true,
            math_type, AsCublasOperation(transa), AsCublasOperation(transb), m,
            n, k, static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
            b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
            c_matrix, AsCudaDataType(type_c), ldc,
            AsCublasComputeType(computation_type),
            static_cast<cublasGemmAlgo_t>(algorithm)));
      } else if (AsCudaDataType(type_c) == CUDA_R_32F) {
        auto *c_matrix = static_cast<float *>(c->opaque()) + batch * stride_c;
        TF_RETURN_IF_ERROR(DoBlasInternalImpl(
            AS_LAMBDA(cublasGemmEx), stream, /*pointer_mode_host=*/true,
            math_type, AsCublasOperation(transa), AsCublasOperation(transb), m,
            n, k, static_cast<const float *>(alpha), a_matrix, CUDA_R_16BF, lda,
            b_matrix, CUDA_R_16BF, ldb, static_cast<const float *>(beta),
            c_matrix, AsCudaDataType(type_c), ldc,
            AsCublasComputeType(computation_type),
            static_cast<cublasGemmAlgo_t>(algorithm)));
      } else {
        return absl::InternalError(absl::StrCat(
            "Unsupported type combination for GEMM: %s and %s",
            blas::DataTypeString(type_a), blas::DataTypeString(type_c)));
      }
    }
    TF_RETURN_IF_ERROR(PopulateProfileFromTimer(timer.get(), algorithm,
                                                output_profile_result));
    return absl::OkStatus();
  }
#endif

  TF_RETURN_IF_ERROR(DoBlasInternalImpl(
      AS_LAMBDA(cublasGemmStridedBatchedEx), stream, /*pointer_mode_host=*/true,
      math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
      alpha, a.opaque(), cuda_in_type, lda, stride_a, b.opaque(), cuda_in_type,
      ldb, stride_b, beta, c->opaque(), AsCudaDataType(type_c), ldc, stride_c,
      batch_count, AsCublasComputeType(computation_type),
      static_cast<cublasGemmAlgo_t>(algorithm)));
  TF_RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, output_profile_result));
  return absl::OkStatus();
}

bool CUDABlas::GetBlasGemmAlgorithms(
    Stream *stream, const gpu::MatrixDescriptor &,
    const gpu::MatrixDescriptor &, gpu::OutputMatrixDescriptor *, const void *,
    const void *, std::vector<blas::AlgorithmType> *out_algorithms) {
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

template <>
struct HalfAsFloat<Eigen::bfloat16> {
  typedef float type;
};

namespace {
// pass-through for non-complex types that don't need conversion to
// cublas-specific type.
template <typename T>
T inline CUDAComplexValue(T v) {
  return v;
}
}  // namespace

template <typename T, typename Scalar, typename FuncT>
absl::Status CUDABlas::DoBlasGemmBatchedInternal(
    FuncT cublas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, Scalar alpha,
    const DeviceMemorySlice<T> &a_ptrs_to_wrappers, int lda,
    const DeviceMemorySlice<T> &b_ptrs_to_wrappers, int ldb, Scalar beta,
    const DeviceMemorySlice<T> &c_ptrs_to_wrappers, int ldc, int batch_count,
    const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator) {
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  typedef typename HalfAsFloat<typename CUDAComplexT<T>::type>::type CUDA_T;

  const size_t size = batch_count * sizeof(CUDA_T *);

  if (scratch_allocator == nullptr) {
    return absl::InternalError("scratch_allocator is null");
  }
  TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> a_bytes,
                      scratch_allocator->AllocateBytes(size));
  TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> b_bytes,
                      scratch_allocator->AllocateBytes(size));
  TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> c_bytes,
                      scratch_allocator->AllocateBytes(size));
  DeviceMemory<CUDA_T *> a(a_bytes);
  DeviceMemory<CUDA_T *> b(b_bytes);
  DeviceMemory<CUDA_T *> c(c_bytes);

  TF_RETURN_IF_ERROR(stream->Memcpy(&a, a_raw_ptrs.data(), size));
  TF_RETURN_IF_ERROR(stream->Memcpy(&b, b_raw_ptrs.data(), size));
  TF_RETURN_IF_ERROR(stream->Memcpy(&c, c_raw_ptrs.data(), size));

  cudaDataType_t data_type = CUDADataType<T>::type;

  if (stream->GetCudaComputeCapability().IsAtLeast(5)) {
    cublasMath_t math_type;
    cublasGemmAlgo_t algo;

#if CUDA_VERSION >= 11000
    bool is_16bit = data_type == CUDA_R_16F || data_type == CUDA_R_16BF;
#else
    bool is_16bit = data_type == CUDA_R_16F;
#endif  // CUDA_VERSION >= 11000

    if (is_16bit) {
#if CUDA_VERSION < 11000
      math_type = CUBLAS_TENSOR_OP_MATH;
#else
      math_type = CUBLAS_DEFAULT_MATH;
#endif
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
#if CUBLAS_VER_MAJOR >= 11
    } else if (data_type == CUDA_R_32F) {
      if (numeric_options.allow_tf32 &&
          tsl::tensor_float_32_execution_enabled()) {
        math_type = CUBLAS_TENSOR_OP_MATH;
        algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
      } else {
        math_type = CUBLAS_DEFAULT_MATH;
        algo = CUBLAS_GEMM_DFALT;
      }
#endif
    } else {
      math_type = CUBLAS_DEFAULT_MATH;
      algo = CUBLAS_GEMM_DFALT;
    }
    cudaDataType_t compute_type = is_16bit ? CUDA_R_32F : data_type;
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
  // SM < 5.0
  if (data_type != CUDA_R_16F) {
    auto cb_alpha = CUDAComplexValue(alpha);
    auto cb_beta = CUDAComplexValue(beta);
    bool ok = DoBlasInternal(
        cublas_func, stream, true /* = pointer_mode_host */,
        AsCublasOperation(transa), AsCublasOperation(transb), m, n, k,
        CUDAComplex(&cb_alpha), const_cast<const CUDA_T **>(GpuMemory(a)), lda,
        const_cast<const CUDA_T **>(GpuMemory(b)), ldb, CUDAComplex(&cb_beta),
        const_cast<CUDA_T **>(GpuMemory(c)), ldc, batch_count);
    if (ok) {
      return absl::OkStatus();
    }
    return absl::InternalError("failed BLAS call, see log for details");
  } else {
    // Fall back to a loop for fp16
    for (int b = 0; b < batch_count; ++b) {
      const DeviceMemory<T> &a_matrix = *a_ptrs_to_wrappers[b];
      const DeviceMemory<T> &b_matrix = *b_ptrs_to_wrappers[b];
      DeviceMemory<T> *c_matrix = c_ptrs_to_wrappers[b];
      TF_RETURN_IF_ERROR(DoBlasGemm(
          stream, transa, transb, m, n, k, blas::ToDataType<T>::value, &alpha,
          a_matrix, lda, b_matrix, ldb, &beta, c_matrix, ldc, numeric_options,
          blas::CallContext::kNone));
    }
    return absl::OkStatus();
  }
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceMemorySlice<Eigen::half> a_array,
    int lda, DeviceMemorySlice<Eigen::half> b_array, int ldb, float beta,
    DeviceMemorySlice<Eigen::half> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  // Note: The func passed here (cublasSgemmBatched) is not actually called,
  // due to special handling of fp16 inside DoBlasGemmBatchedInternal.
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha,
    DeviceMemorySlice<Eigen::bfloat16> a_array, int lda,
    DeviceMemorySlice<Eigen::bfloat16> b_array, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  // Note: The func passed here (cublasSgemmBatched) is not actually called,
  // due to special handling of bf16 inside DoBlasGemmBatchedInternal.
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceMemorySlice<float> a_array,
    int lda, DeviceMemorySlice<float> b_array, int ldb, float beta,
    DeviceMemorySlice<float> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasSgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, double alpha, DeviceMemorySlice<double> a_array,
    int lda, DeviceMemorySlice<double> b_array, int ldb, double beta,
    DeviceMemorySlice<double> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasDgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,

      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, std::complex<float> alpha,
    DeviceMemorySlice<std::complex<float>> a_array, int lda,
    DeviceMemorySlice<std::complex<float>> b_array, int ldb,
    std::complex<float> beta, DeviceMemorySlice<std::complex<float>> c_array,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,

      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool CUDABlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, std::complex<double> alpha,
    DeviceMemorySlice<std::complex<double>> a_array, int lda,
    DeviceMemorySlice<std::complex<double>> b_array, int ldb,
    std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c_array,
    int ldc, int batch_count, const NumericOptions &numeric_options,
    ScratchAllocator *scratch_allocator, blas::CallContext context) {
  absl::Status status = DoBlasGemmBatchedInternal(
      cublasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array, lda,
      b_array, ldb, beta, c_array, ldc, batch_count, numeric_options,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

absl::Status CUDABlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
    const NumericOptions &numeric_options, blas::CallContext context) {
  cublasMath_t math_type = CUBLAS_DEFAULT_MATH;
#if CUDA_VERSION < 11000
  if (dtype == dnn::kHalf) {
    math_type = CUBLAS_TENSOR_OP_MATH;
  }
#else
  if (dtype == dnn::kFloat && numeric_options.allow_tf32) {
    math_type = CUBLAS_TF32_TENSOR_OP_MATH;
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
      return absl::OkStatus();
    }
#endif
    case dnn::kHalf: {
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
      // SM < 5.0. Fall back to a loop.
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
            static_cast<const float *>(alpha), a_matrix, CUDA_R_16F, lda,
            b_matrix, CUDA_R_16F, ldb, static_cast<const float *>(beta),
            c_matrix, CUDA_R_16F, ldc));
      }
      return absl::OkStatus();
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
      cuComplex cb_alpha =
          CUDAComplexValue(*static_cast<const std::complex<float> *>(alpha));
      cuComplex cb_beta =
          CUDAComplexValue(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalImpl(
          cublasCgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, CUDAComplex(&cb_alpha), static_cast<const cuComplex *>(a.opaque()),
          lda, stride_a, static_cast<const cuComplex *>(b.opaque()), ldb,
          stride_b, CUDAComplex(&cb_beta),
          static_cast<cuComplex *>(c->opaque()), ldc, stride_c, batch_count);
    }
    case dnn::kComplexDouble: {
      cuDoubleComplex cb_alpha =
          CUDAComplexValue(*static_cast<const std::complex<double> *>(alpha));
      cuDoubleComplex cb_beta =
          CUDAComplexValue(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalImpl(
          cublasZgemmStridedBatched, stream, true /* = pointer_mode_host */,
          math_type, AsCublasOperation(transa), AsCublasOperation(transb), m, n,
          k, CUDAComplex(&cb_alpha),
          static_cast<const cuDoubleComplex *>(a.opaque()), lda, stride_a,
          static_cast<const cuDoubleComplex *>(b.opaque()), ldb, stride_b,
          CUDAComplex(&cb_beta), static_cast<cuDoubleComplex *>(c->opaque()),
          ldc, stride_c, batch_count);
    }
    default:
      return absl::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64_t n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return DoBlasInternal(cublasStrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64_t n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return DoBlasInternal(cublasDtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        &alpha, GpuMemory(a), lda, GpuMemoryMutable(b), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64_t n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(cublasCtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        CUDAComplex(&cb_alpha), CUDAComplex(GpuMemory(a)), lda,
                        CUDAComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64_t n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(cublasZtrsm, stream, true /* = pointer_mode_host */,
                        CUDABlasSide(side), CUDABlasUpperLower(uplo),
                        AsCublasOperation(transa), CUDABlasDiagonal(diag), m, n,
                        CUDAComplex(&cb_alpha), CUDAComplex(GpuMemory(a)), lda,
                        CUDAComplex(GpuMemoryMutable(b)), ldb);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64_t n,
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
                                 blas::Diagonal diag, uint64_t m, uint64_t n,
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
                                 blas::Diagonal diag, uint64_t m, uint64_t n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(
      cublasCtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), AsCublasOperation(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<float2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<float2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

bool CUDABlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64_t n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
  auto cb_alpha = CUDAComplexValue(alpha);
  return DoBlasInternal(
      cublasZtrsmBatched, stream, true /* = pointer_mode_host */,
      CUDABlasSide(side), CUDABlasUpperLower(uplo), AsCublasOperation(transa),
      CUDABlasDiagonal(diag), m, n, &cb_alpha,
      reinterpret_cast<double2 *const *>(GpuMemory(as)), lda,
      reinterpret_cast<double2 **>(GpuMemoryMutable(bs)), ldb, batch_count);
}

absl::Status CUDABlas::GetVersion(std::string *version) {
  absl::MutexLock lock(&mu_);

  int v;
  auto status = cublasGetVersion(blas_, &v);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return absl::InternalError(ToString(status));
  }
  *version = std::to_string(v);
  return absl::OkStatus();
}

void initialize_cublas() {
  absl::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::BlasFactory>(
          kCudaPlatformId, "cuBLAS",
          [](::stream_executor::StreamExecutor *parent) -> blas::BlasSupport * {
            CUDABlas *blas = new CUDABlas(parent);
            if (!blas->Init()) {
              // Note: Init() will log a more specific error.
              delete blas;
              return nullptr;
            }
            return blas;
          });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuBLAS factory: " << status.message();
  }
}

}  // namespace cuda
}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_cublas, {
  stream_executor::cuda::initialize_cublas();
});
