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

#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_blas.h"

#include "tensorflow/compiler/xla/stream_executor/rocm/rocblas_wrapper.h"

#define EIGEN_USE_GPU
#include <assert.h>

#include <complex>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/platform/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/platform/initialize.h"
#include "tensorflow/compiler/xla/stream_executor/platform/logging.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"
#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/util/determinism.h"
using tsl::OpDeterminismRequired;

namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

extern void broadcast_fp32(void* stream, float* dst, int dst_stride,
                           int batches, int src_batches, float* src, int size);

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const DeviceMemory<T> &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemory(a));
}

template <class T>
const typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    const T &a) {
  return reinterpret_cast<
      const typename RocBlasTypeConversionHelper<T>::mapped_type *>(&a);
}
template <class T>
typename RocBlasTypeConversionHelper<T>::mapped_type *complex_cast(
    DeviceMemory<T> *a) {
  return reinterpret_cast<
      typename RocBlasTypeConversionHelper<T>::mapped_type *>(
      GpuMemoryMutable(a));
}

static void blas_log(const char *c) {}


static string ToString(rocblas_status status) {
  switch (status) {
    case rocblas_status_success:
      return "rocblas_status_success";
    case rocblas_status_invalid_handle:
      return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
      return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
      return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
      return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
      return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
      return "rocblas_status_internal_error";
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret = wrap::rocblas_create_handle(&blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(gpu::GpuExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    gpu::ScopedActivateExecutorContext sac{parent_};
    wrap::rocblas_destroy_handle(blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsGpuStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  gpu::ScopedActivateExecutorContext sac{parent_};
  rocblas_status ret =
      wrap::rocblas_set_stream(blas_, AsGpuStreamValue(stream));
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into rocBLAS arguments.

rocblas_operation ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return rocblas_operation_none;
    case blas::Transpose::kTranspose:
      return rocblas_operation_transpose;
    case blas::Transpose::kConjugateTranspose:
      return rocblas_operation_conjugate_transpose;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

rocblas_fill ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return rocblas_fill_upper;
    case blas::UpperLower::kLower:
      return rocblas_fill_lower;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}

rocblas_diagonal ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return rocblas_diagonal_unit;
    case blas::Diagonal::kNonUnit:
      return rocblas_diagonal_non_unit;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}

rocblas_side ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return rocblas_side_left;
    case blas::Side::kRight:
      return rocblas_side_right;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  gpu::ScopedActivateExecutorContext sac{parent_};

  // set the atomics mode, leaving default to library
  bool allow_atomics = !OpDeterminismRequired();
  rocblas_status ret;
  if (!allow_atomics) {
    ret = wrap::rocblas_set_atomics_mode(blas_, rocblas_atomics_not_allowed);
    if (err_on_failure && ret != rocblas_status_success) {
      LOG(ERROR) << "failed to to set atomics mode before "
                 << rocblas_func.kName << ": " << ToString(ret);
    }
  }

  ret = rocblas_func(blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasAxpy");
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_caxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_zaxpy, stream, /* pointer_mode_host = */ true, elem_count,
      complex_cast(alpha), complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        GpuMemory(x), incx, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_ccopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64_t elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_zcopy, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(x), incx, complex_cast(y), incy);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  blas_log("DoBlasScal<float>");
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        GpuMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_csscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zdscal, stream,
                        /* pointer_mode_host = */ true, elem_count, &alpha,
                        complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_cscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_zscal, stream,
                        /* pointer_mode_host = */ true, elem_count,
                        complex_cast(alpha), complex_cast(x), incx);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  blas_log("DoBlasGemv");
  return DoBlasInternal(
      wrap::rocblas_cgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m,
                          uint64_t n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  blas_log("DoBlasGemv\n");
  return DoBlasInternal(
      wrap::rocblas_zgemv, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(trans), m, n, complex_cast(alpha), complex_cast(a), lda,
      complex_cast(x), incx, complex_cast(beta), complex_cast(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_ssbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64_t n,
                          uint64_t k, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_dsbmv, stream, /* pointer_mode_host = */ true,
      ROCMBlasUpperLower(uplo), n, k, &alpha, GpuMemory(a), lda, GpuMemory(x),
      incx, &beta, GpuMemoryMutable(y), incy);
}

tsl::Status ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64 n,
                                 uint64_t k, blas::DataType dtype,
                                 const void *alpha, const DeviceMemoryBase &a,
                                 int lda, const DeviceMemoryBase &b, int ldb,
                                 const void *beta, DeviceMemoryBase *c, int ldc,
                                 blas::ComputePrecision precision) {
  blas_log("DoBlasGemm");
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);
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

  switch (dtype) {
    case blas::DataType::kHalf: {
      tsl::StatusOr<bool> maybe_hasXDLOPS = GpuDriver::GetMFMASupport();
      if (maybe_hasXDLOPS.ok() && maybe_hasXDLOPS.value()) {
        VLOG(1) << "Using rocblas_gemm_ex";
        return DoBlasInternalStatus(
            wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb),
            (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, alpha, a.opaque(),
            rocblas_datatype_f16_r, lda, b.opaque(), rocblas_datatype_f16_r,
            ldb, beta, c->opaque(), rocblas_datatype_f16_r, ldc, c->opaque(),
            rocblas_datatype_f16_r, ldc, rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard, 0, 0);
      } else {
        VLOG(1) << "Using rocblas_hgemm";
        const Eigen::half alpha_half(*static_cast<const float *>(alpha));
        const Eigen::half beta_half(*static_cast<const float *>(beta));
        return DoBlasInternalStatus(
            wrap::rocblas_hgemm, stream, /* pointer_mode_host = */ true,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
            reinterpret_cast<const rocblas_half *>(&alpha_half),
            reinterpret_cast<const rocblas_half *>(a.opaque()), lda,
            reinterpret_cast<const rocblas_half *>(b.opaque()), ldb,
            reinterpret_cast<const rocblas_half *>(&beta_half),
            reinterpret_cast<rocblas_half *>(c->opaque()), ldc);
      }
    }
    case blas::DataType::kBF16:
      return DoBlasInternalStatus(
          wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), (rocblas_int)m,
          (rocblas_int)n, (rocblas_int)k, alpha, a.opaque(),
          rocblas_datatype_bf16_r, lda, b.opaque(), rocblas_datatype_bf16_r,
          ldb, beta, c->opaque(), rocblas_datatype_bf16_r, ldc, c->opaque(),
          rocblas_datatype_bf16_r, ldc, rocblas_datatype_f32_r,
          rocblas_gemm_algo_standard, 0, 0);
    case blas::DataType::kFloat:
      return DoBlasInternalStatus(
          wrap::rocblas_sgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          static_cast<const float *>(alpha),
          static_cast<const float *>(a.opaque()), lda,
          static_cast<const float *>(b.opaque()), ldb,
          static_cast<const float *>(beta), static_cast<float *>(c->opaque()),
          ldc);
    case blas::DataType::kDouble:
      return DoBlasInternalStatus(
          wrap::rocblas_dgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          static_cast<const double *>(alpha),
          static_cast<const double *>(a.opaque()), lda,
          static_cast<const double *>(b.opaque()), ldb,
          static_cast<const double *>(beta), static_cast<double *>(c->opaque()),
          ldc);
    case blas::DataType::kComplexFloat: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<float> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_cgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          cb_alpha, static_cast<const rocblas_float_complex *>(a.opaque()), lda,
          static_cast<const rocblas_float_complex *>(b.opaque()), ldb, cb_beta,
          static_cast<rocblas_float_complex *>(c->opaque()), ldc);
    }
    case blas::DataType::kComplexDouble: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<double> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_zgemm, stream, /* pointer_mode_host = */ true,
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          cb_alpha, static_cast<const rocblas_double_complex *>(a.opaque()),
          lda, static_cast<const rocblas_double_complex *>(b.opaque()), ldb,
          cb_beta, static_cast<rocblas_double_complex *>(c->opaque()), ldc);
    }
    default:
      return tsl::errors::Internal("Unsupported datatype for GEMM: ",
                                   blas::DataTypeString(dtype));
  }
}

tsl::Status ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ComputePrecision precision,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return tsl::errors::Internal("Not implemented on ROCm");
}

tsl::Status ROCMBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ComputePrecision precision,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return tsl::errors::Internal("Not implemented on ROCm");
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    Stream *stream, std::vector<blas::AlgorithmType> *out_algorithms) {
  // ROCM TODO: properly implement the interface
  return true;
}

struct memory_copy_op {
    char* src_ptr;
    char* dst_ptr;
    uint64_t size;
    uint64_t count;
    uint64_t dst_stride;
    uint64_t src_count;
};

static bool fold(memory_copy_op& y, const memory_copy_op& x) 
{
  bool misaligned = (x.size & 3) 
    || (reinterpret_cast<uint64_t>(x.dst_ptr) & 3)
    || (reinterpret_cast<uint64_t>(x.src_ptr) & 3)
    || (reinterpret_cast<uint64_t>(y.dst_ptr) & 3)
    || (reinterpret_cast<uint64_t>(y.src_ptr) & 3);
  int64_t dst_step = reinterpret_cast<int64_t>(x.dst_ptr)-reinterpret_cast<int64_t>(y.dst_ptr);
  if(x.src_ptr==y.src_ptr
    && x.size==y.size
    && (y.count==1 || x.dst_ptr==y.dst_ptr+y.count*y.dst_stride)
    && !misaligned
    && y.src_count==1
    && !(dst_step & 3) ) {
    if(y.count==1)
      y.dst_stride=dst_step;
    y.count++;
    return true;
  }
  else if(x.src_ptr==y.src_ptr+y.size && x.dst_ptr==y.dst_ptr+y.size && y.count==1 && y.src_count==1) {
    y.size += x.size;
    return true;
  }
  if(x.src_ptr == y.src_ptr+y.size*y.src_count
      && x.dst_ptr == y.dst_ptr + y.dst_stride*y.src_count*y.count
      && x.count == y.count
      && x.dst_stride == y.dst_stride) {
    y.src_count+=x.src_count;
    return true;
  }
  return false;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request
template <typename MAPPED_T>
tsl::Status ReorganizeMemory(Stream *stream,
                             DeviceMemory<MAPPED_T> *device_memory,
                             const std::vector<MAPPED_T *> &raw_ptrs,
                             int batch_count, uint64_t batch_stride,
                             bool gather) {
  if(gather==false)
    return tsl::Status(absl::StatusCode::kInternal, "gather=false unsupported");
  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);

  std::vector<memory_copy_op> ops;
  ops.push_back(memory_copy_op{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1});
  for (int i = 1; i < batch_count; ++i) {
    src_ptr = reinterpret_cast<char*>(raw_ptrs[i]);
    dst_ptr = device_memory_ptr + i * matrix_byte_size;

    memory_copy_op x{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1};
    while(ops.size()>1 && fold(ops[ops.size()-2], ops.back())) {
      ops.pop_back();
    }
    memory_copy_op& op = ops.back();
    if(fold(op, x)) {
      continue;
    }
    ops.push_back(x);
  }
  while(ops.size()>1 && fold(ops[ops.size()-2], ops.back()))
    ops.pop_back();
  int i=0;
  for (auto& x : ops) {
    if(x.src_count>1 || x.count>1) {
      broadcast_fp32(AsGpuStreamValue(stream),
                     reinterpret_cast<float*>(x.dst_ptr),
                     x.dst_stride >> 2, x.count, x.src_count,
                     reinterpret_cast<float*>(x.src_ptr),
                     x.size >> 2);
    }
    else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(x.src_ptr, x.size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(x.dst_ptr, x.size);
      bool a_status = stream->ThenMemcpy(&target_mem, src_mem, x.size).ok();
      if (!a_status) {
        return tsl::Status(
            absl::StatusCode::kInternal,
            "failed to copy device memory in ROCMBlas::DoBlasGemmBatched");
      }
    }
    i++;
  }
  fflush(stdout);
  return tsl::OkStatus();
}

template <typename T>
tsl::Status ROCMBlas::AllocateStridedBuffer(
    const std::vector<typename RocBlasTypeConversionHelper<T>::mapped_type *>
        &raw_ptrs,
    int batch_count, uint64_t batch_stride, ScratchAllocator *scratch_allocator,
    Stream *stream,
    std::unique_ptr<TemporaryDeviceMemory<
        typename RocBlasTypeConversionHelper<T>::mapped_type>> *temp_memory,
    DeviceMemory<typename RocBlasTypeConversionHelper<T>::mapped_type>
        *device_memory,
    bool copy_data, bool &reallocated) {
  assert(device_memory != nullptr);

  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  bool needs_allocate_strided = false;
  for (int i = 1; i < batch_count; ++i) {
    uint64_t tmp_batch_stride = raw_ptrs[i] - raw_ptrs[i - 1];
    if (tmp_batch_stride != batch_stride) {
      needs_allocate_strided = true;
      break;
    }
  }

  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);
  size_t matrix_batch_byte_size = matrix_byte_size * batch_count;

  // No need to do re-allocation, take the short cut and return
  if (!needs_allocate_strided) {
    *device_memory = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    reallocated = false;
    return tsl::OkStatus();
  }

  if (scratch_allocator != nullptr) {
    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8> batch_matrix_bytes,
        scratch_allocator->AllocateBytes(matrix_batch_byte_size));
    *device_memory = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  } else {
    assert(temp_memory != nullptr);
    TF_ASSIGN_OR_RETURN(*temp_memory, stream->AllocateTemporaryArray<MAPPED_T>(
                                          matrix_batch_byte_size));
    *device_memory =
        DeviceMemory<MAPPED_T>(*(*temp_memory)->mutable_device_memory());
  }
  assert(batch_count > 0);
  char* device_memory_ptr = static_cast<char*>(device_memory->opaque());
  char* src_ptr = reinterpret_cast<char*>(raw_ptrs[0]);
  char* dst_ptr = device_memory_ptr;
  uint64_t cur_stride_size = matrix_byte_size;

  reallocated = true;

  if (copy_data)
    return ReorganizeMemory(stream, device_memory, raw_ptrs, batch_count,
                            batch_stride, true);
  return tsl::OkStatus();
}


template <typename T, typename FuncT>
tsl::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64 n, uint64 k, T alpha,
    DeviceMemorySlice<T> a_ptrs_to_wrappers, int lda,
    DeviceMemorySlice<T> b_ptrs_to_wrappers, int ldb, T beta,
    DeviceMemorySlice<T> c_ptrs_to_wrappers, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  using MAPPED_T = typename RocBlasTypeConversionHelper<T>::mapped_type;

  // Sanity checks before making any further progress
  uint64_t batch_stride_a = 0;
  uint64_t batch_stride_b = 0;
  uint64_t batch_stride_c = 0;

  assert(ldc >= m);
  batch_stride_c = ldc * n;

  if (ROCMBlasTranspose(transa) == rocblas_operation_none) {
    assert(lda >= m);
    batch_stride_a = lda * k;
  } else {
    assert(lda >= k);
    batch_stride_a = lda * m;
  }

  if (ROCMBlasTranspose(transb) == rocblas_operation_none) {
    assert(ldb >= k);
    batch_stride_b = ldb * n;
  } else {
    assert(ldb >= n);
    batch_stride_b = ldb * k;
  }

  // Allocate local vectors to hold device pointers to matrices
  std::vector<MAPPED_T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(
        reinterpret_cast<MAPPED_T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  DeviceMemory<MAPPED_T> a;
  // Make sure the temporary memory are in-scope before the function returns
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> a_temp;
  bool reallocated_a, reallocated_b, reallocated_c;
  tsl::Status a_allocation_status = AllocateStridedBuffer<T>(
      a_raw_ptrs, batch_count, batch_stride_a, scratch_allocator, stream,
      &a_temp, &a, true, reallocated_a);
  if (a_allocation_status != tsl::OkStatus()) {
    return a_allocation_status;
  }

  DeviceMemory<MAPPED_T> b;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> b_temp;
  tsl::Status b_allocation_status = AllocateStridedBuffer<T>(
      b_raw_ptrs, batch_count, batch_stride_b, scratch_allocator, stream,
      &b_temp, &b, true, reallocated_b);
  if (b_allocation_status != tsl::OkStatus()) {
    return b_allocation_status;
  }

  DeviceMemory<MAPPED_T> c;
  std::unique_ptr<TemporaryDeviceMemory<MAPPED_T>> c_temp;
  tsl::Status c_allocation_status = AllocateStridedBuffer<T>(
      c_raw_ptrs, batch_count, batch_stride_c, scratch_allocator, stream,
      &c_temp, &c, true, reallocated_c);  // can disable copy if beta=0
  if (c_allocation_status != tsl::OkStatus()) {
    return c_allocation_status;
  }

  bool ok;
  if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
    float alpha_ = static_cast<float>(alpha);
    float beta_ = static_cast<float>(beta);
    const void *alpha_ptr = reinterpret_cast<const void *>(&alpha_);
    const void *beta_ptr = reinterpret_cast<const void *>(&beta_);

    ok = DoBlasInternal(
        rocblas_func, stream, /* pointer_mode_host = */ true,
        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
        alpha_ptr, a.opaque(), rocblas_datatype_bf16_r, lda, batch_stride_a,
        b.opaque(), rocblas_datatype_bf16_r, ldb, batch_stride_b, beta_ptr,
        c.opaque(), rocblas_datatype_bf16_r, ldc, batch_stride_c, c.opaque(),
        rocblas_datatype_bf16_r, ldc, batch_stride_c, batch_count,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
  } else {
    MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
    MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);
    ok = DoBlasInternal(rocblas_func, stream, /* pointer_mode_host = */ true,
                        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m,
                        n, k, GpuComplex(alpha_ptr), GpuMemory(a), lda,
                        batch_stride_a, GpuMemory(b), ldb, batch_stride_b,
                        GpuComplex(beta_ptr), GpuMemoryMutable(&c), ldc,
                        batch_stride_c, batch_count);
  }
  if (!ok)
    return tsl::Status(absl::StatusCode::kInternal,
                       "failed BLAS call, see log for details");
  if (reallocated_c)
    return ReorganizeMemory(stream, &c, c_raw_ptrs, batch_count, batch_stride_c,
                            false);
  return tsl::OkStatus();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64_t n,
                                 uint64 k, float alpha,
                                 DeviceMemorySlice<Eigen::half> a, int lda,
                                 DeviceMemorySlice<Eigen::half> b, int ldb,
                                 float beta, DeviceMemorySlice<Eigen::half> c,
                                 int ldc, int batch_count,
                                 ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  const Eigen::half alpha_half(alpha);
  const Eigen::half beta_half(beta);

  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_hgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha_half, a, lda, b, ldb, beta_half, c, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, float alpha,
    DeviceMemorySlice<Eigen::bfloat16> a_array, int lda,
    DeviceMemorySlice<Eigen::bfloat16> b_array, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c_array, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  const Eigen::bfloat16 alpha_bf16(alpha);
  const Eigen::bfloat16 beta_bf16(beta);

  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_gemm_strided_batched_ex, stream, transa, transb, m, n, k,
      alpha_bf16, a_array, lda, b_array, ldb, beta_bf16, c_array, ldc,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64_t n,
                                 uint64 k, float alpha,
                                 DeviceMemorySlice<float> a_array, int lda,
                                 DeviceMemorySlice<float> b_array, int ldb,
                                 float beta, DeviceMemorySlice<float> c_array,
                                 int ldc, int batch_count,
                                 ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_sgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(Stream *stream, blas::Transpose transa,
                                 blas::Transpose transb, uint64_t m, uint64_t n,
                                 uint64 k, double alpha,
                                 DeviceMemorySlice<double> a_array, int lda,
                                 DeviceMemorySlice<double> b_array, int ldb,
                                 double beta, DeviceMemorySlice<double> c_array,
                                 int ldc, int batch_count,
                                 ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_dgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<float> alpha,
    DeviceMemorySlice<std::complex<float>> a_array, int lda,
    DeviceMemorySlice<std::complex<float>> b_array, int ldb,
    std::complex<float> beta, DeviceMemorySlice<std::complex<float>> c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_cgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}


bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, std::complex<double> alpha,
    DeviceMemorySlice<std::complex<double>> a_array, int lda,
    DeviceMemorySlice<std::complex<double>> b_array, int ldb,
    std::complex<double> beta, DeviceMemorySlice<std::complex<double>> c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  blas_log("DoBlasGemmBatched");
  tsl::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_zgemm_strided_batched, stream, transa, transb, m, n, k,
      alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_strsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  blas_log("DoBlasTrsm");
  return DoBlasInternal(wrap::rocblas_dtrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(a), lda,
                        GpuMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return DoBlasInternal(wrap::rocblas_ctrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64_t m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return DoBlasInternal(wrap::rocblas_ztrsm, stream,
                        /* pointer_mode_host = */ true, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
                        complex_cast(a), lda, complex_cast(b), ldb);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 float alpha, const DeviceMemory<float *> &as,
                                 int lda, DeviceMemory<float *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(wrap::rocblas_strsm_batched, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 double alpha, const DeviceMemory<double *> &as,
                                 int lda, DeviceMemory<double *> *bs, int ldb,
                                 int batch_count) {
  return DoBlasInternal(wrap::rocblas_dtrsm_batched, stream,
                        true /* = pointer_mode_host */, ROCMBlasSide(side),
                        ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
                        ROCMBlasDiagonal(diag), m, n, &alpha, GpuMemory(as),
                        lda, GpuMemoryMutable(bs), ldb, batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<float> alpha,
                                 const DeviceMemory<std::complex<float> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<float> *> *bs,
                                 int ldb, int batch_count) {
  return DoBlasInternal(
      wrap::rocblas_ctrsm_batched, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
      static_cast<const rocblas_float_complex *const *>(as.opaque()), lda,
      static_cast<rocblas_float_complex *const *>(bs->opaque()), ldb,
      batch_count);
}

bool ROCMBlas::DoBlasTrsmBatched(Stream *stream, blas::Side side,
                                 blas::UpperLower uplo, blas::Transpose transa,
                                 blas::Diagonal diag, uint64_t m, uint64 n,
                                 std::complex<double> alpha,
                                 const DeviceMemory<std::complex<double> *> &as,
                                 int lda,
                                 DeviceMemory<std::complex<double> *> *bs,
                                 int ldb, int batch_count) {
  return DoBlasInternal(
      wrap::rocblas_ztrsm_batched, stream, true /* = pointer_mode_host */,
      ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
      ROCMBlasDiagonal(diag), m, n, complex_cast(alpha),
      static_cast<const rocblas_double_complex *const *>(as.opaque()), lda,
      static_cast<rocblas_double_complex *const *>(bs->opaque()), ldb,
      batch_count);
}

tsl::Status ROCMBlas::DoBlasGemmStridedBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64 k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, int64_t stride_a,
    const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
    blas::ComputePrecision precision) {
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS SGEMM Strided Batched<float>: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);

  switch (dtype) {
    case blas::DataType::kHalf: {
      const Eigen::half alpha_half(*static_cast<const float *>(alpha));
      const Eigen::half beta_half(*static_cast<const float *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_hgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const rocblas_half *>(&alpha_half),
          reinterpret_cast<const rocblas_half *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const rocblas_half *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const rocblas_half *>(&beta_half),
          reinterpret_cast<rocblas_half *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    case blas::DataType::kBF16:
      return DoBlasInternalStatus(
          wrap::rocblas_gemm_strided_batched_ex, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, alpha,
          a.opaque(), rocblas_datatype_bf16_r, lda, stride_a, b.opaque(),
          rocblas_datatype_bf16_r, ldb, stride_b, beta, c->opaque(),
          rocblas_datatype_bf16_r, ldc, stride_c, c->opaque(),
          rocblas_datatype_bf16_r, ldc, stride_c, batch_count,
          rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
    case blas::DataType::kFloat:
      return DoBlasInternalStatus(
          wrap::rocblas_sgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const float *>(alpha),
          reinterpret_cast<const float *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const float *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const float *>(beta),
          reinterpret_cast<float *>(c->opaque()), ldc, stride_c, batch_count);
    case blas::DataType::kDouble:
      return DoBlasInternalStatus(
          wrap::rocblas_dgemm_strided_batched, stream,
          false, /* pointer_mode_host */
          ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
          reinterpret_cast<const double *>(alpha),
          reinterpret_cast<const double *>(a.opaque()), lda, stride_a,
          reinterpret_cast<const double *>(b.opaque()), ldb, stride_b,
          reinterpret_cast<const double *>(beta),
          reinterpret_cast<double *>(c->opaque()), ldc, stride_c, batch_count);
    case blas::DataType::kComplexFloat: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<float> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<float> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_cgemm_strided_batched, stream,
          /* pointer_mode_host = */ true, ROCMBlasTranspose(transa),
          ROCMBlasTranspose(transb), m, n, k, cb_alpha,
          static_cast<const rocblas_float_complex *>(a.opaque()), lda, stride_a,
          static_cast<const rocblas_float_complex *>(b.opaque()), ldb, stride_b,
          cb_beta, static_cast<rocblas_float_complex *>(c->opaque()), ldc,
          stride_c, batch_count);
    }
    case blas::DataType::kComplexDouble: {
      auto cb_alpha =
          complex_cast(*static_cast<const std::complex<double> *>(alpha));
      auto cb_beta =
          complex_cast(*static_cast<const std::complex<double> *>(beta));
      return DoBlasInternalStatus(
          wrap::rocblas_zgemm_strided_batched, stream,
          /* pointer_mode_host = */ true, ROCMBlasTranspose(transa),
          ROCMBlasTranspose(transb), m, n, k, cb_alpha,
          static_cast<const rocblas_double_complex *>(a.opaque()), lda,
          stride_a, static_cast<const rocblas_double_complex *>(b.opaque()),
          ldb, stride_b, cb_beta,
          static_cast<rocblas_double_complex *>(c->opaque()), ldc, stride_c,
          batch_count);
    }
    default:
      return tsl::errors::Internal(absl::StrCat(
          "Unsupported datatype for GEMM: ", blas::DataTypeString(dtype)));
  }
}

tsl::Status ROCMBlas::GetVersion(string *version) {
  return tsl::errors::Unimplemented("");
}

}  // namespace gpu

void initialize_rocblas() {
  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);

  if (!rocBlasAlreadyRegistered) {
    tsl::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, gpu::kRocBlasPlugin, "rocBLAS",
                [](internal::StreamExecutorInterface *parent)
                    -> blas::BlasSupport * {
                  gpu::GpuExecutor *rocm_executor =
                      dynamic_cast<gpu::GpuExecutor *>(parent);
                  if (rocm_executor == nullptr) {
                    LOG(ERROR)
                        << "Attempting to initialize an instance of the "
                           "rocBLAS "
                        << "support library with a non-ROCM StreamExecutor";
                    return nullptr;
                  }

                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(rocm_executor);
                  if (!blas->Init()) {
                    // Note: Init() will log a more specific error.
                    delete blas;
                    return nullptr;
                  }
                  return blas;
                });

    if (!status.ok()) {
      LOG(ERROR) << "Unable to register rocBLAS factory: " << status.message();
    }

    PluginRegistry::Instance()->SetDefaultFactory(
        rocm::kROCmPlatformId, PluginKind::kBlas, gpu::kRocBlasPlugin);
  }
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
