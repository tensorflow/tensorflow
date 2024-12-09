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

#include "xla/stream_executor/rocm/rocm_blas.h"

#define EIGEN_USE_GPU
#define EIGEN_USE_HIP

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "rocm/include/hip/amd_detail/hip_fp16_gcc.h"
#include "rocm/include/hipblas/hipblas.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/numeric_options.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/stream_executor/plugin_registry.h"
#include "xla/stream_executor/rocm/rocblas_wrapper.h"
#include "xla/stream_executor/rocm/rocm_complex_converters.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/util/determinism.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"

using tsl::OpDeterminismRequired;

namespace stream_executor {
namespace gpu {

using rocm::ROCMComplex;

extern void rocm_Broadcast_fp32(void *stream, float *dst, int dst_stride,
                                int batches, int src_batches, float *src,
                                int size);

template <class T>
const RocBlasType_t<T> *const *complex_cast(const DeviceMemory<T *> &a) {
  return reinterpret_cast<const RocBlasType_t<T> *const *>(GpuMemory(a));
}

template <class T>
RocBlasType_t<T> *const *complex_cast(DeviceMemory<T *> &a) {
  return reinterpret_cast<RocBlasType_t<T> *const *>(GpuMemory(a));
}

template <class T>
const RocBlasType_t<T> *complex_cast(const DeviceMemory<T> &a) {
  return reinterpret_cast<const RocBlasType_t<T> *>(GpuMemory(a));
}

template <class T>
const RocBlasType_t<T> *complex_cast(const T &a) {
  return reinterpret_cast<const RocBlasType_t<T> *>(&a);
}
template <class T>
RocBlasType_t<T> *complex_cast(DeviceMemory<T> *a) {
  return reinterpret_cast<RocBlasType_t<T> *>(GpuMemoryMutable(a));
}

static std::string ToString(rocblas_status status) {
#define XVAL(x) \
  case x:       \
    return #x
  switch (status) {
    XVAL(rocblas_status_success);
    XVAL(rocblas_status_invalid_handle);
    XVAL(rocblas_status_not_implemented);
    XVAL(rocblas_status_invalid_pointer);
    XVAL(rocblas_status_invalid_size);
    XVAL(rocblas_status_memory_error);
    XVAL(rocblas_status_internal_error);
#if TF_ROCM_VERSION >= 60000
    XVAL(rocblas_status_perf_degraded);
    XVAL(rocblas_status_size_query_mismatch);
    XVAL(rocblas_status_size_increased);
    XVAL(rocblas_status_size_unchanged);
    XVAL(rocblas_status_invalid_value);
    XVAL(rocblas_status_continue);
    XVAL(rocblas_status_check_numerics_fail);
    XVAL(rocblas_status_excluded_from_build);
    XVAL(rocblas_status_arch_mismatch);
#endif
    default:
      return absl::StrCat("<invalid rocBLAS status: ", status, ">");
  }
#undef XVAL
}

bool ROCMBlas::Init() {
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  rocblas_status ret = wrap::rocblas_create_handle(&blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

#if TF_HIPBLASLT
  if (!blas_lt_.Init().ok()) {
    LOG(ERROR) << "Failed to initialize hipblasLt";
    return false;
  }
#endif

  int dev = 0;
  hipError_t result = hipGetDevice(&dev);
  hipDeviceProp_t props;
  result = hipGetDeviceProperties(&props, dev);
  if (result == hipSuccess) {
    auto cap = RocmComputeCapability(props.gcnArchName);
    has_mfma_ = cap.has_mfma_instr_support();
    use_hgemm_alt_impl_ = (cap.gfx_version() == "gfx90a");
  }

  return true;
}

ROCMBlas::ROCMBlas(StreamExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)),
      blas_(nullptr)
#if TF_HIPBLASLT
      ,
      blas_lt_(parent)
#endif
{
}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    std::unique_ptr<ActivateContext> activation = parent_->Activate();
    wrap::rocblas_destroy_handle(blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(blas_ != nullptr);
  auto handle =
      (stream != nullptr)
          ? static_cast<hipStream_t>(stream->platform_specific_handle().stream)
          : nullptr;
  if (auto ret = wrap::rocblas_set_stream(blas_, handle);
      ret != rocblas_status_success) {
    LOG(ERROR) << "failed to set stream for rocBLAS calls: " << ToString(ret);
    return false;
  }
  return true;
}

absl::StatusOr<bool> ROCMBlas::IsMainStreamSet() const {
  absl::MutexLock lock{&mu_};
  CHECK(blas_ != nullptr);
  hipStream_t handle{};
  if (auto ret = wrap::rocblas_get_stream(blas_, &handle);
      ret != rocblas_status_success) {
    return absl::InternalError("failed to get the current stream value");
  }
  return (handle == nullptr);
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

int DtypeSize(blas::DataType type) {
  switch (type) {
    case blas::DataType::kHalf:
    case blas::DataType::kBF16:
      return 2;
    case blas::DataType::kFloat:
      return 4;
    case blas::DataType::kDouble:
      return 8;
    case blas::DataType::kInt8:
      return 1;
    case blas::DataType::kComplexFloat:
      return 8;
    case blas::DataType::kComplexDouble:
      return 16;
    default:
      return 0;
  }
}

absl::StatusOr<rocblas_datatype> AsRocBlasType(blas::DataType type) {
  switch (type) {
    case blas::DataType::kHalf:
      return rocblas_datatype_f16_r;
    case blas::DataType::kBF16:
      return rocblas_datatype_bf16_r;
    case blas::DataType::kFloat:
      return rocblas_datatype_f32_r;
    case blas::DataType::kDouble:
      return rocblas_datatype_f64_r;
    case blas::DataType::kInt8:
      return rocblas_datatype_i8_r;
    case blas::DataType::kInt32:
      return rocblas_datatype_i32_r;
    case blas::DataType::kComplexFloat:
      return rocblas_datatype_f32_c;
    case blas::DataType::kComplexDouble:
      return rocblas_datatype_f64_c;
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported blas data type: %d", (int)type));
  }
}

absl::StatusOr<rocblas_datatype> AsRocBlasComputeType(
    blas::ComputationType type) {
  switch (type) {
    case blas::ComputationType::kF16:
      return rocblas_datatype_f16_r;
    case blas::ComputationType::kF32:
      return rocblas_datatype_f32_r;
    case blas::ComputationType::kF64:
      return rocblas_datatype_f64_r;
    case blas::ComputationType::kI32:
      return rocblas_datatype_i32_r;
    case blas::ComputationType::kF16AsF32:
    case blas::ComputationType::kBF16AsF32:
    case blas::ComputationType::kTF32AsF32:
    default:
      return absl::InternalError(
          absl::StrFormat("Unsupported compute type: %d", (int)type));
  }
}

void CheckPreconditions(blas::Transpose transa, blas::Transpose transb,
                        uint64_t m, uint64_t n, uint64_t k,
                        blas::DataType dtype, int lda, int ldb) {
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
}

uint32_t GemmFloat16Flags(blas::DataType dtype, blas::CallContext context,
                          bool use_alt_impl) {
  bool is_backprop = (context == blas::CallContext::kBackpropInput1 ||
                      context == blas::CallContext::kBackpropInput2);

  return ((dtype == blas::DataType::kHalf) && is_backprop && use_alt_impl)
             ? rocblas_gemm_flags_fp16_alt_impl
             : rocblas_gemm_flags_none;
}

absl::Status PopulateProfileFromTimer(
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

}  // namespace

template <typename FuncT, typename... Args>
absl::Status ROCMBlas::DoBlasInternalImpl(FuncT rocblas_func, Stream *stream,
                                          bool pointer_mode_host,
                                          bool err_on_failure, Args &&...args) {
  absl::MutexLock lock{&mu_};

  CHECK(blas_ != nullptr);
  std::unique_ptr<ActivateContext> activation = parent_->Activate();
  if (!SetStream(stream)) {
    return absl::InternalError("Setting stream failed");
  }

  rocblas_status ret;
  // set the atomics mode, leaving default to library
  bool allow_atomics = !OpDeterminismRequired();
  if (!allow_atomics) {
    ret = wrap::rocblas_set_atomics_mode(blas_, rocblas_atomics_not_allowed);
    if (err_on_failure && ret != rocblas_status_success) {
      LOG(ERROR) << "failed to set atomics mode before " << FuncT::kName << ": "
                 << ToString(ret);
    }
  }
#if 0
// pemeliya: the feature is disabled since rocblas does not perform well under
// graph capture. rocblas_set_workspace seems to use blocking memory functions
// like hipFree/hipMalloc which result in HIP_ERROR_StreamCaptureUnsupported
  {
    auto *workspace = GetWorkspace();
    auto *wptr = workspace != nullptr ? workspace->opaque() : nullptr;
    size_t wsize = workspace != nullptr ? workspace->size() : 0;
    ret = wrap::rocblas_set_workspace(blas_, wptr, wsize);
    if (err_on_failure && ret != rocblas_status_success) {
      LOG(ERROR) << "failed to set workspace before " << FuncT::kName
                 << ": " << ToString(ret);
    }
  }
#endif

  ret = rocblas_func(blas_, std::forward<Args>(args)...);
  SetStream(nullptr);  // Resetting stream after the function call

  if (ret != rocblas_status_success) {
    auto err_str =
        absl::StrFormat("%s failed with: %s", FuncT::kName, ToString(ret));
    if (err_on_failure) {
      LOG(ERROR) << err_str;
    }
    return absl::InternalError(err_str);
  }
  return absl::OkStatus();
}

#define Impl_DoBlasScal(Fun, T, Ta)                                         \
  bool ROCMBlas::DoBlasScal(Stream *stream, uint64_t elem_count, Ta alpha,  \
                            DeviceMemory<T> *x, int incx) {                 \
    return DoBlasInternal(Fun, stream, /* pointer_mode_host = */ true,      \
                          elem_count, complex_cast(alpha), complex_cast(x), \
                          incx);                                            \
  }

Impl_DoBlasScal(wrap::rocblas_sscal, float, float)
    Impl_DoBlasScal(wrap::rocblas_dscal, double, double)
        Impl_DoBlasScal(wrap::rocblas_csscal, std::complex<float>, float)
            Impl_DoBlasScal(wrap::rocblas_zdscal, std::complex<double>, double)
                Impl_DoBlasScal(wrap::rocblas_cscal, std::complex<float>,
                                std::complex<float>)
                    Impl_DoBlasScal(wrap::rocblas_zscal, std::complex<double>,
                                    std::complex<double>)
#define Impl_DoBlasGemv(fun, T)                                                \
  bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64_t m, \
                            uint64_t n, T alpha, const DeviceMemory<T> &a,     \
                            int lda, const DeviceMemory<T> &x, int incx,       \
                            T beta, DeviceMemory<T> *y, int incy) {            \
    return DoBlasInternal(fun, stream, /* pointer_mode_host = */ true,         \
                          ROCMBlasTranspose(trans), m, n, complex_cast(alpha), \
                          complex_cast(a), lda, complex_cast(x), incx,         \
                          complex_cast(beta), complex_cast(y), incy);          \
  }

                        Impl_DoBlasGemv(wrap::rocblas_sgemv, float)
                            Impl_DoBlasGemv(wrap::rocblas_dgemv, double)
                                Impl_DoBlasGemv(wrap::rocblas_cgemv,
                                                std::complex<float>)
                                    Impl_DoBlasGemv(wrap::rocblas_zgemv,
                                                    std::complex<double>)

    /**
     *
     *  ALPHA/BETA TYPES
     *
     * For half and bf16, alpha and beta point to floats.
     * For all other types, alpha and beta point to values of the same type as
     *a/b/c.
     *
     * On the rocblas side, non-ex functions expect the same type as a/b/c
     *    (this seems to be a deviation from the blas standard);
     *    and ex functions expect the same type as the compute type (i.e.
     *floats.)
     *
     **/
    using GemmCallTrace = StreamExecutor::GemmCallTrace;

// Log the GEMM operation if the logging mode is enabled.
void ROCMBlas::MaybeLogGemmOp(GemmCallTrace::GemmType op,
                              blas::CallContext context, uint64_t size1,
                              uint64_t size2) {
  auto status =
      parent_->RecordApiTrace(GemmCallTrace{op, (int)context, size1, size2});
}

absl::Status ROCMBlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, blas::DataType dtype, const void *alpha,
    const DeviceMemoryBase &a, int lda, const DeviceMemoryBase &b, int ldb,
    const void *beta, DeviceMemoryBase *c, int ldc,
    const NumericOptions &numeric_options, blas::CallContext context) {
  MaybeLogGemmOp(GemmCallTrace::GemmType::kPlain, context,
                 m * k * DtypeSize(dtype), n * k * DtypeSize(dtype));

  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc);

  CheckPreconditions(transa, transb, m, n, k, dtype, lda, ldb);

  absl::Status status;
  uint32_t gemm_ex_flags = rocblas_gemm_flags_none;
  bool is_backprop = (context == blas::CallContext::kBackpropInput1) ||
                     (context == blas::CallContext::kBackpropInput2);
  if (is_backprop && use_hgemm_alt_impl_)
    gemm_ex_flags = rocblas_gemm_flags_fp16_alt_impl;

  Eigen::half alpha_half, beta_half;

  const void *alpha_downcast = alpha, *beta_downcast = beta;
  if (dtype == blas::DataType::kHalf) {
    alpha_half = Eigen::half(*static_cast<const float *>(alpha));
    beta_half = Eigen::half(*static_cast<const float *>(beta));
    alpha_downcast = &alpha_half;
    beta_downcast = &beta_half;
  }

  /* I would like to specify the type with a template parameter:
   *
   * auto call_gemm = [&]<class type>(auto func) { ... }
   * ...
   * status = call_gemm<float>(wrap::rocblas_sgemm);
   *
   * but that's a C++20 extension and can't be enabled (the compiler does
   * support it, but enabling it causes compilation errors inside Eigen.) */
  auto call_gemm = [&](auto func, auto type) {
    return DoBlasInternalStatus(
        func, stream, /* pointer_mode_host = */ true, ROCMBlasTranspose(transa),
        ROCMBlasTranspose(transb), m, n, k,
        reinterpret_cast<const decltype(type) *>(alpha_downcast),
        reinterpret_cast<const decltype(type) *>(a.opaque()), lda,
        reinterpret_cast<const decltype(type) *>(b.opaque()), ldb,
        reinterpret_cast<const decltype(type) *>(beta_downcast),
        reinterpret_cast<decltype(type) *>(c->opaque()), ldc);
  };

  auto call_gemm_ex = [&](rocblas_datatype dt) {
    return DoBlasInternalStatus(
        wrap::rocblas_gemm_ex, stream, /* pointer_mode_host = */ true,
        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), (rocblas_int)m,
        (rocblas_int)n, (rocblas_int)k, alpha, a.opaque(), dt, lda, b.opaque(),
        dt, ldb, beta, c->opaque(), dt, ldc, c->opaque(), dt, ldc,
        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, gemm_ex_flags);
  };

  switch (dtype) {
    case blas::DataType::kHalf:
      if (has_mfma_)
        return call_gemm_ex(rocblas_datatype_f16_r);
      else
        return call_gemm(wrap::rocblas_hgemm, rocblas_half());
    case blas::DataType::kBF16:
      return call_gemm_ex(rocblas_datatype_bf16_r);
    case blas::DataType::kFloat:
      return call_gemm(wrap::rocblas_sgemm, 1.0f);
    case blas::DataType::kDouble:
      return call_gemm(wrap::rocblas_dgemm, 1.0);
    case blas::DataType::kComplexFloat:
      return call_gemm(wrap::rocblas_cgemm, rocblas_float_complex());
    case blas::DataType::kComplexDouble:
      return call_gemm(wrap::rocblas_zgemm, rocblas_double_complex());
    default:
      return absl::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

absl::Status ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, const void *beta, DeviceMemoryBase *c,
    blas::DataType type_c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *profile_result, blas::CallContext context) {
  if (type_a != type_b) {
    return absl::InternalError(absl::StrFormat(
        "DoBlasGemmWithAlgorithm: different "
        "datatypes for the inputs a (%d) and b (%d) are unsupported",
        static_cast<int>(type_a), static_cast<int>(type_b)));
  }
  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  // fall back to the default implementation
  if (algorithm == blas::kDefaultAlgorithm && type_a == type_c) {
    TF_RETURN_IF_ERROR(DoBlasGemm(stream, transa, transb, m, n, k, type_a,
                                  alpha, a, lda, b, ldb, beta, c, ldc,
                                  numeric_options, context));

  } else {
    MaybeLogGemmOp(GemmCallTrace::GemmType::kPlain, context,
                   m * k * DtypeSize(type_a), n * k * DtypeSize(type_a));
    CheckPreconditions(transa, transb, m, n, k, type_a, lda, ldb);
    TF_ASSIGN_OR_RETURN(auto roc_type_a, AsRocBlasType(type_a));
    TF_ASSIGN_OR_RETURN(auto roc_type_c, AsRocBlasType(type_c));
    TF_ASSIGN_OR_RETURN(auto roc_comp_type,
                        AsRocBlasComputeType(computation_type));

    VLOG(1) << absl::StreamFormat(
        "doing rocBLAS GEMM with Algorithm: at=%d bt=%d m=%u n=%u "
        "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
        "c=%p ldc=%d algorithm=%d type_a/b=%d type_c=%d comp_type=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
        a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc, algorithm,
        static_cast<int>(roc_type_a), static_cast<int>(roc_type_c),
        static_cast<int>(roc_comp_type));

    TF_RETURN_IF_ERROR(DoBlasInternalImpl(
        wrap::rocblas_gemm_ex, stream,
        /* pointer_mode_host = */ true,
        /* err_on_failure = */ false, ROCMBlasTranspose(transa),
        ROCMBlasTranspose(transb), (rocblas_int)m, (rocblas_int)n,
        (rocblas_int)k, alpha, a.opaque(), roc_type_a, lda, b.opaque(),
        roc_type_a, ldb, beta, c->opaque(), roc_type_c, ldc, c->opaque(),
        roc_type_c, ldc, roc_comp_type, rocblas_gemm_algo_solution_index,
        algorithm, GemmFloat16Flags(type_a, context, use_hgemm_alt_impl_)));
  }
  TF_RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, profile_result));

  return absl::OkStatus();
}

absl::Status ROCMBlas::DoBlasGemmStridedBatchedWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, const void *alpha, const DeviceMemoryBase &a,
    blas::DataType type_a, int lda, int64_t stride_a, const DeviceMemoryBase &b,
    blas::DataType type_b, int ldb, int64_t stride_b, const void *beta,
    DeviceMemoryBase *c, blas::DataType type_c, int ldc, int64_t stride_c,
    int batch_count, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, const NumericOptions &numeric_options,
    blas::ProfileResult *profile_result, blas::CallContext context) {
  if (type_a != type_b) {
    return absl::InternalError(absl::StrFormat(
        "DoBlasGemmStridedBatchedWithAlgorithm: different "
        "datatypes for the inputs a (%d) and b (%d) are unsupported",
        static_cast<int>(type_a), static_cast<int>(type_b)));
  }
  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  // fall back to the default implementation
  if (algorithm == blas::kDefaultAlgorithm && type_a == type_c) {
    TF_RETURN_IF_ERROR(DoBlasGemmStridedBatched(
        stream, transa, transb, m, n, k, type_a, alpha, a, lda, stride_a, b,
        ldb, stride_b, beta, c, ldc, stride_c, batch_count, numeric_options,
        context));
  } else {
    MaybeLogGemmOp(GemmCallTrace::GemmType::kStridedBatched, context, a.size(),
                   b.size());
    VLOG(1) << absl::StreamFormat(
        "doing rocBLAS GEMM strided batched with Algorithm: at=%d bt=%d m=%u "
        "n=%u "
        "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
        "c=%p ldc=%d algorithm=%d type_a/b=%d type_c=%d stride_a/b/c=%d/%d/%d "
        "batch_count=%d",
        static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
        a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc, algorithm,
        static_cast<int>(type_a), static_cast<int>(type_c), stride_a, stride_b,
        stride_c, batch_count);

    TF_ASSIGN_OR_RETURN(auto roc_type_a, AsRocBlasType(type_a));
    TF_ASSIGN_OR_RETURN(auto roc_type_c, AsRocBlasType(type_c));
    TF_ASSIGN_OR_RETURN(auto roc_comp_type,
                        AsRocBlasComputeType(computation_type));

    TF_RETURN_IF_ERROR(DoBlasInternalImpl(
        wrap::rocblas_gemm_strided_batched_ex, stream,
        /* pointer_mode_host = */ true,
        /* err_on_failure = */ false, ROCMBlasTranspose(transa),
        ROCMBlasTranspose(transb), (rocblas_int)m, (rocblas_int)n,
        (rocblas_int)k, alpha, a.opaque(), roc_type_a, lda, stride_a,
        b.opaque(), roc_type_a, ldb, stride_b, beta, c->opaque(), roc_type_c,
        ldc, stride_c, c->opaque(), roc_type_c, ldc, stride_c, batch_count,
        roc_comp_type, rocblas_gemm_algo_solution_index, algorithm,
        GemmFloat16Flags(type_a, context, use_hgemm_alt_impl_)));
  }
  TF_RETURN_IF_ERROR(
      PopulateProfileFromTimer(timer.get(), algorithm, profile_result));

  return absl::OkStatus();
}

template <class Lambda>
struct NameWrap : Lambda {
  using Lambda::operator();
  constexpr static const char *kName = "rocblas_gemm_ex_get_solutions";
};
template <class Func>
NameWrap(Func) -> NameWrap<Func>;

#define ASSIGN_OR_FALSE(lhs, rexpr)                 \
  result = (rexpr);                                 \
  if (TF_PREDICT_FALSE(!result.ok())) return false; \
  lhs = std::move(result).value()

bool ROCMBlas::GetBlasGemmAlgorithms(
    Stream *stream, const gpu::MatrixDescriptor &a,
    const gpu::MatrixDescriptor &b, gpu::OutputMatrixDescriptor *c,
    const void *alpha, const void *beta,
    std::vector<blas::AlgorithmType> *out_algorithms) {
  out_algorithms->clear();
  auto blas_lambda = [this, out_algorithms](auto handle, auto &&blas_func,
                                            auto &&...rest) {
    rocblas_int num_sols = 0;
    // If get_solutions call fails, we still can use the default (fallback)
    // algorithm which is available for almost all number types.
    if (auto ret = blas_func(handle, std::forward<decltype(rest)>(rest)...,
                             nullptr, &num_sols);
        ret == rocblas_status_success) {
      solutions_.resize(num_sols);
      if (ret = blas_func(handle, std::forward<decltype(rest)>(rest)...,
                          solutions_.data(), &num_sols);
          ret != rocblas_status_success) {
        num_sols = 0;
      }
    }
    out_algorithms->resize(num_sols + 1);
    (*out_algorithms)[0] = blas::kDefaultAlgorithm;
    for (rocblas_int i = 0; i < num_sols; i++) {
      (*out_algorithms)[i + 1] = solutions_[i];
    }
    // Sort the list solutions by IDs
    std::sort(out_algorithms->begin() + 1, out_algorithms->end());
    return rocblas_status_success;
  };

  VLOG(1) << absl::StreamFormat(
      "GetBlasAlgorithms: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d type_a/b=%d type_c=%d stride_a/b/c=%d/%d/%d "
      "batch_count=%d",
      static_cast<int>(a.transpose), static_cast<int>(b.transpose), c->m, c->n,
      c->k, alpha, a.data.opaque(), a.leading_dim_stride, b.data.opaque(),
      b.leading_dim_stride, beta, c->data.opaque(), c->leading_dim_stride,
      static_cast<int>(a.type), static_cast<int>(c->type), a.batch_stride,
      b.batch_stride, c->batch_stride, c->batch_size);

  if (a.type != b.type) {
    LOG(ERROR) << "Gemm arguments types differ: no feasible solutions!";
    return false;
  }
  absl::StatusOr<rocblas_datatype> result;
  ASSIGN_OR_FALSE(auto roc_type_a, AsRocBlasType(a.type));
  ASSIGN_OR_FALSE(auto roc_type_c, AsRocBlasType(c->type));
  ASSIGN_OR_FALSE(auto roc_comp_type, AsRocBlasComputeType(c->compute_type));

  if (c->batch_size == 1) {
    return DoBlasInternalFailureOK(
        NameWrap{blas_lambda}, stream, true,
        wrap::rocblas_gemm_ex_get_solutions, ROCMBlasTranspose(a.transpose),
        ROCMBlasTranspose(b.transpose), c->m, c->n, c->k, alpha,
        a.data.opaque(), roc_type_a, a.leading_dim_stride, b.data.opaque(),
        roc_type_a, b.leading_dim_stride, beta, c->data.opaque(), roc_type_c,
        c->leading_dim_stride, c->data.opaque(), roc_type_c,
        c->leading_dim_stride, roc_comp_type, rocblas_gemm_algo_solution_index,
        0);
  }
  return DoBlasInternalFailureOK(
      NameWrap{blas_lambda}, stream, true,
      wrap::rocblas_gemm_strided_batched_ex_get_solutions,
      ROCMBlasTranspose(a.transpose), ROCMBlasTranspose(b.transpose), c->m,
      c->n, c->k, alpha, a.data.opaque(), roc_type_a, a.leading_dim_stride,
      a.batch_stride, b.data.opaque(), roc_type_a, b.leading_dim_stride,
      b.batch_stride, beta, c->data.opaque(), roc_type_c, c->leading_dim_stride,
      c->batch_stride, c->data.opaque(), roc_type_c, c->leading_dim_stride,
      c->batch_stride, c->batch_size, roc_comp_type,
      rocblas_gemm_algo_solution_index, 0);
}
#undef ASSIGN_OR_FALSE

namespace {

struct MemoryCopyOp {
  char *src_ptr;
  char *dst_ptr;
  uint64_t size;
  uint64_t count;
  uint64_t dst_stride;
  uint64_t src_count;
};

// Check whether two Memory Copy Ops can be fold together.
// If it's true, fold it. Otherwise, return false.
bool MemCopyOpsFold(MemoryCopyOp &y, const MemoryCopyOp &x) {
  bool misaligned = (x.size & 3) ||
                    (reinterpret_cast<uint64_t>(x.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(x.src_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.dst_ptr) & 3) ||
                    (reinterpret_cast<uint64_t>(y.src_ptr) & 3);

  int64_t dst_step = reinterpret_cast<int64_t>(x.dst_ptr) -
                     reinterpret_cast<int64_t>(y.dst_ptr);

  if (x.src_ptr == y.src_ptr && x.size == y.size &&
      (y.count == 1 || x.dst_ptr == y.dst_ptr + y.count * y.dst_stride) &&
      !misaligned && y.src_count == 1 && !(dst_step & 3)) {
    if (y.count == 1) {
      y.dst_stride = dst_step;
    }
    y.count++;
    return true;
  } else if (x.src_ptr == y.src_ptr + y.size &&
             x.dst_ptr == y.dst_ptr + y.size && y.count == 1 &&
             y.src_count == 1) {
    y.size += x.size;
    return true;
  }
  if (x.src_ptr == y.src_ptr + y.size * y.src_count &&
      x.dst_ptr == y.dst_ptr + y.dst_stride * y.src_count * y.count &&
      x.count == y.count && x.dst_stride == y.dst_stride) {
    y.src_count += x.src_count;
    return true;
  }
  return false;
}

// This copies from source memory: raw_ptrs[i] to target memory:
// device_memory_ptr at the interval of matrix_byte_size, or vice versa.
// The below algorithm tries to minimize the number of memcpy by consolidating
// neighboring memcpy into a single request.
template <typename MAPPED_T>
absl::Status ReorganizeMemory(Stream *stream,
                              DeviceMemory<MAPPED_T> *device_memory,
                              const std::vector<MAPPED_T *> &raw_ptrs,
                              int batch_count, uint64_t batch_stride,
                              bool gather) {
  if (gather == false) {
    return absl::UnimplementedError("gather=false is unsupported");
  }

  assert(batch_count > 0);
  char *device_memory_ptr = static_cast<char *>(device_memory->opaque());
  char *src_ptr = reinterpret_cast<char *>(raw_ptrs[0]);
  char *dst_ptr = device_memory_ptr;
  size_t matrix_byte_size = batch_stride * sizeof(MAPPED_T);

  std::vector<MemoryCopyOp> mem_copy_ops{
      MemoryCopyOp{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1}};

  for (int i = 1; i < batch_count; ++i) {
    src_ptr = reinterpret_cast<char *>(raw_ptrs[i]);
    dst_ptr = device_memory_ptr + i * matrix_byte_size;

    MemoryCopyOp x{src_ptr, dst_ptr, matrix_byte_size, 1, 0, 1};
    while (mem_copy_ops.size() > 1 &&
           MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                          mem_copy_ops.back())) {
      mem_copy_ops.pop_back();
    }
    MemoryCopyOp &op = mem_copy_ops.back();
    if (MemCopyOpsFold(op, x)) {
      continue;
    }
    mem_copy_ops.push_back(x);
  }

  while (mem_copy_ops.size() > 1 &&
         MemCopyOpsFold(mem_copy_ops[mem_copy_ops.size() - 2],
                        mem_copy_ops.back())) {
    mem_copy_ops.pop_back();
  }

  int i = 0;
  for (auto &x : mem_copy_ops) {
    if (x.src_count > 1 || x.count > 1) {
      rocm_Broadcast_fp32(
          static_cast<hipStream_t>(stream->platform_specific_handle().stream),
          reinterpret_cast<float *>(x.dst_ptr), x.dst_stride >> 2, x.count,
          x.src_count, reinterpret_cast<float *>(x.src_ptr), x.size >> 2);
    } else {
      DeviceMemoryBase src_mem = DeviceMemoryBase(x.src_ptr, x.size);
      DeviceMemoryBase target_mem = DeviceMemoryBase(x.dst_ptr, x.size);
      TF_RETURN_IF_ERROR(stream->Memcpy(&target_mem, src_mem, x.size));
    }
    i++;
  }
  return absl::OkStatus();
}

template <typename T>
struct AllocateStridedResult {
  using Type = RocBlasType_t<T>;
  DeviceMemory<Type> device_mem;
  bool reallocated;
};

// A helper allocation function to convert raw pointers memory layout to
// strided flavor
template <typename T>
absl::StatusOr<AllocateStridedResult<T>> AllocateStridedBuffer(
    const std::vector<RocBlasType_t<T> *> &raw_ptrs, int batch_count,
    uint64_t batch_stride, ScratchAllocator *scratch_allocator, Stream *stream,
    bool copy_data) {
  using MAPPED_T = RocBlasType_t<T>;
  AllocateStridedResult<T> res;

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
    res.device_mem = DeviceMemory<MAPPED_T>(
        DeviceMemoryBase(raw_ptrs[0], matrix_batch_byte_size));
    res.reallocated = false;
    return res;
  }

  if (scratch_allocator == nullptr) {
    return absl::InternalError("scratch_allocator is null");
  }
  TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> batch_matrix_bytes,
                      scratch_allocator->AllocateBytes(matrix_batch_byte_size));
  res.device_mem = DeviceMemory<MAPPED_T>(batch_matrix_bytes);
  res.reallocated = true;
  if (copy_data) {
    TF_RETURN_IF_ERROR(ReorganizeMemory(stream, &res.device_mem, raw_ptrs,
                                        batch_count, batch_stride, true));
  }
  return res;
}

}  // namespace

template <typename T, typename FuncT>
absl::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64_t m, uint64_t n, uint64_t k, T alpha,
    DeviceMemorySlice<T> a_ptrs_to_wrappers, int lda,
    DeviceMemorySlice<T> b_ptrs_to_wrappers, int ldb, T beta,
    DeviceMemorySlice<T> c_ptrs_to_wrappers, int ldc, int batch_count,
    ScratchAllocator *scratch_allocator) {
  using MAPPED_T = RocBlasType_t<T>;

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
  std::vector<MAPPED_T *> a_raw_ptrs(batch_count), b_raw_ptrs(batch_count),
      c_raw_ptrs(batch_count);
  for (int i = 0; i < batch_count; ++i) {
    // static_cast does work when converting Eigen::half* to rocblas_half*,
    // hence the use of reinterpret_cast
    a_raw_ptrs[i] =
        reinterpret_cast<MAPPED_T *>(a_ptrs_to_wrappers[i]->opaque());
    b_raw_ptrs[i] =
        reinterpret_cast<MAPPED_T *>(b_ptrs_to_wrappers[i]->opaque());
    c_raw_ptrs[i] =
        reinterpret_cast<MAPPED_T *>(c_ptrs_to_wrappers[i]->opaque());
  }

  // Make sure the temporary memory are in-scope before the function returns
  TF_ASSIGN_OR_RETURN(
      auto a, AllocateStridedBuffer<T>(a_raw_ptrs, batch_count, batch_stride_a,
                                       scratch_allocator, stream, true));

  TF_ASSIGN_OR_RETURN(
      auto b, AllocateStridedBuffer<T>(b_raw_ptrs, batch_count, batch_stride_b,
                                       scratch_allocator, stream, true));

  TF_ASSIGN_OR_RETURN(
      auto c, AllocateStridedBuffer<T>(c_raw_ptrs, batch_count, batch_stride_c,
                                       scratch_allocator, stream,
                                       true));  // can disable copy if beta=0

  MAPPED_T *alpha_ptr = reinterpret_cast<MAPPED_T *>(&alpha);
  MAPPED_T *beta_ptr = reinterpret_cast<MAPPED_T *>(&beta);
  bool ok = DoBlasInternal(
      rocblas_func, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
      ROCMComplex(alpha_ptr), GpuMemory(a.device_mem), lda, batch_stride_a,
      GpuMemory(b.device_mem), ldb, batch_stride_b, ROCMComplex(beta_ptr),
      GpuMemoryMutable(&c.device_mem), ldc, batch_stride_c, batch_count);

  if (!ok) {
    return absl::Status(absl::StatusCode::kInternal,
                        "failed BLAS call, see log for details");
  }
  if (c.reallocated) {
    return ReorganizeMemory(stream, &c.device_mem, c_raw_ptrs, batch_count,
                            batch_stride_c, false);
  }
  return absl::OkStatus();
}

class rocblas_hgemm_strided_batched_mfma {
  int ALT_;

 public:
  rocblas_hgemm_strided_batched_mfma(int ALT) : ALT_(ALT) {}
  static const char *kName;
  rocblas_status operator()(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k,
                            const rocblas_half *alpha, const rocblas_half *A,
                            rocblas_int lda, rocblas_stride stride_a,
                            const rocblas_half *B, rocblas_int ldb,
                            rocblas_stride stride_b, const rocblas_half *beta,
                            rocblas_half *C, rocblas_int ldc,
                            rocblas_stride stride_c, rocblas_int batch_count) {
    float alpha32 = static_cast<float>(*(const __half *)alpha);
    float beta32 = static_cast<float>(*(const __half *)beta);
    uint32_t flags = rocblas_gemm_flags_none;
    if (ALT_) flags = rocblas_gemm_flags_fp16_alt_impl;
    return wrap::rocblas_gemm_strided_batched_ex(
        handle, transA, transB, m, n, k, &alpha32, A, rocblas_datatype_f16_r,
        lda, stride_a, B, rocblas_datatype_f16_r, ldb, stride_b, &beta32, C,
        rocblas_datatype_f16_r, ldc, stride_c, C, rocblas_datatype_f16_r, ldc,
        stride_c, batch_count, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, flags);
  }
};

const char *rocblas_hgemm_strided_batched_mfma::kName =
    "rocblas_hgemm_strided_batched_mfma";

class rocblas_gemm_strided_batched_bf16 {
 public:
  static const char *kName;
  rocblas_status operator()(rocblas_handle handle, rocblas_operation transA,
                            rocblas_operation transB, rocblas_int m,
                            rocblas_int n, rocblas_int k,
                            const rocblas_bfloat16 *alpha,
                            const rocblas_bfloat16 *A, rocblas_int lda,
                            rocblas_stride stride_a, const rocblas_bfloat16 *B,
                            rocblas_int ldb, rocblas_stride stride_b,
                            const rocblas_bfloat16 *beta, rocblas_bfloat16 *C,
                            rocblas_int ldc, rocblas_stride stride_c,
                            rocblas_int batch_count) {
    float alpha32 = static_cast<float>(*(const Eigen::bfloat16 *)alpha);
    float beta32 = static_cast<float>(*(const Eigen::bfloat16 *)beta);
    uint32_t flags = rocblas_gemm_flags_none;
    return wrap::rocblas_gemm_strided_batched_ex(
        handle, transA, transB, m, n, k, &alpha32, A, rocblas_datatype_bf16_r,
        lda, stride_a, B, rocblas_datatype_bf16_r, ldb, stride_b, &beta32, C,
        rocblas_datatype_bf16_r, ldc, stride_c, C, rocblas_datatype_bf16_r, ldc,
        stride_c, batch_count, rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard, 0, flags);
  }
};

const char *rocblas_gemm_strided_batched_bf16::kName =
    "rocblas_gemm_strided_batched_bf16";
bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha, DeviceMemorySlice<Eigen::half> a,
    int lda, DeviceMemorySlice<Eigen::half> b, int ldb, float beta,
    DeviceMemorySlice<Eigen::half> c, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  MaybeLogGemmOp(GemmCallTrace::GemmType::kBatched, context, a.size(),
                 b.size());
  const Eigen::half alpha_half(alpha);
  const Eigen::half beta_half(beta);
  absl::Status status;

  auto call_gemm = [&](auto x) {
    return DoBlasGemmBatchedInternal(x, stream, transa, transb, m, n, k,
                                     alpha_half, a, lda, b, ldb, beta_half, c,
                                     ldc, batch_count, scratch_allocator);
  };

  if (has_mfma_) {
    bool is_backprop = (context == blas::CallContext::kBackpropInput1) ||
                       (context == blas::CallContext::kBackpropInput2);
    status = call_gemm(
        rocblas_hgemm_strided_batched_mfma(is_backprop && use_hgemm_alt_impl_));
  } else {
    status = call_gemm(wrap::rocblas_hgemm_strided_batched);
  }

  if (!status.ok()) {
    LOG(ERROR) << status;
  }

  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64_t m,
    uint64_t n, uint64_t k, float alpha,
    DeviceMemorySlice<Eigen::bfloat16> a_array, int lda,
    DeviceMemorySlice<Eigen::bfloat16> b_array, int ldb, float beta,
    DeviceMemorySlice<Eigen::bfloat16> c_array, int ldc, int batch_count,
    const NumericOptions &numeric_options, ScratchAllocator *scratch_allocator,
    blas::CallContext context) {
  MaybeLogGemmOp(GemmCallTrace::GemmType::kBatched, context, a_array.size(),
                 b_array.size());
  const Eigen::bfloat16 alpha_bf16(alpha);
  const Eigen::bfloat16 beta_bf16(beta);

  absl::Status status = DoBlasGemmBatchedInternal(
      rocblas_gemm_strided_batched_bf16(), stream, transa, transb, m, n, k,
      alpha_bf16, a_array, lda, b_array, ldb, beta_bf16, c_array, ldc,
      batch_count, scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

#define IMPL_DoBlasGemmBatched(T, Fun)                                         \
  bool ROCMBlas::DoBlasGemmBatched(                                            \
      Stream *stream, blas::Transpose transa, blas::Transpose transb,          \
      uint64_t m, uint64_t n, uint64_t k, T alpha,                             \
      DeviceMemorySlice<T> a_array, int lda, DeviceMemorySlice<T> b_array,     \
      int ldb, T beta, DeviceMemorySlice<T> c_array, int ldc, int batch_count, \
      const NumericOptions &numeric_options,                                   \
      ScratchAllocator *scratch_allocator, blas::CallContext context) {        \
    MaybeLogGemmOp(GemmCallTrace::GemmType::kBatched, context, a_array.size(), \
                   b_array.size());                                            \
    absl::Status status = DoBlasGemmBatchedInternal(                           \
        Fun, stream, transa, transb, m, n, k, alpha, a_array, lda, b_array,    \
        ldb, beta, c_array, ldc, batch_count, scratch_allocator);              \
    if (!status.ok()) {                                                        \
      LOG(ERROR) << status;                                                    \
    }                                                                          \
    return status.ok();                                                        \
  }

IMPL_DoBlasGemmBatched(float, wrap::rocblas_sgemm_strided_batched)
    IMPL_DoBlasGemmBatched(double, wrap::rocblas_dgemm_strided_batched)
        IMPL_DoBlasGemmBatched(std::complex<float>,
                               wrap::rocblas_cgemm_strided_batched)
            IMPL_DoBlasGemmBatched(std::complex<double>,
                                   wrap::rocblas_zgemm_strided_batched)
#define IMPL_DoBlasTrsm(T, Fun, Fun2)                                        \
  bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,                 \
                            blas::UpperLower uplo, blas::Transpose transa,   \
                            blas::Diagonal diag, uint64_t m, uint64_t n,     \
                            T alpha, const DeviceMemory<T> &a, int lda,      \
                            DeviceMemory<T> *b, int ldb) {                   \
    return DoBlasInternal(Fun, stream, /* pointer_mode_host = */ true,       \
                          ROCMBlasSide(side), ROCMBlasUpperLower(uplo),      \
                          ROCMBlasTranspose(transa), ROCMBlasDiagonal(diag), \
                          m, n, complex_cast(alpha), complex_cast(a), lda,   \
                          complex_cast(b), ldb);                             \
  }                                                                          \
                                                                             \
  bool ROCMBlas::DoBlasTrsmBatched(                                          \
      Stream *stream, blas::Side side, blas::UpperLower uplo,                \
      blas::Transpose transa, blas::Diagonal diag, uint64_t m, uint64_t n,   \
      T alpha, const DeviceMemory<T *> &as, int lda, DeviceMemory<T *> *bs,  \
      int ldb, int batch_count) {                                            \
    return DoBlasInternal(Fun2, stream, true /* = pointer_mode_host */,      \
                          ROCMBlasSide(side), ROCMBlasUpperLower(uplo),      \
                          ROCMBlasTranspose(transa), ROCMBlasDiagonal(diag), \
                          m, n, complex_cast(alpha), complex_cast(as), lda,  \
                          complex_cast(*bs), ldb, batch_count);              \
  }

                IMPL_DoBlasTrsm(float, wrap::rocblas_strsm,
                                wrap::rocblas_strsm_batched)
                    IMPL_DoBlasTrsm(double, wrap::rocblas_dtrsm,
                                    wrap::rocblas_dtrsm_batched)
                        IMPL_DoBlasTrsm(std::complex<float>,
                                        wrap::rocblas_ctrsm,
                                        wrap::rocblas_ctrsm_batched)
                            IMPL_DoBlasTrsm(std::complex<double>,
                                            wrap::rocblas_ztrsm,
                                            wrap::rocblas_ztrsm_batched)

                                absl::Status
    ROCMBlas::DoBlasGemmStridedBatched(
        Stream *stream, blas::Transpose transa, blas::Transpose transb,
        uint64_t m, uint64_t n, uint64_t k, blas::DataType dtype,
        const void *alpha, const DeviceMemoryBase &a, int lda, int64_t stride_a,
        const DeviceMemoryBase &b, int ldb, int64_t stride_b, const void *beta,
        DeviceMemoryBase *c, int ldc, int64_t stride_c, int batch_count,
        const NumericOptions &numeric_options, blas::CallContext context) {
  VLOG(1) << absl::StreamFormat(
      "doing rocBLAS GEMM Strided Batched: at=%d bt=%d m=%u n=%u "
      "k=%llu alpha=%p a=%p lda=%d b=%p ldb=%d beta=%p "
      "c=%p ldc=%d stride_a/b/c=%d/%d/%d batch_count=%d",
      static_cast<int>(transa), static_cast<int>(transb), m, n, k, alpha,
      a.opaque(), lda, b.opaque(), ldb, beta, c->opaque(), ldc, stride_a,
      stride_b, stride_c, batch_count);
  MaybeLogGemmOp(GemmCallTrace::GemmType::kStridedBatched, context, a.size(),
                 b.size());

  absl::Status status;
  auto call_gemm = [&](auto func, auto type) {
    return DoBlasInternalStatus(
        func, stream, false, /* pointer_mode_host */
        ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
        reinterpret_cast<const decltype(type) *>(alpha),
        reinterpret_cast<const decltype(type) *>(a.opaque()), lda, stride_a,
        reinterpret_cast<const decltype(type) *>(b.opaque()), ldb, stride_b,
        reinterpret_cast<const decltype(type) *>(beta),
        reinterpret_cast<decltype(type) *>(c->opaque()), ldc, stride_c,
        batch_count);
  };

  switch (dtype) {
    case blas::DataType::kHalf: {
      bool is_backprop = (context == blas::CallContext::kBackpropInput1) ||
                         (context == blas::CallContext::kBackpropInput2);
      Eigen::half alpha_half = Eigen::half(*static_cast<const float *>(alpha));
      Eigen::half beta_half = Eigen::half(*static_cast<const float *>(beta));
      alpha = &alpha_half;
      beta = &beta_half;
      if (has_mfma_) {
        return call_gemm(rocblas_hgemm_strided_batched_mfma(
                             is_backprop && use_hgemm_alt_impl_),
                         rocblas_half());
      } else {
        return call_gemm(wrap::rocblas_hgemm_strided_batched, rocblas_half());
      }
    }
    case blas::DataType::kBF16: {
      Eigen::bfloat16 alpha_bf16, beta_bf16;
      alpha_bf16 = Eigen::bfloat16(*static_cast<const float *>(alpha));
      beta_bf16 = Eigen::bfloat16(*static_cast<const float *>(beta));
      alpha = &alpha_bf16;
      beta = &beta_bf16;
      return call_gemm(rocblas_gemm_strided_batched_bf16(), rocblas_bfloat16());
    }
    case blas::DataType::kFloat:
      return call_gemm(wrap::rocblas_sgemm_strided_batched, 1.0f);
    case blas::DataType::kDouble:
      return call_gemm(wrap::rocblas_dgemm_strided_batched, 1.0);
    case blas::DataType::kComplexFloat:
      return call_gemm(wrap::rocblas_cgemm_strided_batched,
                       rocblas_float_complex());
    case blas::DataType::kComplexDouble:
      return call_gemm(wrap::rocblas_zgemm_strided_batched,
                       rocblas_double_complex());
    default:
      return absl::InternalError(absl::StrCat("Unsupported datatype for GEMM: ",
                                              blas::DataTypeString(dtype)));
  }
}

absl::Status ROCMBlas::GetVersion(std::string *version) {
  absl::MutexLock lock{&mu_};
  size_t len = 0;
  if (auto res = wrap::rocblas_get_version_string_size(&len);
      res != rocblas_status_success) {
    return absl::InternalError(
        absl::StrCat("GetVersion failed with: ", ToString(res)));
  }
  std::vector<char> buf(len + 1);
  if (auto res = wrap::rocblas_get_version_string(buf.data(), len);
      res != rocblas_status_success) {
    return absl::InternalError(
        absl::StrCat("GetVersion failed with: ", ToString(res)));
  }
  *version = std::string(buf.begin(), buf.end());
  return absl::OkStatus();
}

}  // namespace gpu

void initialize_rocblas() {
  auto rocBlasAlreadyRegistered = PluginRegistry::Instance()->HasFactory(
      rocm::kROCmPlatformId, PluginKind::kBlas);

  if (!rocBlasAlreadyRegistered) {
    absl::Status status =
        PluginRegistry::Instance()
            ->RegisterFactory<PluginRegistry::BlasFactory>(
                rocm::kROCmPlatformId, "rocBLAS",
                [](StreamExecutor *parent) -> blas::BlasSupport * {
                  gpu::ROCMBlas *blas = new gpu::ROCMBlas(parent);
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
  }
}

}  // namespace stream_executor

STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(register_rocblas, {
  stream_executor::initialize_rocblas();
});
