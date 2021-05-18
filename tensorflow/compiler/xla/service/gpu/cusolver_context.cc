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

// we can't include core/util/gpu_device_functions.h (where these are defined),
// because this file is not compiled with hipcc
#if GOOGLE_CUDA
using gpuFloatComplex = cuFloatComplex;
using gpuDoubleComplex = cuDoubleComplex;
using gpuStream_t = gpuStream_t;
using gpuEvent_t = cudaEvent_t;
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventDestroy cudaEventDestroy
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuFree cudaFree
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_complex.h"

using gpuFloatComplex = hipFloatComplex;
using gpuDoubleComplex = hipDoubleComplex;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using cudaError = int;
using cudaError_t = int;
#define cudaSuccess 0
/*
#define cudaGetLastError hipGetLastError
#define gpuEventRecord hipEventRecord
#define gpuEventDestroy hipEventDestroy
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuFree hipFree
*/
using cublasFillMode_t = rocblas_fill;
using cusolverStatus_t = rocsolver_status;

namespace rocblas_wrap {

using stream_executor::internal::CachedDsoLoader::GetRocblasDsoHandle;
using tensorflow::Env;

#ifdef PLATFORM_GOOGLE
#define ROCBLAS_API_WRAPPER(__name)           \
  struct WrapperShim__##__name {              \
    static const char* kName;                 \
    template <typename... Args>               \
    rocblas_status operator()(Args... args) { \
      return ::__name(args...);               \
    }                                         \
  } __name;                                   \
  const char* WrapperShim__##__name::kName = #__name;

#else

#define ROCBLAS_API_WRAPPER(__name)                                        \
  struct DynLoadShim__##__name {                                           \
    static const char* kName;                                              \
    using FuncPtrT = std::add_pointer<decltype(::__name)>::type;           \
    static void* GetDsoHandle() {                                          \
      auto s = GetRocblasDsoHandle();                                      \
      return s.ValueOrDie();                                               \
    }                                                                      \
    static FuncPtrT LoadOrDie() {                                          \
      void* f;                                                             \
      auto s =                                                             \
          Env::Default()->GetSymbolFromLibrary(GetDsoHandle(), kName, &f); \
      CHECK(s.ok()) << "could not find " << kName                          \
                    << " in rocblas DSO; dlerror: " << s.error_message();  \
      return reinterpret_cast<FuncPtrT>(f);                                \
    }                                                                      \
    static FuncPtrT DynLoad() {                                            \
      static FuncPtrT f = LoadOrDie();                                     \
      return f;                                                            \
    }                                                                      \
    template <typename... Args>                                            \
    rocblas_status operator()(Args... args) {                              \
      return DynLoad()(args...);                                           \
    }                                                                      \
  } __name;                                                                \
  const char* DynLoadShim__##__name::kName = #__name;

#endif

// clang-format off
#define FOREACH_ROCBLAS_API(__macro)	        \
  __macro(rocblas_create_handle)		\
  __macro(rocblas_destroy_handle)		\
  __macro(rocblas_set_stream)
// clang-format on

FOREACH_ROCBLAS_API(ROCBLAS_API_WRAPPER)

}  // namespace rocblas_wrap

#define cusolverDnCreate rocblas_wrap::rocblas_create_handle
#define cusolverDnSetStream rocblas_wrap::rocblas_set_stream
#define cusolverDnDestroy rocblas_wrap::rocblas_destroy_handle

#define CUBLAS_FILL_MODE_UPPER rocblas_fill_upper
#define CUBLAS_FILL_MODE_LOWER rocblas_fill_lower

#endif



namespace xla {
namespace gpu {

namespace {

// Type traits to get CUDA complex types from std::complex<T>.
template <typename T>
struct CUDAComplexT {
  typedef T type;
};
#if GOOGLE_CUDA
template <>
struct CUDAComplexT<std::complex<float>> {
  typedef cuComplex type;
};
template <>
struct CUDAComplexT<std::complex<double>> {
  typedef cuDoubleComplex type;
};
#else
// can't use gpuFloatComplex, gpuDoubleComplex, because e.g.
// hipFloatComplex and rocblas_float_complex are two unrelated types
template <>
struct CUDAComplexT<std::complex<float>> {
  typedef rocblas_float_complex type;
};
template <>
struct CUDAComplexT<std::complex<double>> {
  typedef rocblas_double_complex type;
};
#endif

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

#if GOOGLE_CUDA
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
#else
// Converts a cuSolver status to a Status.
Status CusolverStatusToStatus(rocblas_status status) {
  switch(status) {
    case rocblas_status_success:
      return Status::OK();
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
#endif

}  // namespace

StatusOr<CusolverContext> CusolverContext::Create(se::Stream* stream) {
  cusolverDnHandle_t handle;
  TF_RETURN_IF_ERROR(CusolverStatusToStatus(cusolverDnCreate(&handle)));
  CusolverContext context(stream, handle);

  if (stream) {
    // StreamExecutor really should just expose the Cuda stream to clients...
    const gpuStream_t* cuda_stream =
        CHECK_NOTNULL(reinterpret_cast<const gpuStream_t*>(
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

#if GOOGLE_CUDA
#define CALL_LAPACK_TYPES(m) \
  m(float, S) m(double, D) m(std::complex<float>, C) m(std::complex<double>, Z)

#define DN_SOLVER_FN(method, type_prefix) cusolverDn##type_prefix##method

#else

#define CALL_LAPACK_TYPES(m)						\
  m(float, s) m(double, d) m(std::complex<float>, c) m(std::complex<double>, z)

#define DN_SOLVER_FN(method, type_prefix) \
  tensorflow::wrap::rocsolver_##type_prefix##method

#endif
// Note: NVidia have promised that it is safe to pass 'nullptr' as the argument
// buffers to cuSolver buffer size methods and this will be a documented
// behavior in a future cuSolver release.
StatusOr<int64> CusolverContext::PotrfBufferSize(PrimitiveType type,
                                                 se::blas::UpperLower uplo,
                                                 int n, int lda) {
#if GOOGLE_CUDA  
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
#else
  return 0;
#endif  
}

#if GOOGLE_CUDA
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
#else
#define POTRF_INSTANCE(T, type_prefix)                                    \
  template <>                                                             \
  Status CusolverContext::Potrf<T>(                                       \
      se::blas::UpperLower uplo, int n, se::DeviceMemory<T> A, int lda,   \
      se::DeviceMemory<int> lapack_info, se::DeviceMemory<T> workspace) { \
    return CusolverStatusToStatus(DN_SOLVER_FN(potrf, type_prefix)(       \
        handle(), CUDABlasUpperLower(uplo), n, ToDevicePointer(A), lda,   \
        ToDevicePointer(lapack_info)));                                   \
  }
#endif

CALL_LAPACK_TYPES(POTRF_INSTANCE);

}  // namespace gpu
}  // namespace xla
