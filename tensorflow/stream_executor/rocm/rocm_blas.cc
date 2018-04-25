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

// Include HIPBLAS headers early, and then set EIGEN_HAS_ROCM_FP16
// if we have new enough ROCM (which we will only know after including
// rocm.h). This ensures that Eigen's Half.h does not attempt to make its own
// __half typedef if ROCM has already defined one (and conversely, that we do
// not include <rocm_fp16.h> after Half.h has made its typedef).
#include "rocm/include/hipblas/hipblas.h"

#if ROCM_VERSION >= 7050
#define EIGEN_HAS_ROCM_FP16
#endif

#if ROCM_VERSION >= 8000
#define SE_ROCM_DATA_HALF ROCM_R_16F
#else
#define SE_ROCM_DATA_HALF HIPBLAS_DATA_HALF
#endif

#include "tensorflow/stream_executor/rocm/rocm_blas.h"

#include <assert.h>
#include <complex>

#include "tensorflow/stream_executor/rocm/rocm_activation.h"
#include "tensorflow/stream_executor/rocm/rocm_gpu_executor.h"
#include "tensorflow/stream_executor/rocm/rocm_helpers.h"
#include "tensorflow/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/stream_executor/rocm/rocm_stream.h"
#include "tensorflow/stream_executor/rocm/rocm_timer.h"
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
namespace rocm {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kHipBlasPlugin);

namespace wrap {

// XXX check if we need to port PERFTOOLS_GPU_TOOLS_HIPBLAS_WRAP from hipTensorFlow
#define PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(__name)                      \
  struct WrapperShim__##__name {                                    \
    static const char *kName;                                       \
    template <typename... Args>                                     \
    hipblasStatus_t operator()(ROCMExecutor *parent, Args... args) { \
      rocm::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;                                                         \
  const char *WrapperShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(__name)

#define HIPBLAS_BLAS_ROUTINE_EACH(__macro) \
/*  __macro(hipblasSnrm2)                    \
  __macro(hipblasDnrm2)                    \
  __macro(hipblasScnrm2)                   \
  __macro(hipblasDznrm2)                   */ \
  __macro(hipblasSdot)                     \
  __macro(hipblasDdot)                     \
/*  __macro(hipblasCdotu)                    \
  __macro(hipblasCdotc)                    \
  __macro(hipblasZdotu)                    \
  __macro(hipblasZdotc)                    */ \
  __macro(hipblasSscal)                    \
  __macro(hipblasDscal)                    \
/*  __macro(hipblasCscal)                    \
  __macro(hipblasCsscal)                   \
  __macro(hipblasZscal)                    \
  __macro(hipblasZdscal)                   */ \
  __macro(hipblasSaxpy)                    \
  __macro(hipblasDaxpy)                    \
/*  __macro(hipblasCaxpy)                    \
  __macro(hipblasZaxpy)                    */ \
  __macro(hipblasScopy)                    \
  __macro(hipblasDcopy)                    \
/*  __macro(hipblasCcopy)                    \
  __macro(hipblasZcopy)                    \
  __macro(hipblasSswap)                    \
  __macro(hipblasDswap)                    \
  __macro(hipblasCswap)                    \
  __macro(hipblasZswap)                    \
  __macro(hipblasIsamax)                   \
  __macro(hipblasIdamax)                   \
  __macro(hipblasIcamax)                   \
  __macro(hipblasIzamax)                   \
  __macro(hipblasIsamin)                   \
  __macro(hipblasIdamin)                   \
  __macro(hipblasIcamin)                   \
  __macro(hipblasIzamin)                   */ \
  __macro(hipblasSasum)                    \
  __macro(hipblasDasum)                    \
/*  __macro(hipblasScasum)                   \
  __macro(hipblasDzasum)                   \
  __macro(hipblasSrot)                     \
  __macro(hipblasDrot)                     \
  __macro(hipblasCrot)                     \
  __macro(hipblasCsrot)                    \
  __macro(hipblasZrot)                     \
  __macro(hipblasZdrot)                    \
  __macro(hipblasSrotg)                    \
  __macro(hipblasDrotg)                    \
  __macro(hipblasCrotg)                    \
  __macro(hipblasZrotg)                    \
  __macro(hipblasSrotm)                    \
  __macro(hipblasDrotm)                    \
  __macro(hipblasSrotmg)                   \
  __macro(hipblasDrotmg)                   */ \
  __macro(hipblasSgemv)                    \
  __macro(hipblasDgemv)                    \
/*  __macro(hipblasCgemv)                    \
  __macro(hipblasZgemv)                    \
  __macro(hipblasSgbmv)                    \
  __macro(hipblasDgbmv)                    \
  __macro(hipblasCgbmv)                    \
  __macro(hipblasZgbmv)                    \
  __macro(hipblasStrmv)                    \
  __macro(hipblasDtrmv)                    \
  __macro(hipblasCtrmv)                    \
  __macro(hipblasZtrmv)                    \
  __macro(hipblasStbmv)                    \
  __macro(hipblasDtbmv)                    \
  __macro(hipblasCtbmv)                    \
  __macro(hipblasZtbmv)                    \
  __macro(hipblasStpmv)                    \
  __macro(hipblasDtpmv)                    \
  __macro(hipblasCtpmv)                    \
  __macro(hipblasZtpmv)                    \
  __macro(hipblasStrsv)                    \
  __macro(hipblasDtrsv)                    \
  __macro(hipblasCtrsv)                    \
  __macro(hipblasZtrsv)                    \
  __macro(hipblasStpsv)                    \
  __macro(hipblasDtpsv)                    \
  __macro(hipblasCtpsv)                    \
  __macro(hipblasZtpsv)                    \
  __macro(hipblasStbsv)                    \
  __macro(hipblasDtbsv)                    \
  __macro(hipblasCtbsv)                    \
  __macro(hipblasZtbsv)                    \
  __macro(hipblasSsymv)                    \
  __macro(hipblasDsymv)                    \
  __macro(hipblasCsymv)                    \
  __macro(hipblasZsymv)                    \
  __macro(hipblasChemv)                    \
  __macro(hipblasZhemv)                    \
  __macro(hipblasSsbmv)                    \
  __macro(hipblasDsbmv)                    \
  __macro(hipblasChbmv)                    \
  __macro(hipblasZhbmv)                    \
  __macro(hipblasSspmv)                    \
  __macro(hipblasDspmv)                    \
  __macro(hipblasChpmv)                    \
  __macro(hipblasZhpmv)                    */ \
  __macro(hipblasSger)                     \
/*  __macro(hipblasDger)                     \
  __macro(hipblasCgeru)                    \
  __macro(hipblasCgerc)                    \
  __macro(hipblasZgeru)                    \
  __macro(hipblasZgerc)                    \
  __macro(hipblasSsyr)                     \
  __macro(hipblasDsyr)                     \
  __macro(hipblasCsyr)                     \
  __macro(hipblasZsyr)                     \
  __macro(hipblasCher)                     \
  __macro(hipblasZher)                     \
  __macro(hipblasSspr)                     \
  __macro(hipblasDspr)                     \
  __macro(hipblasChpr)                     \
  __macro(hipblasZhpr)                     \
  __macro(hipblasSsyr2)                    \
  __macro(hipblasDsyr2)                    \
  __macro(hipblasCsyr2)                    \
  __macro(hipblasZsyr2)                    \
  __macro(hipblasCher2)                    \
  __macro(hipblasZher2)                    \
  __macro(hipblasSspr2)                    \
  __macro(hipblasDspr2)                    \
  __macro(hipblasChpr2)                    \
  __macro(hipblasZhpr2)                    */ \
  __macro(hipblasSgemm)                    \
  __macro(hipblasDgemm)                    \
/*  __macro(hipblasCgemm)                    \
  __macro(hipblasZgemm)                    \
  __macro(hipblasSsyrk)                    \
  __macro(hipblasDsyrk)                    \
  __macro(hipblasCsyrk)                    \
  __macro(hipblasZsyrk)                    \
  __macro(hipblasCherk)                    \
  __macro(hipblasZherk)                    \
  __macro(hipblasSsyr2k)                   \
  __macro(hipblasDsyr2k)                   \
  __macro(hipblasCsyr2k)                   \
  __macro(hipblasZsyr2k)                   \
  __macro(hipblasCher2k)                   \
  __macro(hipblasZher2k)                   \
  __macro(hipblasSsyrkx)                   \
  __macro(hipblasDsyrkx)                   \
  __macro(hipblasCsyrkx)                   \
  __macro(hipblasZsyrkx)                   \
  __macro(hipblasCherkx)                   \
  __macro(hipblasZherkx)                   \
  __macro(hipblasSsymm)                    \
  __macro(hipblasDsymm)                    \
  __macro(hipblasCsymm)                    \
  __macro(hipblasZsymm)                    \
  __macro(hipblasChemm)                    \
  __macro(hipblasZhemm)                    \
  __macro(hipblasStrsm)                    \
  __macro(hipblasDtrsm)                    \
  __macro(hipblasCtrsm)                    \
  __macro(hipblasZtrsm)                    \
  __macro(hipblasStrmm)                    \
  __macro(hipblasDtrmm)                    \
  __macro(hipblasCtrmm)                    \
  __macro(hipblasZtrmm)                    \
  __macro(hipblasSgeam)                    \
  __macro(hipblasDgeam)                    \
  __macro(hipblasCgeam)                    \
  __macro(hipblasZgeam)                    \
  __macro(hipblasSdgmm)                    \
  __macro(hipblasDdgmm)                    \
  __macro(hipblasCdgmm)                    \
  __macro(hipblasZdgmm) */

PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasCreate)
PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasDestroy)
PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasSetStream)
//PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasSetPointerMode)
//PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP(hipblasGetPointerMode)
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasSgemmBatched)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasSgemmStridedBatched)
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasDgemmBatched)
PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasDgemmStridedBatched)
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasCgemmBatched)
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasZgemmBatched)
HIPBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_HIPBLAS_V2_WRAP)

#if ROCM_VERSION >= 7050
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasSgemmEx)
#endif

#if ROCM_VERSION >= 8000
//PERFTOOLS_GPUTOOLS_HIPBLAS_WRAP(hipblasGemmEx)
#endif

}  // namespace wrap

static string ToString(hipblasStatus_t status) {
  switch (status) {
    case HIPBLAS_STATUS_SUCCESS:
      return "HIPBLAS_STATUS_SUCCESS";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "HIPBLAS_STATUS_NOT_INITIALIZED";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "HIPBLAS_STATUS_ALLOC_FAILED";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "HIPBLAS_STATUS_INVALID_VALUE";
    //case HIPBLAS_STATUS_ARCH_MISMATCH:
    //  return "HIPBLAS_STATUS_ARCH_MISMATCH";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "HIPBLAS_STATUS_MAPPING_ERROR";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "HIPBLAS_STATUS_EXECUTION_FAILED";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "HIPBLAS_STATUS_INTERNAL_ERROR";
#if ROCM_VERSION >= 8000
    //case HIPBLAS_STATUS_NOT_SUPPORTED:
    //  return "HIPBLAS_STATUS_NOT_SUPPORTED";
    //case HIPBLAS_STATUS_LICENSE_ERROR:
    //  return "HIPBLAS_STATUS_LICENSE_ERROR";
#endif
    default:
      return port::StrCat("<invalid hipblas status: ", status, ">");
  }
}

// HIPBLAS has interfaces that permit pointers to be passed from either the host
// memory space or the device memory space; however, you must instruct it as to
// which address space those pointers are in with hipblasSetPointerMode.
//
// This helper sets the HIPBLAS pointer mode to a desired value for a HIPBLAS call
// you are about to perform in a given scope.
//
// The prior HIPBLAS pointer mode is retained and restored when this object goes
// out of scope.
/*class ScopedHipblasPointerMode {
 public:
  // Note that, because the setting of the hipblas pointer mode is fallible,
  // construction of this scoped datatype must be paired with a call to
  // Init().
  //
  // Parameters:
  //  handle: The hipblas library handle to act upon in setting the pointer mode.
  explicit ScopedHipblasPointerMode(ROCMExecutor *parent, hipblasHandle_t handle)
      : parent_(parent), handle_(handle), ok_(false) {}

  // Attempts the switch to the requested scoped pointer mode, new_mode.
  //
  // Note that when false is returned, an appropriate error has already been
  // logged.
  bool Init(hipblasPointerMode_t new_mode) {
    hipblasStatus_t ret =
        wrap::hipblasGetPointerMode(parent_, handle_, &old_mode_);
    if (ret != HIPBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to get old hipblas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    ret = wrap::hipblasSetPointerMode(parent_, handle_, new_mode);
    if (ret != HIPBLAS_STATUS_SUCCESS) {
      LOG(ERROR) << "failed to set new hipblas pointer mode: " << ToString(ret);
      return ok_ = false;
    }

    return ok_ = true;
  }

  // Switches back to the prior pointer mode, if the switch operation was
  // successful in the first place.
  ~ScopedHipblasPointerMode() {
    if (ok_) {
      hipblasStatus_t ret =
          wrap::hipblasSetPointerMode(parent_, handle_, old_mode_);
      if (ret != HIPBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "failed to set former hipblas pointer mode: "
                   << ToString(ret);
      }
    }
  }

 private:
  ROCMExecutor *parent_;   // Executor establishing this pointer mode for.
  hipblasHandle_t handle_;  // Handle to the HIPBLAS instance of interest.
  hipblasPointerMode_t old_mode_;  // Prior HIPBLAS pointer mode, to be restored.
  bool ok_;                       // Whether the change was successful.
};*/

bool ROCMBlas::Init() {
  hipblasStatus_t ret = wrap::hipblasCreate(parent_, &blas_);
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to create hipblas handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(rocm::ROCMExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    wrap::hipblasDestroy(parent_, blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsROCMStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  hipblasStatus_t ret =
      wrap::hipblasSetStream(parent_, blas_, AsROCMStreamValue(stream));
  if (ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to set stream for HIPBLAS calls: " << ToString(ret);
    return false;
  }

  return true;
}

namespace {

// Helper functions transforming blas arguments into HIPBLAS arguments.

hipblasOperation_t ROCMBlasTranspose(blas::Transpose trans) {
  switch (trans) {
    case blas::Transpose::kNoTranspose:
      return HIPBLAS_OP_N;
    case blas::Transpose::kTranspose:
      return HIPBLAS_OP_T;
    case blas::Transpose::kConjugateTranspose:
      return HIPBLAS_OP_C;
    default:
      LOG(FATAL) << "Invalid value of blas::Transpose.";
  }
}

/*hipblasFillMode_t ROCMBlasUpperLower(blas::UpperLower uplo) {
  switch (uplo) {
    case blas::UpperLower::kUpper:
      return HIPBLAS_FILL_MODE_UPPER;
    case blas::UpperLower::kLower:
      return HIPBLAS_FILL_MODE_LOWER;
    default:
      LOG(FATAL) << "Invalid value of blas::UpperLower.";
  }
}*/

/*hipblasDiagType_t ROCMBlasDiagonal(blas::Diagonal diag) {
  switch (diag) {
    case blas::Diagonal::kUnit:
      return HIPBLAS_DIAG_UNIT;
    case blas::Diagonal::kNonUnit:
      return HIPBLAS_DIAG_NON_UNIT;
    default:
      LOG(FATAL) << "Invalid value of blas::Diagonal.";
  }
}*/

/*hipblasSideMode_t ROCMBlasSide(blas::Side side) {
  switch (side) {
    case blas::Side::kLeft:
      return HIPBLAS_SIDE_LEFT;
    case blas::Side::kRight:
      return HIPBLAS_SIDE_RIGHT;
    default:
      LOG(FATAL) << "Invalid value of blas::Side.";
  }
}*/

/*
// ROCMDataType<T>::type translates from a C++ type (e.g. float) to a
// rocmDataType_t (e.g. ROCM_R_32F).  ROCMComputationType(ty) translates from a
// blas::ComputationType to a rocmDataType_t.
//
// These are used to build the argument type and computation type args to
// hipblasGemmEx.  hipblasGemmEx and rocmDataType_t are available only on
// ROCM >= 8.0.
#if ROCM_VERSION >= 8000
template <typename T>
struct ROCMDataType;

template <>
struct ROCMDataType<Eigen::half> {
  static constexpr rocmDataType_t type = SE_ROCM_DATA_HALF;
};

template <>
struct ROCMDataType<std::complex<Eigen::half>> {
  static constexpr rocmDataType_t type = ROCM_C_16F;
};

template <>
struct ROCMDataType<float> {
  static constexpr rocmDataType_t type = ROCM_R_32F;
};

template <>
struct ROCMDataType<std::complex<float>> {
  static constexpr rocmDataType_t type = ROCM_C_32F;
};

template <>
struct ROCMDataType<double> {
  static constexpr rocmDataType_t type = ROCM_R_64F;
};

template <>
struct ROCMDataType<std::complex<double>> {
  static constexpr rocmDataType_t type = ROCM_C_64F;
};

template <>
struct ROCMDataType<int> {
  static constexpr rocmDataType_t type = ROCM_R_32I;
};

template <>
struct ROCMDataType<int8> {
  static constexpr rocmDataType_t type = ROCM_R_8I;
};

template <>
struct ROCMDataType<std::complex<int8>> {
  static constexpr rocmDataType_t type = ROCM_C_8I;
};

template <>
struct ROCMDataType<uint8> {
  static constexpr rocmDataType_t type = ROCM_R_8U;
};

template <>
struct ROCMDataType<std::complex<uint8>> {
  static constexpr rocmDataType_t type = ROCM_C_8U;
};

rocmDataType_t ROCMComputationType(blas::ComputationType ty) {
  switch (ty) {
    case blas::ComputationType::kF16:
      return ROCM_R_16F;
    case blas::ComputationType::kF32:
      return ROCM_R_32F;
    case blas::ComputationType::kF64:
      return ROCM_R_64F;
    case blas::ComputationType::kI32:
      return ROCM_R_32I;
    case blas::ComputationType::kComplexF32:
      return ROCM_C_32F;
    case blas::ComputationType::kComplexF64:
      return ROCM_C_64F;
  }
}
#endif
*/

}  // namespace

template <typename FuncT, typename... Args>
bool ROCMBlas::DoBlasInternalImpl(FuncT hipblas_func, Stream *stream,
                                  bool pointer_mode_host, bool err_on_failure,
                                  Args... args) {
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  /*ScopedHipblasPointerMode pointer_mode{parent_, blas_};
  if (!pointer_mode.Init(pointer_mode_host ? HIPBLAS_POINTER_MODE_HOST
                                           : HIPBLAS_POINTER_MODE_DEVICE)) {
    return false;
  }*/

  hipblasStatus_t ret = hipblas_func(parent_, blas_, args...);
  if (err_on_failure && ret != HIPBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "failed to run HIPBLAS routine " << hipblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == HIPBLAS_STATUS_SUCCESS;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::hipblasSasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::hipblasDasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasScasum, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDzasum, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::hipblasSaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::hipblasDaxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasCaxpy, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasZaxpy, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::hipblasScopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::hipblasDcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasCcopy, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemory(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasZcopy, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemory(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::hipblasSdot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::hipblasDdot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCdotc, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZdotc, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCdotu, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZdotu, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(result)));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  //return DoBlasInternal(wrap::hipblasSnrm2, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  //return DoBlasInternal(wrap::hipblasDnrm2, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasScnrm2, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDznrm2, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSrot, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDrot, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  return false;
  //return DoBlasInternal(wrap::hipblasCsrot, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy, &c, &s);
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  return false;
  //return DoBlasInternal(wrap::hipblasZdrot, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy, &c, &s);
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  return false;
  //return DoBlasInternal(wrap::hipblasSrotg, stream,
  //                      false /* = pointer_mode_host */, ROCMMemoryMutable(a),
  //                      ROCMMemoryMutable(b), ROCMMemoryMutable(c),
  //                      ROCMMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  return false;
  //return DoBlasInternal(wrap::hipblasDrotg, stream,
  //                      false /* = pointer_mode_host */,
  //                      ROCMComplex(ROCMMemoryMutable(a)), ROCMMemoryMutable(b),
  //                      ROCMMemoryMutable(c), ROCMMemoryMutable(s));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCrotg, stream, false /* = pointer_mode_host */,
  //    ROCMComplex(ROCMMemoryMutable(a)), ROCMComplex(ROCMMemoryMutable(b)),
  //    ROCMComplex(ROCMMemoryMutable(c)), ROCMComplex(ROCMMemoryMutable(s)));
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZrotg, stream, false /* = pointer_mode_host */,
  //    ROCMComplex(ROCMMemoryMutable(a)), ROCMComplex(ROCMMemoryMutable(b)),
  //    ROCMComplex(ROCMMemoryMutable(c)), ROCMComplex(ROCMMemoryMutable(s)));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  return false;
  //return DoBlasInternal(wrap::hipblasSrotm, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy,
  //                      ROCMMemory(param));
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  return false;
  //return DoBlasInternal(wrap::hipblasDrotm, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy,
  //                      ROCMMemory(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  return false;
  //return DoBlasInternal(wrap::hipblasSrotmg, stream,
  //                      false /* = pointer_mode_host */, ROCMMemoryMutable(d1),
  //                      ROCMMemoryMutable(d2), ROCMMemoryMutable(x1),
  //                      ROCMMemory(y1), ROCMMemoryMutable(param));
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  return false;
  //return DoBlasInternal(wrap::hipblasDrotmg, stream,
  //                      false /* = pointer_mode_host */, ROCMMemoryMutable(d1),
  //                      ROCMMemoryMutable(d2), ROCMMemoryMutable(x1),
  //                      ROCMMemory(y1), ROCMMemoryMutable(param));
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::hipblasSscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::hipblasDscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCsscal, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZdscal, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCscal, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZscal, stream, true /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasSswap, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasDswap, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasCswap, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasZswap, stream,
  //                      true /* = pointer_mode_host */, elem_count,
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx,
  //                      ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(wrap::hipblasIsamax, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(wrap::hipblasIdamax, stream,
  //                      false /* = pointer_mode_host */, elem_count,
  //                      ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIcamax, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIzamax, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIsamin, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIdamin, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIcamin, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasIzamin, stream, false /* = pointer_mode_host */, elem_count,
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSgbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, ROCMMemory(a), lda,
  //    ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDgbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, kl, ku, &alpha, ROCMMemory(a), lda,
  //    ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCgbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, kl, ku, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZgbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, kl, ku, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::hipblasSgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::hipblasDgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCgemv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZgemv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(trans), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      wrap::hipblasSger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDger, stream, true /* = pointer_mode_host */, m, n, &alpha,
  //    ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCgerc, stream, true /* = pointer_mode_host */, m, n,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZgerc, stream, true /* = pointer_mode_host */, m, n,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCgeru, stream, true /* = pointer_mode_host */, m, n,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZgeru, stream, true /* = pointer_mode_host */, m, n,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemory(y)), incy, ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, k, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, k, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChemv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhemv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCher, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, &alpha, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZher, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, &alpha, ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCher2, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZher2, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(a)), lda);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChpmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(ap)), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhpmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(ap)), ROCMComplex(ROCMMemory(x)), incx,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(y)), incy);
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChpr, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhpr, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChpr2, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhpr2, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(x)), incx, ROCMComplex(ROCMMemory(y)), incy,
  //    ROCMComplex(ROCMMemoryMutable(ap)));
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSsbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, k, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
  //    incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDsbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), n, k, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
  //    incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasSspmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(ap),
  //                      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasDspmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(ap),
  //                      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  return false;
  //return DoBlasInternal(wrap::hipblasSspr, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  return false;
  //return DoBlasInternal(wrap::hipblasDspr, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  return false;
  //return DoBlasInternal(wrap::hipblasSspr2, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemory(y), incy, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  return false;
  //return DoBlasInternal(wrap::hipblasDspr2, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemory(y), incy, ROCMMemoryMutable(ap));
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasSsymv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(a), lda,
  //                      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  return false;
  //return DoBlasInternal(wrap::hipblasDsymv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(a), lda,
  //                      ROCMMemory(x), incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return false;
  //return DoBlasInternal(wrap::hipblasSsyr, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return false;
  //return DoBlasInternal(wrap::hipblasDsyr, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  return false;
  //return DoBlasInternal(wrap::hipblasSsyr2, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  return false;
  //return DoBlasInternal(wrap::hipblasDsyr2, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
  //                      incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasStbmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasDtbmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCtbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZtbmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasStbsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasDtbsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, k, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCtbsv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZtbsv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, k, ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasStpmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDtpmv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasCtpmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasZtpmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasStpsv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDtpsv, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //    ROCMBlasDiagonal(diag), n, ROCMMemory(ap), ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasCtpsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasZtpsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(ap)),
  //                      ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasStrmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasDtrmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasCtrmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
  //                      lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasZtrmv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
  //                      lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasStrsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasDtrsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMMemory(a), lda,
  //                      ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasCtrsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
  //                      lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  return false;
  //return DoBlasInternal(wrap::hipblasZtrsv, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans),
  //                      ROCMBlasDiagonal(diag), n, ROCMComplex(ROCMMemory(a)),
  //                      lda, ROCMComplex(ROCMMemoryMutable(x)), incx);
}

bool ROCMBlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
#if ROCM_VERSION >= 7050
  VLOG(1) << port::Printf(
      "doing HIPBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
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
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSgemmEx, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
  //    ROCMMemory(a), SE_ROCM_DATA_HALF, lda, ROCMMemory(b), SE_ROCM_DATA_HALF,
  //    ldb, &beta, ROCMMemoryMutable(c), SE_ROCM_DATA_HALF, ldc);
#else
  LOG(ERROR) << "fp16 sgemm is not implemented in this HIPBLAS version "
             << "(need at least ROCM 7.5)";
  return false;
#endif
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  VLOG(1) << port::Printf(
      "doing HIPBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
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
      wrap::hipblasSgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::hipblasDgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCgemm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
  //    ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZgemm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //    ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
  //    ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

template <typename InT, typename OutT, typename CompT>
bool ROCMBlas::DoBlasGemmWithAlgorithmImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const CompT &alpha, const DeviceMemory<InT> &a, int lda,
    const DeviceMemory<InT> &b, int ldb, const CompT &beta,
    DeviceMemory<OutT> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
#if 0
// ROCM < version 8 and GPUs < sm_50 don't support hipblasGemmEx.
#if ROCM_VERSION < 8000
  return false;
#else
  int cc_major, cc_minor;
  if (stream->parent()->GetDeviceDescription().rocm_compute_capability(
          &cc_major, &cc_minor) &&
      cc_major < 5) {
    return false;
  }

  struct TimerDeleter {
    void operator()(ROCMTimer *t) {
      t->Destroy();
      delete t;
    }
  };
  std::unique_ptr<ROCMTimer, TimerDeleter> timer;
  if (output_profile_result != nullptr) {
    timer.reset(new ROCMTimer(parent_));
    if (!timer->Init() || !timer->Start(AsROCMStream(stream))) {
      return false;
    }
  }

  rocmDataType_t rocm_in_type = ROCMDataType<InT>::type;
  // Since we are converting 'algorithm' to hipblasGemmAlgo_t by static_cast,
  // we do the following compile-time check on the default value:
  static_assert(blas::kDefaultGemmAlgo == HIPBLAS_GEMM_DFALT, "");
  bool result = DoBlasInternalFailureOK(
      wrap::hipblasGemmEx, stream, /* pointer_mode_host = */ true,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), rocm_in_type, lda, ROCMMemory(b), rocm_in_type, ldb, &beta,
      ROCMMemoryMutable(c), ROCMDataType<OutT>::type, ldc,
      ROCMComputationType(computation_type),
      static_cast<hipblasGemmAlgo_t>(algorithm));

  if (timer != nullptr && result) {
    // ROCMTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(AsROCMStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm);
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
#endif
#endif
  return false;
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
// hipblasGemmAlgo_t (and the function that accepts this type, hipblasGemmEx)
// were first introduced in ROCM 8.
#if ROCM_VERSION >= 8000
  for (hipblasGemmAlgo_t algo :
       {HIPBLAS_GEMM_DFALT, HIPBLAS_GEMM_ALGO0, HIPBLAS_GEMM_ALGO1,
        HIPBLAS_GEMM_ALGO2, HIPBLAS_GEMM_ALGO3, HIPBLAS_GEMM_ALGO4,
        HIPBLAS_GEMM_ALGO5, HIPBLAS_GEMM_ALGO6, HIPBLAS_GEMM_ALGO7}) {
    out_algorithms->push_back(algo);
  }
#endif
  return true;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, int alpha, const DeviceMemory<int8> &a, int lda,
    const DeviceMemory<int8> &b, int ldb, int beta, DeviceMemory<int> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const Eigen::half &alpha,
    const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, const Eigen::half &beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  return false;
  //return DoBlasGemmWithAlgorithmImpl(
  //    stream, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
  //    computation_type, algorithm, output_profile_result);
}

template <typename T, typename FuncT>
port::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT hipblas_func, Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k, T alpha,
    const port::ArraySlice<DeviceMemory<T> *> &a_ptrs_to_wrappers, int lda,
    const port::ArraySlice<DeviceMemory<T> *> &b_ptrs_to_wrappers, int ldb,
    T beta, const port::ArraySlice<DeviceMemory<T> *> &c_ptrs_to_wrappers,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  // Alocate local vectors to hold device pointers to matrices
  std::vector<T *> a_raw_ptrs, b_raw_ptrs, c_raw_ptrs;
  for (int i = 0; i < batch_count; ++i) {
    a_raw_ptrs.push_back(static_cast<T *>(a_ptrs_to_wrappers[i]->opaque()));
    b_raw_ptrs.push_back(static_cast<T *>(b_ptrs_to_wrappers[i]->opaque()));
    c_raw_ptrs.push_back(static_cast<T *>(c_ptrs_to_wrappers[i]->opaque()));
  }

  //  batch_count <= 1 is base case, no definable matrix stride, set it same as ld*
  long long bsa = lda;
  long long bsb = ldb;
  long long bsc = ldc;
  bool bsa_is_constant = true;
  bool bsb_is_constant = true;
  bool bsc_is_constant = true;

  if( batch_count > 1 )
  {
    // Remember first stride; if any other stride is different that this one, KABLAM
    bsa = a_raw_ptrs[1] - a_raw_ptrs[0];
    bsb = b_raw_ptrs[1] - b_raw_ptrs[0];
    bsc = c_raw_ptrs[1] - c_raw_ptrs[0];

    //  Loop to verify that batched strides are constant
    //  All the test cases from batch_matmul_op_test.py seem to satisfy this requirement of a constant
    //  stride.  If this can be proven globally, then this loop check can be safely removed
    for( int i=1; i < batch_count-1; ++i )
    {
      long long iterative_bsa = a_raw_ptrs[i+1] - a_raw_ptrs[i];
      if( iterative_bsa != bsa)
      {
        bsa_is_constant = false;
        break;
      }

      long long iterative_bsb = b_raw_ptrs[i+1] - b_raw_ptrs[i];
      if( iterative_bsb != bsb)
      {
        bsb_is_constant = false;
        break;
      }

      long long iterative_bsc = c_raw_ptrs[i+1] - c_raw_ptrs[i];
      if( iterative_bsc != bsc)
      {
        bsc_is_constant = false;
        break;
      }
    }
  }

  assert(!(ldc < m || bsc < ldc * n));

  if (ROCMBlasTranspose(transa) == HIPBLAS_OP_N)
      assert(!(lda < m || bsa < lda * k));
  else
      assert(!(lda < k || bsa < lda * m));

  if (ROCMBlasTranspose(transb) == HIPBLAS_OP_N)
      assert(!(ldb < k || bsb < ldb * n));
  else
      assert(!(ldb < n || bsc < ldc * k));

  if(bsa_is_constant && bsb_is_constant && bsc_is_constant)
  {
    bool ok = DoBlasInternal(
            hipblas_func, stream, true /* = pointer_mode_host */,
            ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k,
            ROCMComplex(&alpha), a_raw_ptrs[ 0 ], lda, bsa,
            b_raw_ptrs[ 0 ], ldb, bsb, ROCMComplex(&beta),
            c_raw_ptrs[ 0 ], ldc, bsc, batch_count);

      if (ok) {
        return port::Status::OK();
      }
  }
 
  return port::Status(port::error::INTERNAL,
                      "failed BLAS call, see log for details");
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::hipblasSgemmStridedBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha,
    const port::ArraySlice<DeviceMemory<double> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<double> *> &b_array, int ldb,
    double beta, const port::ArraySlice<DeviceMemory<double> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::hipblasDgemmStridedBatched, stream, transa, transb, m, n, k, alpha,
      a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count,
      scratch_allocator);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<float> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &b_array,
    int ldb, std::complex<float> beta,
    const port::ArraySlice<DeviceMemory<std::complex<float>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
  //port::Status status = DoBlasGemmBatchedInternal(
  //    wrap::hipblasCgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
  //    lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  //if (!status.ok()) {
  //  LOG(ERROR) << status;
  //}
  //return status.ok();
}

bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, std::complex<double> alpha,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &a_array,
    int lda,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &b_array,
    int ldb, std::complex<double> beta,
    const port::ArraySlice<DeviceMemory<std::complex<double>> *> &c_array,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
  //port::Status status = DoBlasGemmBatchedInternal(
  //    wrap::hipblasZgemmBatched, stream, transa, transb, m, n, k, alpha, a_array,
  //    lda, b_array, ldb, beta, c_array, ldc, batch_count, scratch_allocator);
  //if (!status.ok()) {
  //  LOG(ERROR) << status;
  //}
  //return status.ok();
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasChemm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZhemm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasCherk, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      &beta, ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasZherk, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      &beta, ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasCher2k, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      ROCMComplex(ROCMMemory(b)), ldb, &beta,
  //                      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasZher2k, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      ROCMComplex(ROCMMemory(b)), ldb, &beta,
  //                      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSsymm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, ROCMMemory(a),
  //    lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDsymm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, &alpha, ROCMMemory(a),
  //    lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCsymm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZsymm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemory(b)), ldb,
  //    ROCMComplex(&beta), ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSsyrk, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
  //    ROCMMemory(a), lda, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDsyrk, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
  //    ROCMMemory(a), lda, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCsyrk, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(&beta),
  //    ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZsyrk, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k,
  //    ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(&beta),
  //    ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasSsyr2k, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
  //    ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDsyr2k, stream, true /* = pointer_mode_host */,
  //    ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n, k, &alpha,
  //    ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasCsyr2k, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
  //                      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  return false;
  //return DoBlasInternal(wrap::hipblasZsyr2k, stream,
  //                      true /* = pointer_mode_host */,
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(trans), n,
  //                      k, ROCMComplex(&alpha), ROCMComplex(ROCMMemory(a)), lda,
  //                      ROCMComplex(ROCMMemory(b)), ldb, ROCMComplex(&beta),
  //                      ROCMComplex(ROCMMemoryMutable(c)), ldc);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasStrmm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a), lda,
  //    ROCMMemoryMutable(b), ldb, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasDtrmm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a), lda,
  //    ROCMMemoryMutable(b), ldb, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCtrmm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb,
  //    ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZtrmm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb,
  //    ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  return false;
  //return DoBlasInternal(wrap::hipblasStrsm, stream,
  //                      true /* = pointer_mode_host */, ROCMBlasSide(side),
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //                      ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a),
  //                      lda, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  return false;
  //return DoBlasInternal(wrap::hipblasDtrsm, stream,
  //                      true /* = pointer_mode_host */, ROCMBlasSide(side),
  //                      ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //                      ROCMBlasDiagonal(diag), m, n, &alpha, ROCMMemory(a),
  //                      lda, ROCMMemoryMutable(b), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasCtrsm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  return false;
  //return DoBlasInternal(
  //    wrap::hipblasZtrsm, stream, true /* = pointer_mode_host */,
  //    ROCMBlasSide(side), ROCMBlasUpperLower(uplo), ROCMBlasTranspose(transa),
  //    ROCMBlasDiagonal(diag), m, n, ROCMComplex(&alpha),
  //    ROCMComplex(ROCMMemory(a)), lda, ROCMComplex(ROCMMemoryMutable(b)), ldb);
}

}  // namespace rocm

namespace gpu = ::perftools::gputools;

void initialize_hipblas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::rocm::kROCmPlatformId, gpu::rocm::kHipBlasPlugin, "HIPBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::rocm::ROCMExecutor *rocm_executor =
                    dynamic_cast<gpu::rocm::ROCMExecutor *>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the HIPBLAS "
                      << "support library with a non-ROCM StreamExecutor";
                  return nullptr;
                }

                gpu::rocm::ROCMBlas *blas =
                    new gpu::rocm::ROCMBlas(rocm_executor);
                if (!blas->Init()) {
                  // Note: Init() will log a more specific error.
                  delete blas;
                  return nullptr;
                }
                return blas;
              });

  if (!status.ok()) {
    LOG(ERROR) << "Unable to register HIPBLAS factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::rocm::kROCmPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::rocm::kHipBlasPlugin);
}

}  // namespace gputools
}  // namespace perftools

REGISTER_MODULE_INITIALIZER(register_hipblas,
                            { perftools::gputools::initialize_hipblas(); });
