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

#include "rocm/include/rocblas.h"

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

namespace stream_executor {
namespace rocm {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kRocBlasPlugin);

namespace wrap {

#define PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(__name)                      \
  struct WrapperShim__##__name {                                    \
    static const char *kName;                                       \
    template <typename... Args>                                     \
    rocblas_status operator()(ROCMExecutor *parent, Args... args) { \
      rocm::ScopedActivateExecutorContext sac{parent};              \
      return ::__name(args...);                                     \
    }                                                               \
  } __name;                                                         \
  const char *WrapperShim__##__name::kName = #__name;

#define PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(__name) \
  PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(__name)

#define HIPBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(rocblas_snrm2)                    \
  __macro(rocblas_dnrm2)                    \
/*  __macro(rocblas_scnrm2)                   \
  __macro(rocblas_dznrm2)                   */ \
  __macro(rocblas_sdot)                     \
  __macro(rocblas_ddot)                     \
/*  __macro(rocblas_cdotu)                    \
  __macro(rocblas_cdotc)                    \
  __macro(rocblas_zdotu)                    \
  __macro(rocblas_zdotc)                    */ \
  __macro(rocblas_sscal)                    \
  __macro(rocblas_dscal)                    \
/*  __macro(rocblas_cscal)                    \
  __macro(rocblas_csscal)                   \
  __macro(rocblas_zscal)                    \
  __macro(rocblas_zdscal)                   */ \
  __macro(rocblas_saxpy)                    \
  __macro(rocblas_daxpy)                    \
/*  __macro(rocblas_caxpy)                    \
  __macro(rocblas_zaxpy)                    */ \
  __macro(rocblas_scopy)                    \
  __macro(rocblas_dcopy)                    \
/*  __macro(rocblas_ccopy)                    \
  __macro(rocblas_zcopy)                    */ \
  __macro(rocblas_sswap)                    \
  __macro(rocblas_dswap)                    \
/*  __macro(rocblas_cswap)                    \
  __macro(rocblas_zswap)                    */ \
  __macro(rocblas_isamax)                   \
  __macro(rocblas_idamax)                   \
/*  __macro(rocblas_icamax)                   \
  __macro(rocblas_izamax)                   */ \
  __macro(rocblas_isamin)                   \
  __macro(rocblas_idamin)                   \
/*  __macro(rocblas_icamin)                   \
  __macro(rocblas_izamin)                   */ \
  __macro(rocblas_sasum)                    \
  __macro(rocblas_dasum)                    \
/*  __macro(rocblas_scasum)                   \
  __macro(rocblas_dzasum)                   \
  __macro(rocblas_srot)                     \
  __macro(rocblas_drot)                     \
  __macro(rocblas_crot)                     \
  __macro(rocblas_csrot)                    \
  __macro(rocblas_zrot)                     \
  __macro(rocblas_zdrot)                    \
  __macro(rocblas_srotg)                    \
  __macro(rocblas_drotg)                    \
  __macro(rocblas_Crotg)                    \
  __macro(rocblas_crotg)                    \
  __macro(rocblas_zrotm)                    \
  __macro(rocblas_drotm)                    \
  __macro(rocblas_srotmg)                   \
  __macro(rocblas_drotmg)                   */ \
  __macro(rocblas_sgemv)                    \
  __macro(rocblas_dgemv)                    \
/*  __macro(rocblas_cgemv)                    \
  __macro(rocblas_zgemv)                    \
  __macro(rocblas_sgbmv)                    \
  __macro(rocblas_dgbmv)                    \
  __macro(rocblas_cgbmv)                    \
  __macro(rocblas_zgbmv)                    \
  __macro(rocblas_strmv)                    \
  __macro(rocblas_dtrmv)                    \
  __macro(rocblas_ctrmv)                    \
  __macro(rocblas_ztrmv)                    \
  __macro(rocblas_stbmv)                    \
  __macro(rocblas_dtbmv)                    \
  __macro(rocblas_ctbmv)                    \
  __macro(rocblas_ztbmv)                    \
  __macro(rocblas_stpmv)                    \
  __macro(rocblas_dtpmv)                    \
  __macro(rocblas_ctpmv)                    \
  __macro(rocblas_ztpmv)                    \
  __macro(rocblas_strsv)                    \
  __macro(rocblas_dtrsv)                    \
  __macro(rocblas_ctrsv)                    \
  __macro(rocblas_ztrsv)                    \
  __macro(rocblas_stpsv)                    \
  __macro(rocblas_dtpsv)                    \
  __macro(rocblas_ctpsv)                    \
  __macro(rocblas_ztpsv)                    \
  __macro(rocblas_stbsv)                    \
  __macro(rocblas_dtbsv)                    \
  __macro(rocblas_ctbsv)                    \
  __macro(rocblas_ztbsv)                    \
  __macro(rocblas_ssymv)                    \
  __macro(rocblas_dsymv)                    \
  __macro(rocblas_csymv)                    \
  __macro(rocblas_zsymv)                    \
  __macro(rocblas_chemv)                    \
  __macro(rocblas_zhemv)                    \
  __macro(rocblas_ssbmv)                    \
  __macro(rocblas_dsbmv)                    \
  __macro(rocblas_chbmv)                    \
  __macro(rocblas_zhbmv)                    \
  __macro(rocblas_sspmv)                    \
  __macro(rocblas_dspmv)                    \
  __macro(rocblas_chpmv)                    \
  __macro(rocblas_zhpmv)                    */ \
  __macro(rocblas_sger)                     \
  __macro(rocblas_dger)                     \
/*  __macro(rocblas_cgeru)                    \
  __macro(rocblas_cgerc)                    \
  __macro(rocblas_zgeru)                    \
  __macro(rocblas_zgerc)                    */ \
  __macro(rocblas_ssyr)                     \
  __macro(rocblas_dsyr)                     \
/*  __macro(rocblas_csyr)                     \
  __macro(rocblas_zsyr)                     \
  __macro(rocblas_cher)                     \
  __macro(rocblas_zher)                     \
  __macro(rocblas_sspr)                     \
  __macro(rocblas_dspr)                     \
  __macro(rocblas_chpr)                     \
  __macro(rocblas_zhpr)                     \
  __macro(rocblas_ssyr2)                    \
  __macro(rocblas_dsyr2)                    \
  __macro(rocblas_csyr2)                    \
  __macro(rocblas_zsyr2)                    \
  __macro(rocblas_cher2)                    \
  __macro(rocblas_zher2)                    \
  __macro(rocblas_sspr2)                    \
  __macro(rocblas_dspr2)                    \
  __macro(rocblas_chpr2)                    \
  __macro(rocblas_zhpr2)                    */ \
  __macro(rocblas_sgemm)                    \
  __macro(rocblas_dgemm)                    \
/*  __macro(rocblas_cgemm)                    \
  __macro(rocblas_zgemm)                    \
  __macro(rocblas_ssyrk)                    \
  __macro(rocblas_dsyrk)                    \
  __macro(rocblas_csyrk)                    \
  __macro(rocblas_zsyrk)                    \
  __macro(rocblas_cherk)                    \
  __macro(rocblas_zherk)                    \
  __macro(rocblas_ssyr2k)                   \
  __macro(rocblas_dsyr2k)                   \
  __macro(rocblas_csyr2k)                   \
  __macro(rocblas_zsyr2k)                   \
  __macro(rocblas_cher2k)                   \
  __macro(rocblas_zher2k)                   \
  __macro(rocblas_ssyrkx)                   \
  __macro(rocblas_dsyrkx)                   \
  __macro(rocblas_csyrkx)                   \
  __macro(rocblas_zsyrkx)                   \
  __macro(rocblas_cherkx)                   \
  __macro(rocblas_zherkx)                   \
  __macro(rocblas_ssymm)                    \
  __macro(rocblas_dsymm)                    \
  __macro(rocblas_csymm)                    \
  __macro(rocblas_zsymm)                    \
  __macro(rocblas_chemm)                    \
  __macro(rocblas_zhemm)                    */ \
  __macro(rocblas_strsm)                    \
  __macro(rocblas_dtrsm)                    \
/*  __macro(rocblas_ctrsm)                    \
  __macro(rocblas_ztrsm)                    \
  __macro(rocblas_strmm)                    \
  __macro(rocblas_dtrmm)                    \
  __macro(rocblas_ctrmm)                    \
  __macro(rocblas_ztrmm)                    */ \
  __macro(rocblas_sgeam)                    \
  __macro(rocblas_dgeam)                    \
/*  __macro(rocblas_cgeam)                    \
  __macro(rocblas_zgeam)                    \
  __macro(rocblas_sdgmm)                    \
  __macro(rocblas_ddgmm)                    \
  __macro(rocblas_cdgmm)                    \
  __macro(rocblas_zdgmm) */

PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(rocblas_create_handle)
PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(rocblas_destroy_handle)
PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(rocblas_set_stream)
//PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(rocblas_set_pointer_mode)
//PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP(rocblas_get_pointer_mode)
//PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_sgemm_batched)
PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_sgemm_strided_batched)
//PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_dgemm_batched)
PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_dgemm_strided_batched)
//PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_cgemm_batched)
//PERFTOOLS_GPUTOOLS_ROCBLAS_WRAP(rocblas_zgemm_batched)
HIPBLAS_BLAS_ROUTINE_EACH(PERFTOOLS_GPUTOOLS_ROCBLAS_V2_WRAP)

}  // namespace wrap

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
      return port::StrCat("<invalid rocBLAS status: ", status, ">");
  }
}

bool ROCMBlas::Init() {
  rocblas_status ret = wrap::rocblas_create_handle(parent_, &blas_);
  if (ret != rocblas_status_success) {
    LOG(ERROR) << "failed to create rocBLAS handle: " << ToString(ret);
    return false;
  }

  return true;
}

ROCMBlas::ROCMBlas(rocm::ROCMExecutor *parent)
    : parent_(CHECK_NOTNULL(parent)), blas_(nullptr) {}

ROCMBlas::~ROCMBlas() {
  if (blas_ != nullptr) {
    wrap::rocblas_destroy_handle(parent_, blas_);
  }
}

bool ROCMBlas::SetStream(Stream *stream) {
  CHECK(stream != nullptr);
  CHECK(AsROCMStreamValue(stream) != nullptr);
  CHECK(blas_ != nullptr);
  rocblas_status ret =
      wrap::rocblas_set_stream(parent_, blas_, AsROCMStreamValue(stream));
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
  mutex_lock lock{mu_};

  CHECK(blas_ != nullptr);
  if (!SetStream(stream)) {
    return false;
  }

  rocblas_status ret = rocblas_func(parent_, blas_, args...);
  if (err_on_failure && ret != rocblas_status_success) {
    LOG(ERROR) << "failed to run ROCBLAS routine " << rocblas_func.kName << ": "
               << ToString(ret);
  }
  return ret == rocblas_status_success;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::rocblas_sasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::rocblas_dasum, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the ASUM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasAsum(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the ASUM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, float alpha,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_saxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count, double alpha,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_daxpy, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the AXPY operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasAxpy(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the AXPY operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_scopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dcopy, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the COPY operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasCopy(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the COPY operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *result) {
  return DoBlasInternal(
      wrap::rocblas_sdot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDot(Stream *stream, uint64 elem_count,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *result) {
  return DoBlasInternal(
      wrap::rocblas_ddot, stream, false /* = pointer_mode_host */, elem_count,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasDotc(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasDotu(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the DOT operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<float> &x, int incx,
                          DeviceMemory<float> *result) {
  return DoBlasInternal(wrap::rocblas_snrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<double> &x, int incx,
                          DeviceMemory<double> *result) {
  return DoBlasInternal(wrap::rocblas_dnrm2, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          DeviceMemory<float> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the NRM2 operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasNrm2(Stream *stream, uint64 elem_count,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          DeviceMemory<double> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the NRM2 operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<float> *x, int incx,
                         DeviceMemory<float> *y, int incy, float c, float s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<double> *x, int incx,
                         DeviceMemory<double> *y, int incy, double c,
                         double s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<float>> *x, int incx,
                         DeviceMemory<std::complex<float>> *y, int incy,
                         float c, float s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRot(Stream *stream, uint64 elem_count,
                         DeviceMemory<std::complex<double>> *x, int incx,
                         DeviceMemory<std::complex<double>> *y, int incy,
                         double c, double s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROT operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<float> *a,
                          DeviceMemory<float> *b, DeviceMemory<float> *c,
                          DeviceMemory<float> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<double> *a,
                          DeviceMemory<double> *b, DeviceMemory<double> *c,
                          DeviceMemory<double> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<float>> *a,
                          DeviceMemory<std::complex<float>> *b,
                          DeviceMemory<float> *c,
                          DeviceMemory<std::complex<float>> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotg(Stream *stream, DeviceMemory<std::complex<double>> *a,
                          DeviceMemory<std::complex<double>> *b,
                          DeviceMemory<double> *c,
                          DeviceMemory<std::complex<double>> *s) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTG operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy,
                          const DeviceMemory<float> &param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTM operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotm(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy,
                          const DeviceMemory<double> &param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTM operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<float> *d1,
                           DeviceMemory<float> *d2, DeviceMemory<float> *x1,
                           const DeviceMemory<float> &y1,
                           DeviceMemory<float> *param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTMG operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasRotmg(Stream *stream, DeviceMemory<double> *d1,
                           DeviceMemory<double> *d2, DeviceMemory<double> *x1,
                           const DeviceMemory<double> &y1,
                           DeviceMemory<double> *param) {
  LOG(ERROR) << "rocBLAS does not currently support the ROTMG operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<float> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_sscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<double> *x, int incx) {
  return DoBlasInternal(wrap::rocblas_dscal, stream,
                        true /* = pointer_mode_host */, elem_count, &alpha,
                        ROCMMemoryMutable(x), incx);
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, float alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the SCAL operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count, double alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the SCAL operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<float> alpha,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the SCAL operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasScal(Stream *stream, uint64 elem_count,
                          std::complex<double> alpha,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the SCAL operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<float> *x, int incx,
                          DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_sswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<double> *x, int incx,
                          DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(wrap::rocblas_dswap, stream,
                        true /* = pointer_mode_host */, elem_count,
                        ROCMMemoryMutable(x), incx, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<float>> *x, int incx,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SWAP operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSwap(Stream *stream, uint64 elem_count,
                          DeviceMemory<std::complex<double>> *x, int incx,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SWAP operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::rocblas_isamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(wrap::rocblas_idamax, stream,
                        false /* = pointer_mode_host */, elem_count,
                        ROCMMemory(x), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMAX operation "
            << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasIamax(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMAX operation "
            << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<float> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::rocblas_isamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<double> &x, int incx,
                           DeviceMemory<int> *result) {
  return DoBlasInternal(
      wrap::rocblas_idamin, stream, false /* = pointer_mode_host */, elem_count,
      ROCMComplex(ROCMMemory(x)), incx, ROCMMemoryMutable(result));
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<float>> &x, int incx,
                           DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMIN operation "
            << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasIamin(Stream *stream, uint64 elem_count,
                           const DeviceMemory<std::complex<double>> &x,
                           int incx, DeviceMemory<int> *result) {
  LOG(ERROR) << "rocBLAS does not currently support the AMIN operation "
            << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGbmv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, uint64 kl, uint64 ku,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GBMV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_sgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  return DoBlasInternal(
      wrap::rocblas_dgemv, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(trans), m, n, &alpha, ROCMMemory(a), lda, ROCMMemory(x),
      incx, &beta, ROCMMemoryMutable(y), incy);
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMV operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemv(Stream *stream, blas::Transpose trans, uint64 m,
                          uint64 n, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMV operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, float alpha,
                         const DeviceMemory<float> &x, int incx,
                         const DeviceMemory<float> &y, int incy,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(
      wrap::rocblas_sger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGer(Stream *stream, uint64 m, uint64 n, double alpha,
                         const DeviceMemory<double> &x, int incx,
                         const DeviceMemory<double> &y, int incy,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(
      wrap::rocblas_dger, stream, true /* = pointer_mode_host */, m, n, &alpha,
      ROCMMemory(x), incx, ROCMMemory(y), incy, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GER operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGerc(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GER operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GERU operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGeru(Stream *stream, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the GERU operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HBMV operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HBMV operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMV operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHemv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMV operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2 operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2 operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &ap,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HPMV operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &ap,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the HPMV operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha,
                         const DeviceMemory<std::complex<float>> &x, int incx,
                         DeviceMemory<std::complex<float>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha,
                         const DeviceMemory<std::complex<double>> &x, int incx,
                         DeviceMemory<std::complex<double>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &x, int incx,
                          const DeviceMemory<std::complex<float>> &y, int incy,
                          DeviceMemory<std::complex<float>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR2 operation "
             << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &x, int incx,
                          const DeviceMemory<std::complex<double>> &y, int incy,
                          DeviceMemory<std::complex<double>> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the HPR2 operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, float alpha, const DeviceMemory<float> &a,
                          int lda, const DeviceMemory<float> &x, int incx,
                          float beta, DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SBMV operation "
             << "for the \"complex<float>\" dataype" ;

  return false;
}

bool ROCMBlas::DoBlasSbmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          uint64 k, double alpha, const DeviceMemory<double> &a,
                          int lda, const DeviceMemory<double> &x, int incx,
                          double beta, DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SBMV operation "
             << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &ap,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SPMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpmv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &ap,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SPMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR operation "
             << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR operation "
             << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR2 operation "
             << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSpr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *ap) {
  LOG(ERROR) << "rocBLAS does not currently support the SPR2 operation "
             << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &x, int incx, float beta,
                          DeviceMemory<float> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymv(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &x, int incx, double beta,
                          DeviceMemory<double> *y, int incy) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         float alpha, const DeviceMemory<float> &x, int incx,
                         DeviceMemory<float> *a, int lda) {
  return DoBlasInternal(wrap::rocblas_ssyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr(Stream *stream, blas::UpperLower uplo, uint64 n,
                         double alpha, const DeviceMemory<double> &x, int incx,
                         DeviceMemory<double> *a, int lda) {
  return DoBlasInternal(wrap::rocblas_dsyr, stream,
                        true /* = pointer_mode_host */,
                        ROCMBlasUpperLower(uplo), n, &alpha, ROCMMemory(x),
                        incx, ROCMMemoryMutable(a), lda);
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          float alpha, const DeviceMemory<float> &x, int incx,
                          const DeviceMemory<float> &y, int incy,
                          DeviceMemory<float> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2 operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr2(Stream *stream, blas::UpperLower uplo, uint64 n,
                          double alpha, const DeviceMemory<double> &x, int incx,
                          const DeviceMemory<double> &y, int incy,
                          DeviceMemory<double> *a, int lda) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2 operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBMV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<float>> &a,
                          int lda, DeviceMemory<std::complex<float>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTbsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          uint64 k, const DeviceMemory<std::complex<double>> &a,
                          int lda, DeviceMemory<std::complex<double>> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TBSV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPMV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &ap, DeviceMemory<float> *x,
                          int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &ap,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &ap,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTpsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &ap,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TPSV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsv(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, blas::Diagonal diag, uint64 n,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *x, int incx) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSV operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemm(
    Stream *stream, blas::Transpose transa,
    blas::Transpose transb, uint64 m, uint64 n, uint64 k,
    float alpha, const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc) {
  LOG(ERROR) << "fp16 sgemm is not implemented in this rocBLAS version";
  return false;
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  VLOG(1) << port::Printf(
      "doing rocBLAS SGEMM: at=%d bt=%d m=%llu n=%llu "
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
      wrap::rocblas_sgemm, stream, true /* = pointer_mode_host */,
      ROCMBlasTranspose(transa), ROCMBlasTranspose(transb), m, n, k, &alpha,
      ROCMMemory(a), lda, ROCMMemory(b), ldb, &beta, ROCMMemoryMutable(c), ldc);
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  return DoBlasInternal(
      wrap::rocblas_dgemm, stream, true /* = pointer_mode_host */,
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
  LOG(ERROR) << "rocBLAS does not currently support the GEMM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemm(Stream *stream, blas::Transpose transa,
                          blas::Transpose transb, uint64 m, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, float alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &x,
    int incx, float beta, DeviceMemory<float> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, double alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &x,
    int incx, double beta, DeviceMemory<double> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<float> alpha, const DeviceMemory<std::complex<float>> &a,
    int lda, const DeviceMemory<std::complex<float>> &x, int incx,
    std::complex<float> beta, DeviceMemory<std::complex<float>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemvWithProfiling(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n,
    std::complex<double> alpha, const DeviceMemory<std::complex<double>> &a,
    int lda, const DeviceMemory<std::complex<double>> &x, int incx,
    std::complex<double> beta, DeviceMemory<std::complex<double>> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemvWithProfilingImpl(stream, trans, m, n, alpha, a, lda, x,
                                     incx, beta, y, incy,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<Eigen::half> &a,
    int lda, const DeviceMemory<Eigen::half> &b, int ldb, float beta,
    DeviceMemory<Eigen::half> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha, const DeviceMemory<float> &a, int lda,
    const DeviceMemory<float> &b, int ldb, float beta, DeviceMemory<float> *c,
    int ldc, blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, double alpha, const DeviceMemory<double> &a, int lda,
    const DeviceMemory<double> &b, int ldb, double beta,
    DeviceMemory<double> *c, int ldc,
    blas::ProfileResult *output_profile_result) {
  return DoBlasGemmWithProfilingImpl(stream, transa, transb, m, n, k, alpha, a,
                                     lda, b, ldb, beta, c, ldc,
                                     output_profile_result);
}

bool ROCMBlas::DoBlasGemmWithProfiling(
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

bool ROCMBlas::DoBlasGemmWithProfiling(
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
bool ROCMBlas::DoBlasGemvWithProfilingImpl(
    Stream *stream, blas::Transpose trans, uint64 m, uint64 n, const T &alpha,
    const DeviceMemory<T> &a, int lda, const DeviceMemory<T> &x, int incx,
    const T &beta, DeviceMemory<T> *y, int incy,
    blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return false;
}

template <typename T, typename ParamType>
bool ROCMBlas::DoBlasGemmWithProfilingImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const ParamType &alpha, const DeviceMemory<T> &a,
    int lda, const DeviceMemory<T> &b, int ldb, const ParamType &beta,
    DeviceMemory<T> *c, int ldc, blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return false;
}

template <typename InT, typename OutT, typename CompT>
bool ROCMBlas::DoBlasGemmWithAlgorithmImpl(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const CompT &alpha, const DeviceMemory<InT> &a, int lda,
    const DeviceMemory<InT> &b, int ldb, const CompT &beta,
    DeviceMemory<OutT> *c, int ldc, blas::ComputationType computation_type,
    blas::AlgorithmType algorithm, blas::ProfileResult *output_profile_result) {
  // ROCM TODO: properly implement the interface
  return false;
}

bool ROCMBlas::GetBlasGemmAlgorithms(
    std::vector<blas::AlgorithmType> *out_algorithms) {
  // ROCM TODO: properly implement the interface
  return true;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<int> &alpha,
    const DeviceMemory<int8> &a, int lda, const DeviceMemory<int8> &b,
    int ldb, const HostOrDeviceScalar<int> &beta, DeviceMemory<int32> *c,
    int ldc, blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
             << "for the \"int8\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<Eigen::half> &alpha,
    const DeviceMemory<Eigen::half> &a, int lda,
    const DeviceMemory<Eigen::half> &b, int ldb,
    const HostOrDeviceScalar<Eigen::half> &beta, DeviceMemory<Eigen::half> *c,
    int ldc, blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
             << "for the \"half\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<float> &alpha,
    const DeviceMemory<float> &a, int lda, const DeviceMemory<float> &b,
    int ldb, const HostOrDeviceScalar<float> &beta, DeviceMemory<float> *c,
    int ldc, blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
             << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<double> &alpha,
    const DeviceMemory<double> &a, int lda, const DeviceMemory<double> &b,
    int ldb, const HostOrDeviceScalar<double> &beta, DeviceMemory<double> *c,
    int ldc, blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
             << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<std::complex<float>> &alpha,
    const DeviceMemory<std::complex<float>> &a, int lda,
    const DeviceMemory<std::complex<float>> &b, int ldb,
    const HostOrDeviceScalar<std::complex<float>> &beta,
    DeviceMemory<std::complex<float>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasGemmWithAlgorithm(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, const HostOrDeviceScalar<std::complex<double>> &alpha,
    const DeviceMemory<std::complex<double>> &a, int lda,
    const DeviceMemory<std::complex<double>> &b, int ldb,
    const HostOrDeviceScalar<std::complex<double>> &beta,
    DeviceMemory<std::complex<double>> *c, int ldc,
    blas::ComputationType computation_type, blas::AlgorithmType algorithm,
    blas::ProfileResult *output_profile_result) {
  LOG(ERROR) << "rocBLAS does not currently support the GEMMwithAlgorithm operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

template <typename T, typename FuncT>
port::Status ROCMBlas::DoBlasGemmBatchedInternal(
    FuncT rocblas_func, Stream *stream, blas::Transpose transa,
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

  if (ROCMBlasTranspose(transa) == rocblas_operation_none)
      assert(!(lda < m || bsa < lda * k));
  else
      assert(!(lda < k || bsa < lda * m));

  if (ROCMBlasTranspose(transb) == rocblas_operation_none)
      assert(!(ldb < k || bsb < ldb * n));
  else
      assert(!(ldb < n || bsc < ldc * k));

  if(bsa_is_constant && bsb_is_constant && bsc_is_constant)
  {
    bool ok = DoBlasInternal(
            rocblas_func, stream, true /* = pointer_mode_host */,
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
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &a, int lda,
    const port::ArraySlice<DeviceMemory<Eigen::half> *> &b, int ldb,
    float beta, const port::ArraySlice<DeviceMemory<Eigen::half> *> &c,
    int ldc, int batch_count, ScratchAllocator *scratch_allocator) {
  return false;
}


bool ROCMBlas::DoBlasGemmBatched(
    Stream *stream, blas::Transpose transa, blas::Transpose transb, uint64 m,
    uint64 n, uint64 k, float alpha,
    const port::ArraySlice<DeviceMemory<float> *> &a_array, int lda,
    const port::ArraySlice<DeviceMemory<float> *> &b_array, int ldb, float beta,
    const port::ArraySlice<DeviceMemory<float> *> &c_array, int ldc,
    int batch_count, ScratchAllocator *scratch_allocator) {
  port::Status status = DoBlasGemmBatchedInternal(
      wrap::rocblas_sgemm_strided_batched, stream, transa, transb, m, n, k, alpha,
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
      wrap::rocblas_dgemm_strided_batched, stream, transa, transb, m, n, k, alpha,
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
  LOG(ERROR) << "rocBLAS does not currently support the GEMMBatched operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
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
  LOG(ERROR) << "rocBLAS does not currently support the GEMMBatched operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHemm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HEMM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          float beta, DeviceMemory<std::complex<float>> *c,
                          int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HERK operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHerk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          double beta, DeviceMemory<std::complex<double>> *c,
                          int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HERK operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           float beta, DeviceMemory<std::complex<float>> *c,
                           int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2K operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasHer2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           double beta, DeviceMemory<std::complex<double>> *c,
                           int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the HER2K operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          const DeviceMemory<float> &b, int ldb, float beta,
                          DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          const DeviceMemory<double> &b, int ldb, double beta,
                          DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          const DeviceMemory<std::complex<float>> &b, int ldb,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSymm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          const DeviceMemory<std::complex<double>> &b, int ldb,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYMM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          float alpha, const DeviceMemory<float> &a, int lda,
                          float beta, DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          double alpha, const DeviceMemory<double> &a, int lda,
                          double beta, DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          std::complex<float> beta,
                          DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyrk(Stream *stream, blas::UpperLower uplo,
                          blas::Transpose trans, uint64 n, uint64 k,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          std::complex<double> beta,
                          DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYRK operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           float alpha, const DeviceMemory<float> &a, int lda,
                           const DeviceMemory<float> &b, int ldb, float beta,
                           DeviceMemory<float> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           double alpha, const DeviceMemory<double> &a, int lda,
                           const DeviceMemory<double> &b, int ldb, double beta,
                           DeviceMemory<double> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<float> alpha,
                           const DeviceMemory<std::complex<float>> &a, int lda,
                           const DeviceMemory<std::complex<float>> &b, int ldb,
                           std::complex<float> beta,
                           DeviceMemory<std::complex<float>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasSyr2k(Stream *stream, blas::UpperLower uplo,
                           blas::Transpose trans, uint64 n, uint64 k,
                           std::complex<double> alpha,
                           const DeviceMemory<std::complex<double>> &a, int lda,
                           const DeviceMemory<std::complex<double>> &b, int ldb,
                           std::complex<double> beta,
                           DeviceMemory<std::complex<double>> *c, int ldc) {
  LOG(ERROR) << "rocBLAS does not currently support the SYR2K operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrmm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRMM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, float alpha,
                          const DeviceMemory<float> &a, int lda,
                          DeviceMemory<float> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
	     << "for the \"float\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n, double alpha,
                          const DeviceMemory<double> &a, int lda,
                          DeviceMemory<double> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
	     << "for the \"double\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<float> alpha,
                          const DeviceMemory<std::complex<float>> &a, int lda,
                          DeviceMemory<std::complex<float>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
	     << "for the \"complex<float>\" dataype" ;
  return false;
}

bool ROCMBlas::DoBlasTrsm(Stream *stream, blas::Side side,
                          blas::UpperLower uplo, blas::Transpose transa,
                          blas::Diagonal diag, uint64 m, uint64 n,
                          std::complex<double> alpha,
                          const DeviceMemory<std::complex<double>> &a, int lda,
                          DeviceMemory<std::complex<double>> *b, int ldb) {
  LOG(ERROR) << "rocBLAS does not currently support the TRSM operation "
	     << "for the \"complex<double>\" dataype" ;
  return false;
}

}  // namespace rocm

namespace gpu = ::stream_executor;

void initialize_rocblas() {
  gpu::port::Status status =
      gpu::PluginRegistry::Instance()
          ->RegisterFactory<gpu::PluginRegistry::BlasFactory>(
              gpu::rocm::kROCmPlatformId, gpu::rocm::kRocBlasPlugin, "rocBLAS",
              [](gpu::internal::StreamExecutorInterface
                     *parent) -> gpu::blas::BlasSupport * {
                gpu::rocm::ROCMExecutor *rocm_executor =
                    dynamic_cast<gpu::rocm::ROCMExecutor *>(parent);
                if (rocm_executor == nullptr) {
                  LOG(ERROR)
                      << "Attempting to initialize an instance of the rocBLAS "
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
    LOG(ERROR) << "Unable to register rocBLAS factory: "
               << status.error_message();
  }

  gpu::PluginRegistry::Instance()->SetDefaultFactory(gpu::rocm::kROCmPlatformId,
                                                     gpu::PluginKind::kBlas,
                                                     gpu::rocm::kRocBlasPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_rocblas,
                            { stream_executor::initialize_rocblas(); });
