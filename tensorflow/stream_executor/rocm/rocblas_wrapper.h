/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file wraps rocblas API calls with dso loader so that we don't need to
// have explicit linking to librocblas. All TF hipsarse API usage should route
// through this wrapper.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCBLAS_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCBLAS_WRAPPER_H_

#include "rocm/include/rocblas.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {
namespace wrap {

using stream_executor::internal::CachedDsoLoader::GetRocblasDsoHandle;

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
#define FOREACH_ROCBLAS_API(__macro)            \
  __macro(rocblas_snrm2)                        \
  __macro(rocblas_dnrm2)                        \
  __macro(rocblas_scnrm2)                       \
  __macro(rocblas_dznrm2)                       \
  __macro(rocblas_sdot)                         \
  __macro(rocblas_ddot)                         \
  __macro(rocblas_cdotu)                        \
  __macro(rocblas_cdotc)                        \
  __macro(rocblas_zdotu)                        \
  __macro(rocblas_zdotc)                        \
  __macro(rocblas_sscal)                        \
  __macro(rocblas_dscal)                        \
  __macro(rocblas_cscal)                        \
  __macro(rocblas_csscal)                       \
  __macro(rocblas_zscal)                        \
  __macro(rocblas_zdscal)                       \
  __macro(rocblas_saxpy)                        \
  __macro(rocblas_daxpy)                        \
  __macro(rocblas_caxpy)                        \
  __macro(rocblas_zaxpy)                        \
  __macro(rocblas_scopy)                        \
  __macro(rocblas_dcopy)                        \
  __macro(rocblas_ccopy)                        \
  __macro(rocblas_zcopy)                        \
  __macro(rocblas_sswap)                        \
  __macro(rocblas_dswap)                        \
  __macro(rocblas_cswap)                        \
  __macro(rocblas_zswap)                        \
  __macro(rocblas_isamax)                       \
  __macro(rocblas_idamax)                       \
  __macro(rocblas_icamax)                       \
  __macro(rocblas_izamax)                       \
  __macro(rocblas_isamin)                       \
  __macro(rocblas_idamin)                       \
  __macro(rocblas_icamin)                       \
  __macro(rocblas_izamin)                       \
  __macro(rocblas_sasum)                        \
  __macro(rocblas_dasum)                        \
  __macro(rocblas_scasum)                       \
  __macro(rocblas_dzasum)                       \
  __macro(rocblas_srot)                         \
  __macro(rocblas_drot)                         \
  __macro(rocblas_crot)                         \
  __macro(rocblas_csrot)                        \
  __macro(rocblas_zrot)                         \
  __macro(rocblas_zdrot)                        \
  __macro(rocblas_srotg)                        \
  __macro(rocblas_drotg)                        \
  __macro(rocblas_crotg)                        \
  __macro(rocblas_zrotg)                        \
  __macro(rocblas_srotm)                        \
  __macro(rocblas_drotm)                        \
  __macro(rocblas_srotmg)                       \
  __macro(rocblas_drotmg)                       \
  __macro(rocblas_sgemv)                        \
  __macro(rocblas_dgemv)                        \
  __macro(rocblas_cgemv)                        \
  __macro(rocblas_zgemv)                        \
  __macro(rocblas_sgbmv)                        \
  __macro(rocblas_dgbmv)                        \
  __macro(rocblas_cgbmv)                        \
  __macro(rocblas_zgbmv)                        \
  __macro(rocblas_strmv)                        \
  __macro(rocblas_dtrmv)                        \
  __macro(rocblas_ctrmv)                        \
  __macro(rocblas_ztrmv)                        \
  __macro(rocblas_stbmv)                        \
  __macro(rocblas_dtbmv)                        \
  __macro(rocblas_ctbmv)                        \
  __macro(rocblas_ztbmv)                        \
  __macro(rocblas_stpmv)                        \
  __macro(rocblas_dtpmv)                        \
  __macro(rocblas_ctpmv)                        \
  __macro(rocblas_ztpmv)                        \
  __macro(rocblas_strsv)                        \
  __macro(rocblas_dtrsv)                        \
  __macro(rocblas_ctrsv)                        \
  __macro(rocblas_ztrsv)                        \
  __macro(rocblas_stpsv)                        \
  __macro(rocblas_dtpsv)                        \
  __macro(rocblas_ctpsv)                        \
  __macro(rocblas_ztpsv)                        \
  __macro(rocblas_stbsv)                        \
  __macro(rocblas_dtbsv)                        \
  __macro(rocblas_ctbsv)                        \
  __macro(rocblas_ztbsv)                        \
  __macro(rocblas_ssymv)                        \
  __macro(rocblas_dsymv)                        \
  __macro(rocblas_csymv)                        \
  __macro(rocblas_zsymv)                        \
  __macro(rocblas_chemv)                        \
  __macro(rocblas_zhemv)                        \
  __macro(rocblas_ssbmv)                        \
  __macro(rocblas_dsbmv)                        \
  __macro(rocblas_chbmv)                        \
  __macro(rocblas_zhbmv)                        \
  __macro(rocblas_sspmv)                        \
  __macro(rocblas_dspmv)                        \
  __macro(rocblas_chpmv)                        \
  __macro(rocblas_zhpmv)                        \
  __macro(rocblas_sger)                         \
  __macro(rocblas_dger)                         \
  __macro(rocblas_cgeru)                        \
  __macro(rocblas_cgerc)                        \
  __macro(rocblas_zgeru)                        \
  __macro(rocblas_zgerc)                        \
  __macro(rocblas_ssyr)                         \
  __macro(rocblas_dsyr)                         \
  __macro(rocblas_csyr)                         \
  __macro(rocblas_zsyr)                         \
  __macro(rocblas_cher)                         \
  __macro(rocblas_zher)                         \
  __macro(rocblas_sspr)                         \
  __macro(rocblas_dspr)                         \
  __macro(rocblas_chpr)                         \
  __macro(rocblas_zhpr)                         \
  __macro(rocblas_ssyr2)                        \
  __macro(rocblas_dsyr2)                        \
  __macro(rocblas_csyr2)                        \
  __macro(rocblas_zsyr2)                        \
  __macro(rocblas_cher2)                        \
  __macro(rocblas_zher2)                        \
  __macro(rocblas_sspr2)                        \
  __macro(rocblas_dspr2)                        \
  __macro(rocblas_chpr2)                        \
  __macro(rocblas_zhpr2)                        \
  __macro(rocblas_sgemm)                        \
  __macro(rocblas_dgemm)                        \
  __macro(rocblas_hgemm)                        \
  __macro(rocblas_cgemm)                        \
  __macro(rocblas_zgemm)                        \
  __macro(rocblas_ssyrk)                        \
  __macro(rocblas_dsyrk)                        \
  __macro(rocblas_csyrk)                        \
  __macro(rocblas_zsyrk)                        \
  __macro(rocblas_cherk)                        \
  __macro(rocblas_zherk)                        \
  __macro(rocblas_ssyr2k)                       \
  __macro(rocblas_dsyr2k)                       \
  __macro(rocblas_csyr2k)                       \
  __macro(rocblas_zsyr2k)                       \
  __macro(rocblas_cher2k)                       \
  __macro(rocblas_zher2k)                       \
  __macro(rocblas_ssyrkx)                       \
  __macro(rocblas_dsyrkx)                       \
  __macro(rocblas_csyrkx)                       \
  __macro(rocblas_zsyrkx)                       \
  __macro(rocblas_cherkx)                       \
  __macro(rocblas_zherkx)                       \
  __macro(rocblas_ssymm)                        \
  __macro(rocblas_dsymm)                        \
  __macro(rocblas_csymm)                        \
  __macro(rocblas_zsymm)                        \
  __macro(rocblas_chemm)                        \
  __macro(rocblas_zhemm)                        \
  __macro(rocblas_strsm)                        \
  __macro(rocblas_dtrsm)                        \
  __macro(rocblas_ctrsm)                        \
  __macro(rocblas_ztrsm)                        \
  __macro(rocblas_strmm)                        \
  __macro(rocblas_dtrmm)                        \
  __macro(rocblas_ctrmm)                        \
  __macro(rocblas_ztrmm)                        \
  __macro(rocblas_sgeam)                        \
  __macro(rocblas_dgeam)                        \
  __macro(rocblas_cgeam)                        \
  __macro(rocblas_zgeam)                        \
  __macro(rocblas_sdgmm)                        \
  __macro(rocblas_ddgmm)                        \
  __macro(rocblas_cdgmm)                        \
  __macro(rocblas_zdgmm)                        \
  __macro(rocblas_sgemm_batched)                \
  __macro(rocblas_dgemm_batched)                \
  __macro(rocblas_cgemm_batched)                \
  __macro(rocblas_zgemm_batched)                \
  __macro(rocblas_hgemm_strided_batched)        \
  __macro(rocblas_sgemm_strided_batched)        \
  __macro(rocblas_dgemm_strided_batched)        \
  __macro(rocblas_cgemm_strided_batched)        \
  __macro(rocblas_zgemm_strided_batched)        \
  __macro(rocblas_gemm_ex)                      \
  __macro(rocblas_gemm_strided_batched_ex)      \
  __macro(rocblas_create_handle)                \
  __macro(rocblas_destroy_handle)               \
  __macro(rocblas_set_stream)

// clang-format on

FOREACH_ROCBLAS_API(ROCBLAS_API_WRAPPER)

}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCBLAS_WRAPPER_H_
