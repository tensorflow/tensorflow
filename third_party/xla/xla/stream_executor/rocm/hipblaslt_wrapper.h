/* Copyright 2023 The OpenXLA Authors.
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

// This file wraps rocsolver API calls with dso loader so that we don't need to
// have explicit linking to librocsolver. All TF hipsarse API usage should route
// through this wrapper.

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_WRAPPER_H_

#define __HIP_DISABLE_CPP_FUNCTIONS__

#include "rocm/rocm_config.h"

#if TF_HIPBLASLT
#if TF_ROCM_VERSION >= 50500
#include "rocm/include/hipblaslt/hipblaslt.h"
#else
#include "rocm/include/hipblaslt.h"
#endif
#include "xla/stream_executor/platform/dso_loader.h"
#include "xla/stream_executor/platform/port.h"
#include "tsl/platform/env.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define HIPBLASLT_API_WRAPPER(api_name)                          \
  template <typename... Args>                                    \
  auto api_name(Args... args) -> decltype(::api_name(args...)) { \
    return ::api_name(args...);                                  \
  }

#else

#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)

#define HIPBLASLT_API_WRAPPER(api_name)                                       \
  template <typename... Args>                                                 \
  auto api_name(Args... args) -> decltype(::api_name(args...)) {              \
    using FuncPtrT = std::add_pointer<decltype(::api_name)>::type;            \
    static FuncPtrT loaded = []() -> FuncPtrT {                               \
      static const char* kName = TO_STR(api_name);                            \
      void* f;                                                                \
      auto s = tsl::Env::Default() -> GetSymbolFromLibrary(                   \
          stream_executor::internal::CachedDsoLoader::GetHipblasltDsoHandle() \
              .value(),                                                       \
          kName, &f);                                                         \
      CHECK(s.ok()) << "could not find " << kName                             \
                    << " in hipblaslt lib; dlerror: " << s.message();         \
      return reinterpret_cast<FuncPtrT>(f);                                   \
    }();                                                                      \
    return loaded(args...);                                                   \
  }

#endif

// clang-format off
#define FOREACH_HIPBLASLT_API(__macro)      \
  __macro(hipblasLtCreate) \
  __macro(hipblasLtDestroy) \
  __macro(hipblasLtMatmulPreferenceCreate) \
  __macro(hipblasLtMatmulPreferenceSetAttribute) \
  __macro(hipblasLtMatmulPreferenceDestroy) \
  __macro(hipblasLtMatmulDescSetAttribute) \
  __macro(hipblasLtMatmulDescGetAttribute) \
  __macro(hipblasLtMatmulAlgoGetHeuristic) \
  __macro(hipblasLtMatrixLayoutCreate) \
  __macro(hipblasLtMatrixLayoutDestroy) \
  __macro(hipblasLtMatrixLayoutSetAttribute) \
  __macro(hipblasLtMatrixLayoutGetAttribute) \
  __macro(hipblasLtMatmulDescCreate) \
  __macro(hipblasLtMatmulDescDestroy) \
  __macro(hipblasLtMatmul) \
  __macro(hipblasStatusToString)
// clang-format on

FOREACH_HIPBLASLT_API(HIPBLASLT_API_WRAPPER)

#undef TO_STR_
#undef TO_STR
#undef FOREACH_HIPBLASLT_API
#undef HIPBLASLT_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // TF_HIPBLASLT

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIPBLASLT_WRAPPER_H_
