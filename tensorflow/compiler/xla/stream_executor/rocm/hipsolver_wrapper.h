/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file wraps hipsolver API calls with dso loader so that we don't need to
// have explicit linking to libhipsolver. All TF hipsolver API usage should
// route through this wrapper.

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIPSOLVER_WRAPPER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIPSOLVER_WRAPPER_H_

#include "rocm/rocm_config.h"

#if TF_ROCM_VERSION >= 40500

#include "rocm/include/hipsolver.h"
#include "tensorflow/compiler/xla/stream_executor/lib/env.h"
#include "tensorflow/compiler/xla/stream_executor/platform/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/platform/port.h"

namespace stream_executor {
namespace wrap {

#ifdef PLATFORM_GOOGLE

#define HIPSOLVER_API_WRAPPER(api_name)                        \
  template <typename... Args>                                  \
  auto api_name(Args... args)->decltype(::api_name(args...)) { \
    return ::api_name(args...);                                \
  }

#else

#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)

#define HIPSOLVER_API_WRAPPER(api_name)                                       \
  template <typename... Args>                                                 \
  auto api_name(Args... args)->decltype(::api_name(args...)) {                \
    using FuncPtrT = std::add_pointer<decltype(::api_name)>::type;            \
    static FuncPtrT loaded = []() -> FuncPtrT {                               \
      static const char* kName = TO_STR(api_name);                            \
      void* f;                                                                \
      auto s = stream_executor::port::Env::Default()->GetSymbolFromLibrary(   \
          stream_executor::internal::CachedDsoLoader::GetHipsolverDsoHandle() \
              .value(),                                                  \
          kName, &f);                                                         \
      CHECK(s.ok()) << "could not find " << kName                             \
                    << " in hipsolver lib; dlerror: " << s.error_message();   \
      return reinterpret_cast<FuncPtrT>(f);                                   \
    }();                                                                      \
    return loaded(args...);                                                   \
  }

#endif

// clang-format off
#define FOREACH_HIPSOLVER_API(__macro)       \
  __macro(hipsolverCreate)                   \
  __macro(hipsolverDestroy)                  \
  __macro(hipsolverSetStream)                \
  __macro(hipsolverCgetrf)                   \
  __macro(hipsolverCgetrf_bufferSize)        \
  __macro(hipsolverDgetrf)                   \
  __macro(hipsolverDgetrf_bufferSize)        \
  __macro(hipsolverSgetrf)                   \
  __macro(hipsolverSgetrf_bufferSize)        \
  __macro(hipsolverZgetrf)                   \
  __macro(hipsolverZgetrf_bufferSize)        \
  __macro(hipsolverCgetrs)                   \
  __macro(hipsolverCgetrs_bufferSize)        \
  __macro(hipsolverDgetrs)                   \
  __macro(hipsolverDgetrs_bufferSize)        \
  __macro(hipsolverSgetrs)                   \
  __macro(hipsolverSgetrs_bufferSize)        \
  __macro(hipsolverZgetrs)                   \
  __macro(hipsolverZgetrs_bufferSize)        \
  __macro(hipsolverCpotrf)                   \
  __macro(hipsolverCpotrf_bufferSize)        \
  __macro(hipsolverDpotrf)                   \
  __macro(hipsolverDpotrf_bufferSize)        \
  __macro(hipsolverSpotrf)                   \
  __macro(hipsolverSpotrf_bufferSize)        \
  __macro(hipsolverZpotrf)                   \
  __macro(hipsolverZpotrf_bufferSize)        \
  __macro(hipsolverCpotrfBatched)            \
  __macro(hipsolverCpotrfBatched_bufferSize) \
  __macro(hipsolverDpotrfBatched)            \
  __macro(hipsolverDpotrfBatched_bufferSize) \
  __macro(hipsolverSpotrfBatched)            \
  __macro(hipsolverSpotrfBatched_bufferSize) \
  __macro(hipsolverZpotrfBatched)            \
  __macro(hipsolverZpotrfBatched_bufferSize) \
  __macro(hipsolverCgeqrf)                   \
  __macro(hipsolverCgeqrf_bufferSize)        \
  __macro(hipsolverDgeqrf)                   \
  __macro(hipsolverDgeqrf_bufferSize)        \
  __macro(hipsolverSgeqrf)                   \
  __macro(hipsolverSgeqrf_bufferSize)        \
  __macro(hipsolverZgeqrf)                   \
  __macro(hipsolverZgeqrf_bufferSize)        \
  __macro(hipsolverCunmqr)                   \
  __macro(hipsolverCunmqr_bufferSize)        \
  __macro(hipsolverZunmqr)                   \
  __macro(hipsolverZunmqr_bufferSize)        \
  __macro(hipsolverCungqr)                   \
  __macro(hipsolverCungqr_bufferSize)        \
  __macro(hipsolverZungqr)                   \
  __macro(hipsolverZungqr_bufferSize)        \
  __macro(hipsolverCheevd)                   \
  __macro(hipsolverCheevd_bufferSize)        \
  __macro(hipsolverZheevd)                   \
  __macro(hipsolverZheevd_bufferSize)
// clang-format on

FOREACH_HIPSOLVER_API(HIPSOLVER_API_WRAPPER)

#undef TO_STR_
#undef TO_STR
#undef FOREACH_HIPSOLVER_API
#undef HIPSOLVER_API_WRAPPER

}  // namespace wrap
}  // namespace stream_executor

#endif  // TF_ROCM_VERSION
#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_ROCM_HIPSOLVER_WRAPPER_H_
