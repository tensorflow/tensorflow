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

// This file wraps rocm driver calls with dso loader so that we don't need to
// have explicit linking to librocm. All TF rocm driver usage should route
// through this wrapper.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_

#define __HIP_DISABLE_CPP_FUNCTIONS__

#include "rocm/include/hip/hip_runtime.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/platform/dso_loader.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace tensorflow {
namespace wrap {
#ifdef PLATFORM_GOOGLE
// Use static linked library
#define STREAM_EXECUTOR_HIP_WRAP(hipSymbolName)                          \
  template <typename... Args>                                            \
  auto hipSymbolName(Args... args)->decltype(::hipSymbolName(args...)) { \
    return ::hipSymbolName(args...);                                     \
  }

// This macro wraps a global identifier, given by hipSymbolName, in a callable
// structure that loads the DLL symbol out of the DSO handle in a thread-safe
// manner on first use. This dynamic loading technique is used to avoid DSO
// dependencies on vendor libraries which may or may not be available in the
// deployed binary environment.
#else
#define TO_STR_(x) #x
#define TO_STR(x) TO_STR_(x)

#define STREAM_EXECUTOR_HIP_WRAP(hipSymbolName)                             \
  template <typename... Args>                                               \
  auto hipSymbolName(Args... args)->decltype(::hipSymbolName(args...)) {    \
    using FuncPtrT = std::add_pointer<decltype(::hipSymbolName)>::type;     \
    static FuncPtrT loaded = []() -> FuncPtrT {                             \
      static const char *kName = TO_STR(hipSymbolName);                     \
      void *f;                                                              \
      auto s = stream_executor::port::Env::Default()->GetSymbolFromLibrary( \
          stream_executor::internal::CachedDsoLoader::GetHipDsoHandle()     \
              .ValueOrDie(),                                                \
          kName, &f);                                                       \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in HIP DSO; dlerror: " << s.error_message();       \
      return reinterpret_cast<FuncPtrT>(f);                                 \
    }();                                                                    \
    return loaded(args...);                                                 \
  }
#endif

// clang-format off
#define HIP_ROUTINE_EACH(__macro)                   \
  __macro(hipDeviceCanAccessPeer)                   \
  __macro(hipDeviceEnablePeerAccess)                \
  __macro(hipDeviceGet)                             \
  __macro(hipDeviceGetAttribute)                    \
  __macro(hipDeviceGetName)                         \
  __macro(hipDeviceGetPCIBusId)                     \
  __macro(hipDeviceGetSharedMemConfig)              \
  __macro(hipDeviceSetSharedMemConfig)              \
  __macro(hipDeviceSynchronize)                     \
  __macro(hipDeviceTotalMem)                        \
  __macro(hipDriverGetVersion)                      \
  __macro(hipEventCreateWithFlags)                  \
  __macro(hipEventDestroy)                          \
  __macro(hipEventElapsedTime)                      \
  __macro(hipEventQuery)                            \
  __macro(hipEventRecord)                           \
  __macro(hipEventSynchronize)                      \
  __macro(hipFree)                                  \
  __macro(hipFuncSetCacheConfig)                    \
  __macro(hipGetDevice)                             \
  __macro(hipGetDeviceCount)                        \
  __macro(hipGetDeviceProperties)                   \
  __macro(hipHostFree)                              \
  __macro(hipHostMalloc)                            \
  __macro(hipHostRegister)                          \
  __macro(hipHostUnregister)                        \
  __macro(hipInit)                                  \
  __macro(hipMalloc)                                \
  __macro(hipMemGetAddressRange)                    \
  __macro(hipMemGetInfo)                            \
  __macro(hipMemcpyDtoD)                            \
  __macro(hipMemcpyDtoDAsync)                       \
  __macro(hipMemcpyDtoH)                            \
  __macro(hipMemcpyDtoHAsync)                       \
  __macro(hipMemcpyHtoD)                            \
  __macro(hipMemcpyHtoDAsync)                       \
  __macro(hipMemset)                                \
  __macro(hipMemsetD32)                             \
  __macro(hipMemsetD8)                              \
  __macro(hipMemsetAsync)                           \
  __macro(hipMemsetD32Async)                        \
  __macro(hipModuleGetFunction)                     \
  __macro(hipModuleGetGlobal)                       \
  __macro(hipModuleLaunchKernel)                    \
  __macro(hipModuleLoadData)                        \
  __macro(hipModuleUnload)                          \
  __macro(hipPointerGetAttributes)                  \
  __macro(hipSetDevice)                             \
  __macro(hipStreamAddCallback)                     \
  __macro(hipStreamCreateWithFlags)                 \
  __macro(hipStreamCreateWithPriority)		    \
  __macro(hipStreamDestroy)                         \
  __macro(hipStreamQuery)                           \
  __macro(hipStreamSynchronize)                     \
  __macro(hipStreamWaitEvent)                       \
// clang-format on

HIP_ROUTINE_EACH(STREAM_EXECUTOR_HIP_WRAP)
#undef HIP_ROUTINE_EACH
#undef STREAM_EXECUTOR_HIP_WRAP
#undef TO_STR
#undef TO_STR_
}  // namespace wrap
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
