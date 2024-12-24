/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_

#include "rocm/include/hip/hip_runtime.h"
#include "rocm/rocm_config.h"
#include "tsl/platform/dso_loader.h"
#include "tsl/platform/env.h"

namespace stream_executor {
namespace wrap {
#ifdef PLATFORM_GOOGLE
// Use static linked library
#define STREAM_EXECUTOR_HIP_WRAP(hipSymbolName)                            \
  template <typename... Args>                                              \
  auto hipSymbolName(Args... args) -> decltype(::hipSymbolName(args...)) { \
    return ::hipSymbolName(args...);                                       \
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
  auto hipSymbolName(Args... args) -> decltype(::hipSymbolName(args...)) {  \
    using FuncPtrT = std::add_pointer<decltype(::hipSymbolName)>::type;     \
    static FuncPtrT loaded = []() -> FuncPtrT {                             \
      static const char *kName = TO_STR(hipSymbolName);                     \
      void *f;                                                              \
      auto s = tsl::Env::Default()->GetSymbolFromLibrary(                   \
          tsl::internal::CachedDsoLoader::GetHipDsoHandle().value(), kName, \
          &f);                                                              \
      CHECK(s.ok()) << "could not find " << kName                           \
                    << " in HIP DSO; dlerror: " << s.message();             \
      return reinterpret_cast<FuncPtrT>(f);                                 \
    }();                                                                    \
    return loaded(args...);                                                 \
  }
#endif

// clang-format off
// IMPORTANT: if you add a new HIP API to this list, please notify
// the rocm-profiler developers to track the API traces.
#define HIP_ROUTINE_EACH(__macro)                   \
  __macro(hipCtxGetDevice)                          \
  __macro(hipCtxSetCurrent)                         \
  __macro(hipCtxEnablePeerAccess)                   \
  __macro(hipDeviceCanAccessPeer)                   \
  __macro(hipDeviceEnablePeerAccess)                \
  __macro(hipDeviceGet)                             \
  __macro(hipDeviceGetAttribute)                    \
  __macro(hipDeviceGetName)                         \
  __macro(hipDeviceGetPCIBusId)                     \
  __macro(hipDeviceGetSharedMemConfig)              \
  __macro(hipDeviceGetStreamPriorityRange)          \
  __macro(hipDeviceGraphMemTrim)                    \
  __macro(hipDevicePrimaryCtxGetState)              \
  __macro(hipDevicePrimaryCtxSetFlags)              \
  __macro(hipDevicePrimaryCtxRetain)                \
  __macro(hipDevicePrimaryCtxRelease)               \
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
  __macro(hipFuncGetAttribute)                      \
  __macro(hipFuncSetAttribute)                      \
  __macro(hipGetDevice)                             \
  __macro(hipGetDeviceCount)                        \
  __macro(hipGetDeviceProperties)                   \
  __macro(hipGetErrorString)                        \
  __macro(hipGraphAddKernelNode)                    \
  __macro(hipGraphAddChildGraphNode)                \
  __macro(hipGraphAddEmptyNode)                     \
  __macro(hipGraphAddMemAllocNode)                  \
  __macro(hipGraphAddMemcpyNode1D)                  \
  __macro(hipGraphAddMemsetNode)                    \
  __macro(hipGraphAddMemFreeNode)                   \
  __macro(hipGraphCreate)                           \
  __macro(hipGraphDebugDotPrint)                    \
  __macro(hipGraphDestroy)                          \
  __macro(hipGraphGetNodes)                         \
  __macro(hipGraphExecChildGraphNodeSetParams)      \
  __macro(hipGraphExecDestroy)                      \
  __macro(hipGraphExecKernelNodeSetParams)          \
  __macro(hipGraphExecMemcpyNodeSetParams1D)        \
  __macro(hipGraphExecMemsetNodeSetParams)          \
  __macro(hipGraphExecUpdate)                       \
  __macro(hipGraphInstantiate)                      \
  __macro(hipGraphMemAllocNodeGetParams)            \
  __macro(hipGraphLaunch)                           \
  __macro(hipGraphNodeGetType)                      \
  __macro(hipGraphNodeSetEnabled)                   \
  __macro(hipHostFree)                              \
  __macro(hipHostMalloc)                            \
  __macro(hipHostRegister)                          \
  __macro(hipHostUnregister)                        \
  __macro(hipInit)                                  \
  __macro(hipKernelNameRefByPtr)                    \
  __macro(hipLaunchHostFunc)                        \
  __macro(hipLaunchKernel)                          \
  __macro(hipMalloc)                                \
  __macro(hipMallocManaged)                         \
  __macro(hipMemGetAddressRange)                    \
  __macro(hipMemGetInfo)                            \
  __macro(hipMemcpyDtoD)                            \
  __macro(hipMemcpyDtoDAsync)                       \
  __macro(hipMemcpyDtoH)                            \
  __macro(hipMemcpyDtoHAsync)                       \
  __macro(hipMemcpyHtoD)                            \
  __macro(hipMemcpyHtoDAsync)                       \
  __macro(hipMemset)                                \
  __macro(hipMemsetD8)                              \
  __macro(hipMemsetD16)                             \
  __macro(hipMemsetD32)                             \
  __macro(hipMemsetAsync)                           \
  __macro(hipMemsetD8Async)                         \
  __macro(hipMemsetD16Async)                        \
  __macro(hipMemsetD32Async)                        \
  __macro(hipModuleGetFunction)                     \
  __macro(hipModuleGetGlobal)                       \
  __macro(hipModuleLaunchKernel)                    \
  __macro(hipModuleLoadData)                        \
  __macro(hipModuleUnload)                          \
  __macro(hipModuleOccupancyMaxActiveBlocksPerMultiprocessor) \
  __macro(hipModuleOccupancyMaxPotentialBlockSize)  \
  __macro(hipPointerGetAttribute)                   \
  __macro(hipPointerGetAttributes)                  \
  __macro(hipRuntimeGetVersion)                     \
  __macro(hipSetDevice)                             \
  __macro(hipStreamAddCallback)                     \
  __macro(hipStreamBeginCapture)                    \
  __macro(hipStreamCreateWithFlags)                 \
  __macro(hipStreamCreateWithPriority)              \
  __macro(hipStreamDestroy)                         \
  __macro(hipStreamEndCapture)                      \
  __macro(hipStreamIsCapturing)                     \
  __macro(hipStreamQuery)                           \
  __macro(hipStreamSynchronize)                     \
  __macro(hipStreamWaitEvent)  // clang-format on

HIP_ROUTINE_EACH(STREAM_EXECUTOR_HIP_WRAP)

#if TF_ROCM_VERSION >= 60200

// clang-format off
#define HIP_ROUTINE_EACH_62(__macro)            \
  __macro(hipGetFuncBySymbol)                   \
  __macro(hipStreamBeginCaptureToGraph)
// clang-format on

HIP_ROUTINE_EACH_62(STREAM_EXECUTOR_HIP_WRAP)

#undef HIP_ROUTINE_EACH_62
#endif  // TF_ROCM_VERSION >= 60200

#undef HIP_ROUTINE_EACH
#undef STREAM_EXECUTOR_HIP_WRAP
#undef TO_STR
#undef TO_STR_

}  // namespace wrap
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DRIVER_WRAPPER_H_
