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

#include "xla/service/gpu/kernels/cutlass_gemm.h"

#include <cstdint>
#include <optional>
#include <string>

#include "tsl/platform/logging.h"

#if !defined(PLATFORM_WINDOWS)
#include <dlfcn.h>
#endif

namespace xla::gpu::kernel::gemm_universal {

// TODO(b/315492043): We should add an XLA PJRT style C API for registering
// libraries of custom CUTLASS kernels compiled into shared libraries. It should
// be possible to bundle multiple custom CUTLASS kernels into a single shared
// library, and then load them optionally by name. For now we assume that there
// is a 1-to-1 mapping from a kernel to shared library, and they exported with
// a simple C API, and we hope that functions exported from a library has ABI
// that matches our expectations.

using BlockDimFn = void (*)(int32_t m, int32_t n, int32_t k, uint32_t* x,
                            uint32_t* y, uint32_t* z);
using ThreadDimFn = void (*)(uint32_t* x, uint32_t* y, uint32_t* z);
using SharedMemoryBytesFn = int32_t (*)();
using CanImplementFn = bool (*)(int32_t m, int32_t n, int32_t k);
using WorkspaceSizeFn = int64_t (*)(int32_t m, int32_t n, int32_t k);
using InitializeFn = void (*)(void* params, int32_t m, int32_t n, int32_t k,
                              void* lhs, void* rhs, void* out, void* workspace,
                              int32_t* out_offset, int32_t device_sms,
                              int32_t sm_occupancy);
using KernelSymboFn = void* (*)();

static constexpr const char* kBlockDimFn = "xla_cutlass_kernel_block_dim";
static constexpr const char* kThreadDimFn = "xla_cutlass_kernel_thread_dim";
static constexpr const char* kSharedMemoryBytes =
    "xla_cutlass_kernel_shared_memory_bytes";
static constexpr const char* kCanImplement = "xla_cutlass_kernel_can_implement";
static constexpr const char* kWorkspaceSize =
    "xla_cutlass_kernel_workspace_size";
static constexpr const char* kInitialize = "xla_cutlass_kernel_initialize";
static constexpr const char* kKernelSymbol = "xla_cutlass_kernel_symbol";

static void* Dlopen(const char* path) {
#if defined(PLATFORM_WINDOWS)
  return nullptr;
#else
  return dlopen(path, RTLD_LAZY);
#endif  // defined(PLATFORM_WINDOWS)
}

static void* Dlsym(void* handle, const char* name) {
#if defined(PLATFORM_WINDOWS)
  return nullptr;
#else
  return dlsym(handle, name);
#endif  // defined(PLATFORM_WINDOWS)
}

//===----------------------------------------------------------------------===//
// CUTLASS Host Side Adaptor
//===----------------------------------------------------------------------===//

std::optional<Adaptor<DlOpenedKernel>> Adaptor<DlOpenedKernel>::Load(
    const std::string& path) {
  VLOG(3) << "Load CUTLASS adaptor from a shared library: " << path;

  void* library = Dlopen(path.c_str());
  if (library == nullptr) return std::nullopt;

  auto resolve = [&](const char* name) -> void* {
    void* sym = Dlsym(library, name);
    if (sym == nullptr) {
      LOG(ERROR) << "Failed to resolve CUTLASS adaptor function: " << name
                 << " in library: " << path;
    }
    return sym;
  };

  void* block_dim_fn = resolve(kBlockDimFn);
  if (block_dim_fn == nullptr) return std::nullopt;

  void* thread_dim_fn = resolve(kThreadDimFn);
  if (thread_dim_fn == nullptr) return std::nullopt;

  void* shared_memory_bytes_fn = resolve(kSharedMemoryBytes);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  void* can_implement_fn = resolve(kCanImplement);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  void* workspace_size_fn = resolve(kWorkspaceSize);
  if (workspace_size_fn == nullptr) return std::nullopt;

  void* initialize_fn = resolve(kInitialize);
  if (shared_memory_bytes_fn == nullptr) return std::nullopt;

  return Adaptor(library, block_dim_fn, thread_dim_fn, shared_memory_bytes_fn,
                 can_implement_fn, workspace_size_fn, initialize_fn);
}

std::optional<Dim3> Adaptor<DlOpenedKernel>::ClusterDim() const {
  return std::nullopt;
}

Dim3 Adaptor<DlOpenedKernel>::BlockDim(int32_t m, int32_t n, int32_t k) const {
  Dim3 dim;
  reinterpret_cast<BlockDimFn>(block_dim_fn_)(m, n, k, &dim.x, &dim.y, &dim.z);
  return dim;
}

Dim3 Adaptor<DlOpenedKernel>::ThreadDim() const {
  Dim3 dim;
  reinterpret_cast<ThreadDimFn>(thread_dim_fn_)(&dim.x, &dim.y, &dim.z);
  return dim;
}

int32_t Adaptor<DlOpenedKernel>::SharedMemoryBytes() const {
  return reinterpret_cast<SharedMemoryBytesFn>(shared_memory_bytes_fn_)();
}

bool Adaptor<DlOpenedKernel>::CanImplement(const Arguments& args) const {
  return reinterpret_cast<CanImplementFn>(can_implement_fn_)(args.m, args.n,
                                                             args.k);
}

int64_t Adaptor<DlOpenedKernel>::WorkspaceSize(const Arguments& args) const {
  return reinterpret_cast<WorkspaceSizeFn>(workspace_size_fn_)(args.m, args.n,
                                                               args.k);
}

void Adaptor<DlOpenedKernel>::Initialize(void* params, const Arguments& args,
                                         int32_t device_sms,
                                         int32_t sm_occupancy) const {
  reinterpret_cast<InitializeFn>(initialize_fn_)(
      params, args.m, args.n, args.k, args.lhs, args.rhs, args.out,
      args.workspace, args.slices.out, device_sms, sm_occupancy);
}

Adaptor<DlOpenedKernel>::Adaptor(void* handle, void* block_dim_fn,
                                 void* thread_dim_fn,
                                 void* shared_memory_bytes_fn,
                                 void* can_implement_fn,
                                 void* workspace_size_fn, void* initialize_fn)
    : handle_(handle),
      block_dim_fn_(block_dim_fn),
      thread_dim_fn_(thread_dim_fn),
      shared_memory_bytes_fn_(shared_memory_bytes_fn),
      can_implement_fn_(can_implement_fn),
      workspace_size_fn_(workspace_size_fn),
      initialize_fn_(initialize_fn) {}

//===----------------------------------------------------------------------===//
// CUTLASS Device Side Adaptor
//===----------------------------------------------------------------------===//

std::optional<DeviceKernel<DlOpenedKernel>> DeviceKernel<DlOpenedKernel>::Load(
    const std::string& path) {
  VLOG(3) << "Load CUTLASS device kernel from a shared library: " << path;

  void* library = Dlopen(path.c_str());
  if (library == nullptr) return std::nullopt;

  auto resolve = [&](const char* name) -> void* {
    void* sym = Dlsym(library, name);
    if (sym == nullptr) {
      LOG(ERROR) << "Failed to resolve CUTLASS kernel function: " << name
                 << " in library: " << path;
    }
    return sym;
  };

  void* kernel_symbol_fn = resolve(kKernelSymbol);
  if (kernel_symbol_fn == nullptr) return std::nullopt;

  return DeviceKernel(library, kernel_symbol_fn);
}

void* DeviceKernel<DlOpenedKernel>::symbol() const {
  return reinterpret_cast<KernelSymboFn>(symbol_fn_)();
}

DeviceKernel<DlOpenedKernel>::DeviceKernel(void* handle, void* symbol_fn)
    : handle_(handle), symbol_fn_(symbol_fn) {}

}  // namespace xla::gpu::kernel::gemm_universal
