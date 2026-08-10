/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/cudart_kernel_registry.h"

#include <elf.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

namespace stream_executor::cuda {
namespace {

// Standard uncompressed CUDA Fatbinary container magic number (NVCC).
constexpr uint32_t kFatbinMagicUncompressed = 0xba55d10a;

// Compressed CUDA Fatbinary container magic number (CUDA 11+ / 12+ NVCC).
constexpr uint32_t kFatbinMagicCompressed = 0xba55ed50;

// Legacy/alternative CUDA Fatbinary payload header magic number.
constexpr uint32_t kFatbinMagicLegacy = 0x00101001;

// CUDA fatbin wrapper structure passed to __cudaRegisterFatBinary.
struct FatbinWrapper {
  uint32_t magic;
  uint32_t version;
  const void* data;
  void* filename_or_fatbins;
};

struct FatHeader {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t fat_size;
};

// Helper function to parse CUDA Fatbinary containers or raw ELF CUBINs from
// a FatbinWrapper pointer and return an absl::Span pointing to the binary
// payload.
std::optional<absl::Span<const uint8_t>> ParseFatBinaryOrElf(
    const void* fat_cubin) {
  if (fat_cubin == nullptr) {
    return std::nullopt;
  }

  const auto* wrapper = static_cast<const FatbinWrapper*>(fat_cubin);
  if (wrapper->data == nullptr) {
    return std::nullopt;
  }

  const uint8_t* data_bytes = static_cast<const uint8_t*>(wrapper->data);
  size_t total_size = 0;
  const auto* header = reinterpret_cast<const FatHeader*>(data_bytes);
  if (header->magic == kFatbinMagicUncompressed ||
      header->magic == kFatbinMagicCompressed) {
    total_size = static_cast<size_t>(header->header_size) +
                 static_cast<size_t>(header->fat_size);
  } else if (header->magic == kFatbinMagicLegacy) {
    total_size =
        static_cast<size_t>(*reinterpret_cast<const uint64_t*>(data_bytes + 8));
  } else if (data_bytes[0] == 0x7f && data_bytes[1] == 'E' &&
             data_bytes[2] == 'L' && data_bytes[3] == 'F') {
    const auto* elf_header = reinterpret_cast<const Elf64_Ehdr*>(data_bytes);
    total_size = static_cast<size_t>(
        elf_header->e_shoff +
        (static_cast<uint64_t>(elf_header->e_shnum) * elf_header->e_shentsize));
  } else {
    return std::nullopt;
  }

  return absl::Span<const uint8_t>(data_bytes, total_size);
}

class KernelRegistry {
 public:
  void RegisterFatBinary(void** handle, const void* fat_cubin);
  void RegisterCudaRuntimeKernel(void** handle, const void* host_fun,
                                 absl::string_view name);
  absl::StatusOr<CudaRuntimeKernel> FindCudaRuntimeKernel(
      const void* host_fun) const;

 private:
  struct RegisteredKernel {
    void** fatbin_handle;
    std::string name;
  };

  mutable absl::Mutex mutex_;
  absl::flat_hash_map<void**, absl::Span<const uint8_t>> fatbin_map_
      ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<const void*, RegisteredKernel> kernel_map_
      ABSL_GUARDED_BY(mutex_);
};

void KernelRegistry::RegisterFatBinary(void** handle, const void* fat_cubin) {
  if (handle == nullptr || fat_cubin == nullptr) {
    return;
  }
  auto span_opt = ParseFatBinaryOrElf(fat_cubin);
  if (!span_opt.has_value()) {
    return;
  }
  absl::MutexLock lock(mutex_);
  fatbin_map_[handle] = *span_opt;
}

void KernelRegistry::RegisterCudaRuntimeKernel(void** handle,
                                               const void* host_fun,
                                               absl::string_view name) {
  if (host_fun == nullptr) {
    return;
  }
  absl::MutexLock lock(mutex_);
  kernel_map_[host_fun] = {handle, std::string(name)};
}

absl::StatusOr<CudaRuntimeKernel> KernelRegistry::FindCudaRuntimeKernel(
    const void* host_fun) const {
  if (host_fun == nullptr) {
    return absl::InvalidArgumentError("Host function pointer cannot be null");
  }
  absl::MutexLock lock(mutex_);

  auto k_it = kernel_map_.find(host_fun);
  if (k_it == kernel_map_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Kernel with host function pointer %p not found in CUDA runtime kernel "
        "registry. Did you use the embeddable_cuda_library Bazel macro?",
        host_fun));
  }

  void** fat_handle = k_it->second.fatbin_handle;
  auto f_it = fatbin_map_.find(fat_handle);
  if (f_it == fatbin_map_.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Fatbinary for kernel '%s' (handle %p) not found in CUDA runtime "
        "kernel registry",
        k_it->second.name, fat_handle));
  }

  return CudaRuntimeKernel{f_it->second, k_it->second.name};
}

KernelRegistry& GetKernelRegistry() {
  static absl::NoDestructor<KernelRegistry> registry;
  return *registry;
}

}  // namespace

absl::StatusOr<CudaRuntimeKernel> FindCudaRuntimeKernel(const void* host_fun) {
  return GetKernelRegistry().FindCudaRuntimeKernel(host_fun);
}

}  // namespace stream_executor::cuda

extern "C" {

// Declarations of symbols we want to wrap.
void** __cudaRegisterFatBinary(void* fat_cubin);
void __cudaRegisterFunction(void** fat_cubin_handle, const char* host_fun,
                            char* device_fun, const char* device_name,
                            int thread_limit, void* tid, void* bid, void* b_dim,
                            void* g_dim, int* w_size);

// When this library is linked as a shared object (.so), the linker might omit
// wrapping of __cudaRegisterFatBinary/__cudaRegisterFunction if there are no
// references to them in the library. This dummy function forces references to
// these symbols, ensuring the linker applies the --wrap flags and resolves
// the corresponding __real_ symbols. We call this from the wrapper to prevent
// the compiler/linker from optimizing it away.
uintptr_t dummy_use_to_force_wrap() {
  volatile uintptr_t p1 = reinterpret_cast<uintptr_t>(&__cudaRegisterFatBinary);
  volatile uintptr_t p2 = reinterpret_cast<uintptr_t>(&__cudaRegisterFunction);
  return p1 + p2;
}

// Declaration of real symbols provided by CUDA runtime.
void** __real___cudaRegisterFatBinary(void* fat_cubin);
void __real___cudaRegisterFunction(void** fat_cubin_handle,
                                   const char* host_fun, char* device_fun,
                                   const char* device_name, int thread_limit,
                                   void* tid, void* bid, void* b_dim,
                                   void* g_dim, int* w_size);

// Linker wrapper intercepting calls to __cudaRegisterFatBinary.
void** __wrap___cudaRegisterFatBinary(void* fat_cubin) {
  uintptr_t dummy = dummy_use_to_force_wrap();
  (void)dummy;
  void** handle = __real___cudaRegisterFatBinary(fat_cubin);
  ::stream_executor::cuda::GetKernelRegistry().RegisterFatBinary(handle,
                                                                 fat_cubin);
  return handle;
}

// Linker wrapper intercepting calls to __cudaRegisterFunction.
void __wrap___cudaRegisterFunction(void** fat_cubin_handle,
                                   const char* host_fun, char* device_fun,
                                   const char* device_name, int thread_limit,
                                   void* tid, void* bid, void* b_dim,
                                   void* g_dim, int* w_size) {
  ::stream_executor::cuda::GetKernelRegistry().RegisterCudaRuntimeKernel(
      fat_cubin_handle, reinterpret_cast<const void*>(host_fun),
      device_name != nullptr ? device_name : "");

  __real___cudaRegisterFunction(fat_cubin_handle, host_fun, device_fun,
                                device_name, thread_limit, tid, bid, b_dim,
                                g_dim, w_size);
}

}  // extern "C"
