/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_KERNEL_SYMBOL_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_KERNEL_SYMBOL_REGISTRY_H_

#include <string>
#include <tuple>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"

namespace stream_executor {

/**
 * A registry for GPU kernel symbols.
 *
 * We use void* pointers to the host entry functions of CUDA C++ kernels to
 * identify them and load their GPU implementation into the GPU.
 *
 * This registry allows us to do this consistently and reliably across different
 * processes by mapping the host entry function to a persistent name and using
 * that name to look up the host entry function pointer in the registry.
 *
 * Maps a (name, platform_id) tuple to a void* pointer.
 *
 * You can use the KERNEL_SYMBOL_REGISTRY_REGISTER_SYMBOL_STATICALLY macro to
 * register symbols during initialization. Make sure to mark the target as
 * `alwayslink = True` so that it won't be stripped by the linker.
 *
 * The class is thread-safe.
 */
class KernelSymbolRegistry {
 public:
  static KernelSymbolRegistry& GetGlobalInstance();

  // Registers a symbol in the registry. The symbol is identified by a name and
  // a platform ID.
  //
  // Returns an error if the symbol is already registered.
  absl::Status RegisterSymbol(absl::string_view name, Platform::Id platform_id,
                              void* symbol);

  // Convenience overload for registering any raw function pointer as a symbol.
  template <typename... Args>
  absl::Status RegisterSymbol(absl::string_view name, Platform::Id platform_id,
                              void (*symbol)(Args...)) {
    return RegisterSymbol(name, platform_id, absl::bit_cast<void*>(symbol));
  }

  absl::StatusOr<void*> FindSymbol(absl::string_view name,
                                   Platform::Id platform_id) const;

 private:
  mutable absl::Mutex mutex_;
  using RegistryKey = std::tuple<std::string, Platform::Id>;
  absl::flat_hash_map<RegistryKey, void*> symbols_ ABSL_GUARDED_BY(mutex_);
};

// Registers a symbol in the kernel symbol registry.
//
// This macro registers a symbol in the kernel symbol registry during static
// initialization. It uses the identifier to generate a unique name for the
// symbol and logs a fatal error if the symbol is already registered.
//
// Example usage:
//
//   __global__ void my_cuda_kernel(int* x);
//   KERNEL_SYMBOL_REGISTRY_REGISTER_SYMBOL_STATICALLY(my_unique_persistent_name,
//                                                     cuda::kCudaPlatformId,
//                                                     &my_cuda_kernel);
//
// The symbol will be registered with the name "my_unique_persistent_name" and
// the platform ID cuda::kCudaPlatformId. The name will also be used to generate
// a C++ identifier for the static initializer. therefore it needs to be a valid
// C++ variable name.
//
// Make sure to mark the target as `alwayslink = True` so that it won't be
// stripped by the linker.
#define KERNEL_SYMBOL_REGISTRY_REGISTER_SYMBOL_STATICALLY(identifier,          \
                                                          platform_id, symbol) \
  static void RegisterSymbol##identifier##Impl() {                             \
    absl::Status result =                                                      \
        stream_executor::KernelSymbolRegistry::GetGlobalInstance()             \
            .RegisterSymbol(#identifier, platform_id, symbol);                 \
    if (!result.ok()) {                                                        \
      LOG(FATAL) << "Failed to register symbol: " << result;                   \
    }                                                                          \
  }                                                                            \
  STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(                                 \
      RegisterSymbol##identifier, RegisterSymbol##identifier##Impl());

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_KERNEL_SYMBOL_REGISTRY_H_
