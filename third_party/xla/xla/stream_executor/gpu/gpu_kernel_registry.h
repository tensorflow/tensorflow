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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_REGISTRY_H_

#include <functional>
#include <tuple>
#include <typeindex>
#include <typeinfo>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"  // IWYU pragma: keep
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {

// This is a global registry for GPU kernels.
//
// The registry is keyed by a KernelTrait, which is a struct that contains the
// KernelType and the target platform.
//
// KernelTrait example:
//
//   struct MyKernelTrait {
//     using KernelType =
//         stream_executor::TypedKernel<stream_executor::DeviceMemoryBase,
//                                      size_t, size_t,
//                                      stream_executor::DeviceMemoryBase>;
//   };
//
// The registry is thread-safe. Registered kernels are immutable and cannot be
// overwritten.
//
// Use the macro GPU_KERNEL_REGISTRY_REGISTER_STATIC_KERNEL to register a
// kernel during application initialization.
class GpuKernelRegistry {
 public:
  // Loads a kernel from the registry into the given executor. The kernel is
  // identified by the KernelTrait. This function is thread-safe.
  template <typename KernelTrait>
  absl::StatusOr<typename KernelTrait::KernelType> LoadKernel(
      StreamExecutor* executor) {
    TF_ASSIGN_OR_RETURN(
        const MultiKernelLoaderSpec& spec,
        GetKernelSpec(typeid(KernelTrait), executor->GetPlatform()->id()));

    return KernelTrait::KernelType::FactoryType::Create(executor, spec);
  }

  // Looks up a kernel in the registry and returns a reference to the spec
  // object. Also have a look at `LoadKernel` which is a more convenient way to
  // load a kernel into a StreamExecutor instance. This function is
  // thread-safe.
  template <typename KernelTrait>
  absl::StatusOr<std::reference_wrapper<const MultiKernelLoaderSpec>>
  FindKernel(Platform::Id platform_id) {
    return GetKernelSpec(typeid(KernelTrait), platform_id);
  }

  // Registers a kernel `kernel` in the registry. This function is thread-safe.
  template <typename KernelTrait>
  absl::Status RegisterKernel(Platform::Id platform_id,
                              const MultiKernelLoaderSpec& kernel) {
    return RegisterKernel(typeid(KernelTrait), platform_id, kernel);
  }

  // Returns a reference to the process-wide instance of the registry.
  static GpuKernelRegistry& GetGlobalRegistry();

 private:
  absl::Status RegisterKernel(const std::type_info& type,
                              Platform::Id platform_id,
                              const MultiKernelLoaderSpec& kernel);

  absl::StatusOr<std::reference_wrapper<const MultiKernelLoaderSpec>>
  GetKernelSpec(const std::type_info& type, Platform::Id platform_id);

  absl::Mutex mutex_;
  using KernelRegistryKey = std::tuple<std::type_index, Platform::Id>;
  absl::flat_hash_map<KernelRegistryKey, MultiKernelLoaderSpec> kernel_specs_
      ABSL_GUARDED_BY(mutex_);
};

#define GPU_KERNEL_REGISTRY_REGISTER_KERNEL_STATICALLY(                \
    identifier, KernelTrait, platform_id, kernel)                      \
  static void RegisterKernel##identifier##Impl() {                     \
    absl::Status result =                                              \
        stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()   \
            .RegisterKernel<KernelTrait>(                              \
                platform_id,                                           \
                kernel(KernelTrait::KernelType::kNumberOfParameters)); \
    if (!result.ok()) {                                                \
      LOG(FATAL) << "Failed to register kernel: " << result;           \
    }                                                                  \
  }                                                                    \
  STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(                         \
      RegisterKernel##identifier, RegisterKernel##identifier##Impl());

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_KERNEL_REGISTRY_H_
