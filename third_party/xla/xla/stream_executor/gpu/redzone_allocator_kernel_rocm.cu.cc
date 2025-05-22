/* Copyright 2024 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "tsl/platform/statusor.h"

namespace {
__global__ void redzone_checker_kernel(uint8_t* input_buffer,
                                       uint8_t redzone_pattern,
                                       uint64_t buffer_length,
                                       uint32_t* out_mismatched_ptr) {
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= buffer_length) return;
  if (input_buffer[idx] != redzone_pattern) atomicAdd(out_mismatched_ptr, 1);
}
}  // namespace

namespace stream_executor {
template <typename... Args>
static absl::StatusOr<TypedKernel<Args...>*> LoadKernelOrGetPtr(
    StreamExecutor* executor, absl::string_view kernel_name, void* kernel_ptr) {
  using KernelPtrCacheKey = std::tuple<StreamExecutor*, std::string, void*>;

  static absl::Mutex kernel_ptr_cache_mutex(absl::kConstInit);
  static auto& kernel_ptr_cache ABSL_GUARDED_BY(kernel_ptr_cache_mutex) =
      *new std::map<KernelPtrCacheKey, TypedKernel<Args...>>;
  KernelPtrCacheKey kernel_ptr_cache_key{executor, kernel_name, kernel_ptr};
  absl::MutexLock lock(&kernel_ptr_cache_mutex);

  auto it = kernel_ptr_cache.find(kernel_ptr_cache_key);
  if (it == kernel_ptr_cache.end()) {
    TF_ASSIGN_OR_RETURN(TypedKernel<Args...> loaded,
                        (TypedKernelFactory<Args...>::Create(
                            executor, kernel_name, kernel_ptr)));
    it =
        kernel_ptr_cache.emplace(kernel_ptr_cache_key, std::move(loaded)).first;
  }

  CHECK(it != kernel_ptr_cache.end());
  return &it->second;
}
absl::StatusOr<ComparisonKernel*> GetComparisonKernel(
    StreamExecutor* executor, GpuAsmOpts /*gpu_asm_opts*/) {
  return LoadKernelOrGetPtr<DeviceMemory<uint8_t>, uint8_t, uint64_t,
                            DeviceMemory<uint64_t>>(
      executor, "redzone_checker",
      reinterpret_cast<void*>(redzone_checker_kernel));
}

}  // namespace stream_executor
