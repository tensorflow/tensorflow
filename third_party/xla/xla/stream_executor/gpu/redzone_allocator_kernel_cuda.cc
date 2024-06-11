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
#include <tuple>
#include <utility>

#include "absl/base/call_once.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/cuda/cuda_asm_compiler.h"
#include "xla/stream_executor/cuda/cuda_driver.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator_kernel.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
// Maintains a cache of pointers to loaded kernels
template <typename... Args>
static absl::StatusOr<TypedKernel<Args...>*> LoadKernelOrGetPtr(
    StreamExecutor* executor, absl::string_view kernel_name,
    absl::string_view ptx, absl::Span<const uint8_t> cubin_data) {
  using KernelPtrCacheKey =
      std::tuple<CUcontext, absl::string_view, absl::string_view>;

  static absl::Mutex kernel_ptr_cache_mutex(absl::kConstInit);
  static auto& kernel_ptr_cache ABSL_GUARDED_BY(kernel_ptr_cache_mutex) =
      *new absl::node_hash_map<KernelPtrCacheKey, TypedKernel<Args...>>();
  CUcontext current_context = cuda::CurrentContextOrDie();
  KernelPtrCacheKey kernel_ptr_cache_key{current_context, kernel_name, ptx};
  absl::MutexLock lock(&kernel_ptr_cache_mutex);

  auto it = kernel_ptr_cache.find(kernel_ptr_cache_key);
  if (it == kernel_ptr_cache.end()) {
    TF_ASSIGN_OR_RETURN(TypedKernel<Args...> loaded,
                        (TypedKernelFactory<Args...>::Create(
                            executor, kernel_name, ptx, cubin_data)));
    it =
        kernel_ptr_cache.emplace(kernel_ptr_cache_key, std::move(loaded)).first;
  }

  CHECK(it != kernel_ptr_cache.end());
  return &it->second;
}

// PTX blob for the function which checks that every byte in
// input_buffer (length is buffer_length) is equal to redzone_pattern.
//
// On mismatch, increment the counter pointed to by out_mismatch_cnt_ptr.
//
// Generated from:
// __global__ void redzone_checker(unsigned char* input_buffer,
//                                 unsigned char redzone_pattern,
//                                 unsigned long long buffer_length,
//                                 int* out_mismatched_ptr) {
//   unsigned long long idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   if (input_buffer[idx] != redzone_pattern) atomicAdd(out_mismatched_ptr, 1);
// }
//
// Code must compile for the oldest GPU XLA may be compiled for.
static const char* redzone_checker_ptx = R"(
.version 4.2
.target sm_30
.address_size 64

.visible .entry redzone_checker(
  .param .u64 input_buffer,
  .param .u8 redzone_pattern,
  .param .u64 buffer_length,
  .param .u64 out_mismatch_cnt_ptr
)
{
  .reg .pred   %p<3>;
  .reg .b16   %rs<3>;
  .reg .b32   %r<6>;
  .reg .b64   %rd<8>;

  ld.param.u64   %rd6, [buffer_length];
  mov.u32   %r1, %tid.x;
  mov.u32   %r2, %ctaid.x;
  mov.u32   %r3, %ntid.x;
  mad.lo.s32   %r4, %r3, %r2, %r1;
  cvt.u64.u32   %rd3, %r4;
  setp.ge.u64   %p1, %rd3, %rd6;
  @%p1 bra   LBB6_3;
  ld.param.u8   %rs1, [redzone_pattern];
  ld.param.u64   %rd4, [input_buffer];
  cvta.to.global.u64   %rd2, %rd4;
  add.s64   %rd7, %rd2, %rd3;
  ld.global.u8   %rs2, [%rd7];
  setp.eq.s16   %p2, %rs2, %rs1;
  @%p2 bra   LBB6_3;
  ld.param.u64   %rd5, [out_mismatch_cnt_ptr];
  cvta.to.global.u64   %rd1, %rd5;
  atom.global.add.u32   %r5, [%rd1], 1;
LBB6_3:
  ret;
}
)";

absl::StatusOr<const ComparisonKernel*> GetComparisonKernel(
    StreamExecutor* executor, GpuAsmOpts gpu_asm_opts) {
  absl::Span<const uint8_t> compiled_ptx = {};
  absl::StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
      CompileGpuAsmOrGetCached(executor->device_ordinal(), redzone_checker_ptx,
                               gpu_asm_opts);
  if (compiled_ptx_or.ok()) {
    compiled_ptx = compiled_ptx_or.value();
  } else {
    static absl::once_flag ptxas_not_found_logged;
    absl::call_once(ptxas_not_found_logged, [&]() {
      LOG(WARNING) << compiled_ptx_or.status()
                   << "\nRelying on driver to perform ptx compilation. "
                   << "\nModify $PATH to customize ptxas location."
                   << "\nThis message will be only logged once.";
    });
  }

  return LoadKernelOrGetPtr<DeviceMemory<uint8_t>, uint8_t, uint64_t,
                            DeviceMemory<uint64_t>>(
      executor, "redzone_checker", redzone_checker_ptx, compiled_ptx);
}
}  // namespace stream_executor
