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

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/redzone_allocator_kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"

namespace stream_executor {
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
static const char* kRedzoneCheckerPtx = R"(
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

absl::StatusOr<ComparisonKernel> GetComparisonKernel(StreamExecutor* executor) {
  MultiKernelLoaderSpec spec(/*arity=*/4);
  spec.AddCudaPtxInMemory(kRedzoneCheckerPtx, "redzone_checker");

  return TypedKernelFactory<DeviceMemory<uint8_t>, uint8_t, uint64_t,
                            DeviceMemory<uint64_t>>::Create(executor, spec);
}
}  // namespace stream_executor
