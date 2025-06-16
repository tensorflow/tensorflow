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

#include "xla/stream_executor/gpu/gpu_test_kernels.h"

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/gpu/gpu_test_kernel_traits.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor::gpu {
absl::StatusOr<internal::AddI32Kernel::KernelType> LoadAddI32TestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::AddI32Kernel>(executor);
}

absl::StatusOr<internal::MulI32Kernel::KernelType> LoadMulI32TestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::MulI32Kernel>(executor);
}

absl::StatusOr<internal::IncAndCmpKernel::KernelType> LoadCmpAndIncTestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::IncAndCmpKernel>(executor);
}

absl::StatusOr<internal::AddI32Ptrs3Kernel::KernelType>
LoadAddI32Ptrs3TestKernel(StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::AddI32Ptrs3Kernel>(executor);
}

absl::StatusOr<internal::CopyKernel::KernelType> LoadCopyTestKernel(
    StreamExecutor* executor) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .LoadKernel<internal::CopyKernel>(executor);
}

absl::StatusOr<MultiKernelLoaderSpec> GetAddI32TestKernelSpec(
    Platform::Id platform_id) {
  return GpuKernelRegistry::GetGlobalRegistry()
      .FindKernel<internal::AddI32Kernel>(platform_id);
}

MultiKernelLoaderSpec GetAddI32PtxKernelSpec() {
  // PTX kernel compiled from:
  //
  //  __global__ void add(int* a, int* b, int* c) {
  //    int index = threadIdx.x + blockIdx.x * blockDim.x;
  //    c[index] = a[index] + b[index];
  //  }
  //
  // Easiest way to get PTX from C++ is to use https://godbolt.org.
  static constexpr absl::string_view kAddI32KernelPtx = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry AddI32(
        .param .u64 AddI32_param_0,
        .param .u64 AddI32_param_1,
        .param .u64 AddI32_param_2
)
{
        .reg .b32       %r<8>;
        .reg .b64       %rd<11>;
        .loc    1 1 0

        ld.param.u64    %rd1, [AddI32_param_0];
        ld.param.u64    %rd2, [AddI32_param_1];
        ld.param.u64    %rd3, [AddI32_param_2];
        .loc    1 3 3
        cvta.to.global.u64      %rd4, %rd3;
        cvta.to.global.u64      %rd5, %rd2;
        cvta.to.global.u64      %rd6, %rd1;
        mov.u32         %r1, %tid.x;
        mov.u32         %r2, %ctaid.x;
        mov.u32         %r3, %ntid.x;
        mad.lo.s32      %r4, %r2, %r3, %r1;
        .loc    1 4 3
        mul.wide.s32    %rd7, %r4, 4;
        add.s64         %rd8, %rd6, %rd7;
        ld.global.u32   %r5, [%rd8];
        add.s64         %rd9, %rd5, %rd7;
        ld.global.u32   %r6, [%rd9];
        add.s32         %r7, %r6, %r5;
        add.s64         %rd10, %rd4, %rd7;
        st.global.u32   [%rd10], %r7;
        .loc    1 5 1
        ret;

})";

  return MultiKernelLoaderSpec::CreateCudaPtxInMemorySpec(kAddI32KernelPtx,
                                                          "AddI32", 3);
}
}  // namespace stream_executor::gpu
