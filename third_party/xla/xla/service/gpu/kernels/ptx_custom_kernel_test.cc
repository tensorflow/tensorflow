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

#include "xla/service/gpu/kernels/ptx_custom_kernel.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/stream_executor/cuda/cuda_platform.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu::kernel {

namespace se = ::stream_executor;

constexpr absl::string_view kAddI32KernelPtx = R"(
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

TEST(PtxCustomKernelTest, GetPtxCustomKernel) {
  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  se::gpu::CudaPlatform platform;
  TF_ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                          platform.ExecutorForDevice(0));
  TF_ASSERT_OK_AND_ASSIGN(
      CustomKernel custom_kernel,
      GetPtxCustomKernel("AddI32", kAddI32KernelPtx, 3, se::BlockDim(4),
                         se::ThreadDim(1), byte_length));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Kernel> kernel,
                          executor->LoadKernel(custom_kernel.kernel_spec()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                          executor->CreateStream());
  se::DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);
  TF_CHECK_OK(stream->Memset32(&a, 1, byte_length));
  TF_CHECK_OK(stream->Memset32(&b, 2, byte_length));
  TF_CHECK_OK(stream->MemZero(&c, byte_length));

  se::KernelArgsDeviceMemoryArray args(
      std::vector<se::DeviceMemoryBase>({a, b, c}),
      custom_kernel.shared_memory_bytes());
  TF_CHECK_OK(stream->Launch(custom_kernel.thread_dims(),
                             custom_kernel.block_dims(), *kernel, args));

  TF_CHECK_OK(stream->BlockHostUntilDone());

  std::vector<int32_t> dst(4, 42);
  TF_CHECK_OK(stream->Memcpy(dst.data(), c, byte_length));

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

}  // namespace xla::gpu::kernel
