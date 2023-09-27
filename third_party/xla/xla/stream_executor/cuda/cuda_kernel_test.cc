/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/multi_platform_manager.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/test.h"

namespace stream_executor::cuda {

// PTX kernel compiled from:
//
//  __global__ void add(int* a, int* b, int* c) {
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    c[index] = a[index] + b[index];
//  }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
static std::string_view kAddI32Kernel = R"(
.version 8.2
.target sm_50
.address_size 64

.visible .entry add(
        .param .u64 add_param_0,
        .param .u64 add_param_1,
        .param .u64 add_param_2
)
{
        .reg .b32       %r<8>;
        .reg .b64       %rd<11>;
        .loc    1 1 0

        ld.param.u64    %rd1, [add_param_0];
        ld.param.u64    %rd2, [add_param_1];
        ld.param.u64    %rd3, [add_param_2];
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

TEST(CudaKernelTest, Add) {
  using AddI32Kernel = TypedKernel<DeviceMemory<int32_t>, DeviceMemory<int32_t>,
                                   DeviceMemory<int32_t>>;

  Platform* platform = MultiPlatformManager::PlatformWithName("CUDA").value();
  StreamExecutor* executor = platform->ExecutorForDevice(0).value();

  Stream stream(executor);
  stream.Init();
  ASSERT_TRUE(stream.ok());

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddCudaPtxInMemory(kAddI32Kernel, "add");

  AddI32Kernel add_kernel(executor);
  ASSERT_TRUE(executor->GetKernel(spec, &add_kernel).ok());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;

  // Prepare arguments: a=1, b=2, c=0
  DeviceMemory<int32_t> a = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> b = executor->AllocateArray<int32_t>(length, 0);
  DeviceMemory<int32_t> c = executor->AllocateArray<int32_t>(length, 0);

  stream.ThenMemset32(&a, 1, byte_length);
  stream.ThenMemset32(&b, 2, byte_length);
  stream.ThenMemZero(&c, byte_length);

  // Launch kernel.
  auto st = stream.ThenLaunch(ThreadDim(), BlockDim(4), add_kernel, a, b, c);
  ASSERT_TRUE(st.ok());

  // Copy data back to host.
  std::vector<int32_t> dst(4, 42);
  stream.ThenMemcpy(dst.data(), c, byte_length);

  std::vector<int32_t> expected = {3, 3, 3, 3};
  ASSERT_EQ(dst, expected);
}

}  // namespace stream_executor::cuda
