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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_TEST_KERNELS_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_TEST_KERNELS_H_

#include <string_view>

namespace stream_executor::cuda::internal {

// This is a collection of CUDA kernels compiled to PTX for use in various CUDA
// tests in this package.

// PTX kernel compiled from:
//
//  __global__ void add(int* a, int* b, int* c) {
//    int index = threadIdx.x + blockIdx.x * blockDim.x;
//    c[index] = a[index] + b[index];
//  }
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr std::string_view kAddI32Kernel = R"(
.version 8.0
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

}  // namespace stream_executor::cuda::internal

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_TEST_KERNELS_H_
