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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_KERNELS_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_KERNELS_H_

#include <string_view>

namespace stream_executor::gpu {

// Collection of helper kernels required by StreamExecutor Gpu backend.

// PTX kernel compiled from:
//
//  __global__ void noop() {}
//
// Easiest way to get PTX from C++ is to use https://godbolt.org.
inline constexpr std::string_view kNoOpKernel = R"(
.version 4.0
.target sm_50
.address_size 64

.visible .entry noop()
{

        .loc    1 1 0

        .loc    1 4 1
        ret;

})";

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_KERNELS_H_
