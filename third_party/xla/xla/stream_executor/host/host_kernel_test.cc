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

#include "xla/stream_executor/host/host_kernel.h"

#include <cstdint>
#include <vector>

#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace stream_executor::host {

static SE_HOST_KernelError* AddI32(const SE_HOST_KernelCallFrame* call_frame) {
  SE_HOST_KernelArg& lhs = call_frame->args[0];
  SE_HOST_KernelArg& rhs = call_frame->args[1];
  SE_HOST_KernelArg& out = call_frame->args[2];

  int32_t* lhs_ptr = reinterpret_cast<int32_t*>(lhs.data);
  int32_t* rhs_ptr = reinterpret_cast<int32_t*>(rhs.data);
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out.data);

  uint64_t x = call_frame->thread->x;
  *(out_ptr + x) = *(lhs_ptr + x) + *(rhs_ptr + x);

  return nullptr;
}

TEST(HostKernelTest, Addition) {
  HostKernel kernel(/*arity=*/3, AddI32);

  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(ThreadDim(4), args));

  std::vector<int32_t> expected = {6, 8, 10, 12};
  EXPECT_EQ(out, expected);
}

}  // namespace stream_executor::host
