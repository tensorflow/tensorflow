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
#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel_factory.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
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

static const char* llvm_kernel_add = R"(
%SE_HOST_KernelCallFrame = type { ptr, ptr, i64, ptr }
%struct.SE_HOST_KernelArg = type { ptr, i64 }

define ptr @LlvmAddI32(ptr noundef %0) {
  %2 = getelementptr inbounds %SE_HOST_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct.SE_HOST_KernelArg, ptr %3, i64 1
  %5 = getelementptr inbounds %struct.SE_HOST_KernelArg, ptr %3, i64 2
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = getelementptr inbounds %SE_HOST_KernelCallFrame, ptr %0, i32 0, i32 1
  %10 = load ptr, ptr %9, align 8
  %11 = load i64, ptr %10, align 8
  %12 = getelementptr inbounds i32, ptr %6, i64 %11
  %13 = load i32, ptr %12, align 4
  %14 = getelementptr inbounds i32, ptr %7, i64 %11
  %15 = load i32, ptr %14, align 4
  %16 = add nsw i32 %13, %15
  %17 = getelementptr inbounds i32, ptr %8, i64 %11
  store i32 %16, ptr %17, align 4
  ret ptr null
}
)";

static std::unique_ptr<StreamExecutor> NewStreamExecutor() {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  StreamExecutorConfig config(/*ordinal=*/0);
  return platform->GetUncachedExecutor(config).value();
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

TEST(HostKernelTest, LlvmAddition) {
  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddLlvmHostKernel(llvm_kernel_add, "LlvmAddI32", "LlvmAddI32",
                         absl::Span<std::string>());

  auto executor = NewStreamExecutor();
  auto eg = executor.get();
  EXPECT_NE(eg, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto add, KernelFactory::Create(eg, spec));

  // TODO(tsilytskyi): implement Launch part
  // TF_ASSERT_OK(executor->Launch(ThreadDim(4), args));

  // std::vector<int32_t> expected = {6, 8, 10, 12};
  // EXPECT_EQ(out, expected);
  // EXPECT_TRUE(true);
}

}  // namespace stream_executor::host
