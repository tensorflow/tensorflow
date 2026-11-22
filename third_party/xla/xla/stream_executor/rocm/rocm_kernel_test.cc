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
#include <memory>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_test_kernels.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::gpu {
namespace {
using testing::Ge;

TEST(RocmKernelTest, GetMaxOccupiedBlocksPerCore) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add_kernel,
                       GetAddI32TestKernelSpec(rocm::kROCmPlatformId));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> rocm_kernel,
                       executor->LoadKernel(add_kernel));

  EXPECT_EQ(rocm_kernel->Arity(), 3);
  EXPECT_THAT(rocm_kernel->GetMaxOccupiedBlocksPerCore(
                  ThreadDim(1, 1, 1), /*dynamic_shared_memory_bytes=*/0),
              absl_testing::IsOkAndHolds(Ge(1)));
}

TEST(RocmKernelTest, ArityZeroKernelSkipsArgumentCountCheck) {
  ASSERT_OK_AND_ASSIGN(Platform * platform,
                       PlatformManager::PlatformWithName("ROCM"));
  ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  // Load AddI32 without a typed signature, like FFI registered kernels.
  ASSERT_OK_AND_ASSIGN(KernelLoaderSpec add_i32,
                       GetAddI32TestKernelSpec(rocm::kROCmPlatformId));
  ASSERT_TRUE(add_i32.has_in_process_symbol());
  KernelLoaderSpec spec = KernelLoaderSpec::CreateInProcessSymbolSpec(
      add_i32.in_process_symbol()->symbol, add_i32.kernel_name(), /*arity=*/0);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                       executor->LoadKernel(spec));
  EXPECT_EQ(kernel->Arity(), 0);

  constexpr int32_t kLength = 4;
  constexpr int64_t kByteLength = sizeof(int32_t) * kLength;
  DeviceAddress<int32_t> a = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> b = executor->AllocateArray<int32_t>(kLength, 0);
  DeviceAddress<int32_t> c = executor->AllocateArray<int32_t>(kLength, 0);
  std::vector<int32_t> a_host = {1, 2, 3, 4};
  std::vector<int32_t> b_host = {10, 20, 30, 40};
  ASSERT_OK(stream->Memcpy(&a, a_host.data(), kByteLength));
  ASSERT_OK(stream->Memcpy(&b, b_host.data(), kByteLength));
  ASSERT_OK(stream->MemZero(&c, kByteLength));

  std::vector<DeviceAddressBase> args = {a, b, c};
  ASSERT_OK(kernel->Launch(ThreadDim(kLength, 1, 1), BlockDim(1, 1, 1),
                           /*cluster_dims=*/std::nullopt, stream.get(),
                           *PackKernelArgs(args, /*shmem_bytes=*/0)));

  std::vector<int32_t> dst(kLength, 0);
  ASSERT_OK(stream->Memcpy(dst.data(), c, kByteLength));
  EXPECT_EQ(dst, std::vector<int32_t>({11, 22, 33, 44}));
}

}  // namespace
}  // namespace stream_executor::gpu
