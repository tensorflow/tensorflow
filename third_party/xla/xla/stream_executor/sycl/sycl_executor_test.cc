/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_executor.h"

#include <fstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/debug_options_flags.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tests/llvm_irgen_test_base.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::sycl {
namespace {

using testing::IsEmpty;
using testing::Not;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

constexpr size_t kMemoryAllocationSize = 1024;

class SyclExecutorTest : public xla::LlvmIrGenTestBase {};

// TODO(intel-tf): Add unit tests for DeviceDescription once it is ready.
TEST_F(SyclExecutorTest, GetSyclKernel) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::sycl::kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(kDefaultDeviceOrdinal));

  std::string hlo_text = R"(
    ENTRY e {
      p0 = u32[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT res = u32[4] add(p0, p1)
    })";

  xla::HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsForTest());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::HloModule> hlo_module,
      xla::ParseAndReturnUnverifiedModule(hlo_text, config));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<xla::Executable> exec,
      CompileToExecutable(std::move(hlo_module),
                          /*run_optimization_passes=*/true));

  auto* gpu_exec = static_cast<xla::gpu::GpuExecutable*>(exec.get());
  ASSERT_NE(gpu_exec, nullptr);

  const xla::gpu::SequentialThunk& seq_thunk = gpu_exec->GetThunk();
  EXPECT_EQ(seq_thunk.thunks().size(), 1);

  const xla::gpu::Thunk* thunk = seq_thunk.thunks().at(0).get();
  ASSERT_NE(thunk, nullptr);
  EXPECT_EQ(thunk->kind(), xla::gpu::Thunk::Kind::kKernel);

  const auto* kernel_thunk = dynamic_cast<const xla::gpu::KernelThunk*>(thunk);
  ASSERT_NE(kernel_thunk, nullptr);

  std::string kernel_name = kernel_thunk->kernel_name();

  std::vector<uint8_t> spv_bin(gpu_exec->binary());

  KernelLoaderSpec spec =
      KernelLoaderSpec::CreateCudaCubinInMemorySpec(spv_bin, kernel_name, 3);

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          executor->LoadKernel(spec));

  auto sycl_executor = dynamic_cast<SyclExecutor*>(executor);
  ASSERT_NE(sycl_executor, nullptr);
  EXPECT_THAT(sycl_executor->GetSyclKernel(kernel.get()),
              IsOkAndHolds(kernel.get()));

  sycl_executor->UnloadKernel(kernel.get());
  EXPECT_THAT(sycl_executor->GetSyclKernel(kernel.get()),
              StatusIs(absl::StatusCode::kNotFound));

  EXPECT_THAT(sycl_executor->GetSyclKernel(nullptr),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(SyclExecutorTest, CreateUnifiedMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::sycl::kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemoryType::kUnified));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(kMemoryAllocationSize));
  EXPECT_NE(allocation->opaque(), nullptr);
  EXPECT_EQ(allocation->size(), kMemoryAllocationSize);
  allocation.reset();
}

TEST_F(SyclExecutorTest, CreateHostMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::sycl::kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocator> allocator,
                          executor->CreateMemoryAllocator(MemoryType::kHost));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(kMemoryAllocationSize));
  EXPECT_NE(allocation->opaque(), nullptr);
  EXPECT_EQ(allocation->size(), kMemoryAllocationSize);
  allocation.reset();
}

TEST_F(SyclExecutorTest, CreateCollectiveMemoryAllocatorWorks) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::sycl::kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MemoryAllocator> allocator,
      executor->CreateMemoryAllocator(MemoryType::kCollective));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MemoryAllocation> allocation,
                          allocator->Allocate(kMemoryAllocationSize));
  EXPECT_NE(allocation->opaque(), nullptr);
  EXPECT_EQ(allocation->size(), kMemoryAllocationSize);
  allocation.reset();
}

TEST_F(SyclExecutorTest, CreateUnsupportedMemoryAllocatorsFail) {
  TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                          stream_executor::PlatformManager::PlatformWithId(
                              stream_executor::sycl::kSyclPlatformId));
  TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                          platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  EXPECT_THAT(executor->CreateMemoryAllocator(MemoryType::kDevice),
              Not(IsOk()));
}

}  // namespace
}  // namespace stream_executor::sycl
