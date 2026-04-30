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

#include "xla/stream_executor/sycl/sycl_timer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/tests/llvm_irgen_test_base.h"

namespace stream_executor::sycl {
namespace {

const int kDefaultDeviceOrdinal = 0;

using ::absl_testing::IsOk;
using ::testing::Gt;

class SyclTimerTest : public xla::LlvmIrGenTestBase {
 public:
  void LaunchSomeKernel(StreamExecutor* executor, Stream* stream) {
    using AddKernel =
        TypedKernelFactory<DeviceAddress<int32_t>, DeviceAddress<int32_t>,
                           DeviceAddress<int32_t>>;

    absl::string_view hlo_ir = R"(
    ENTRY e {
      p0 = u32[4] parameter(0)
      p1 = u32[4] parameter(1)
      ROOT res = u32[4] add(p0, p1)
    })";

    xla::HloModuleConfig config;
    config.set_debug_options(GetDebugOptionsForTest());
    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::HloModule> hlo_module,
        xla::ParseAndReturnUnverifiedModule(hlo_ir, config));

    TF_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<xla::Executable> exec,
        CompileToExecutable(std::move(hlo_module),
                            /*run_optimization_passes=*/true));

    auto* gpu_exec = static_cast<xla::gpu::GpuExecutable*>(exec.get());
    ASSERT_NE(gpu_exec, nullptr);

    const xla::gpu::ThunkExecutor& thunk_exec = gpu_exec->thunk_executor();
    EXPECT_EQ(thunk_exec.thunks().size(), 1);

    const xla::gpu::Thunk* thunk = thunk_exec.thunks().at(0).get();
    ASSERT_NE(thunk, nullptr);
    EXPECT_EQ(thunk->kind(), xla::gpu::Thunk::Kind::kKernel);

    const auto* kernel_thunk =
        dynamic_cast<const xla::gpu::KernelThunk*>(thunk);
    ASSERT_NE(kernel_thunk, nullptr);

    std::string kernel_name = kernel_thunk->kernel_name();

    std::vector<uint8_t> spirv_binary(gpu_exec->binary());

    KernelLoaderSpec spec = KernelLoaderSpec::CreateCudaCubinInMemorySpec(
        spirv_binary, kernel_name, 3);

    TF_ASSERT_OK_AND_ASSIGN(auto add, AddKernel::Create(executor, spec));

    constexpr int64_t kLength = 4;
    constexpr int64_t kByteLength = sizeof(int32_t) * kLength;

    // Prepare arguments: a=3, b=2, c=0
    DeviceAddress<int32_t> a = executor_->AllocateArray<int32_t>(kLength, 0);
    DeviceAddress<int32_t> b = executor_->AllocateArray<int32_t>(kLength, 0);
    DeviceAddress<int32_t> c = executor_->AllocateArray<int32_t>(kLength, 0);

    EXPECT_THAT(stream->Memset32(&a, 3, kByteLength), absl_testing::IsOk());
    EXPECT_THAT(stream->Memset32(&b, 2, kByteLength), absl_testing::IsOk());
    EXPECT_THAT(stream->MemZero(&c, kByteLength), absl_testing::IsOk());
    EXPECT_THAT(add.Launch(ThreadDim(kLength), BlockDim(), stream, a, b, c),
                absl_testing::IsOk());
  }

 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(
        Platform * platform,
        stream_executor::PlatformManager::PlatformWithId(kSyclPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(StreamExecutor * executor,
                            platform->ExecutorForDevice(kDefaultDeviceOrdinal));
    executor_ = static_cast<SyclExecutor*>(executor);
    TF_ASSERT_OK_AND_ASSIGN(stream_,
                            executor_->CreateStream(/*priority=*/std::nullopt));
  }

  SyclExecutor* executor_;
  std::unique_ptr<Stream> stream_;
};

TEST_F(SyclTimerTest, Create) {
  TF_ASSERT_OK_AND_ASSIGN(SyclTimer timer,
                          SyclTimer::Create(executor_, stream_.get()));

  // We don't really care what kernel we launch here as long as it takes a
  // non-zero amount of time.
  LaunchSomeKernel(executor_, stream_.get());

  TF_ASSERT_OK_AND_ASSIGN(absl::Duration timer_result,
                          timer.GetElapsedDuration());
  EXPECT_THAT(timer_result, Gt(absl::ZeroDuration()));
  EXPECT_THAT(timer.GetElapsedDuration(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
}  // namespace stream_executor::sycl
