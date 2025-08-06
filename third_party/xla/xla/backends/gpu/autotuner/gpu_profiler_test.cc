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

#include "xla/backends/gpu/autotuner/gpu_profiler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using absl_testing::IsOkAndHolds;
using absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Field;
using ::testing::Ne;

class MockExecutable : public Executable {
 public:
  explicit MockExecutable(std::shared_ptr<HloModule> module, int duration_ns,
                          bool should_fail = false)
      : Executable(module),
        duration_ns_(duration_ns),
        should_fail_(should_fail) {}
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    if (should_fail_) {
      return absl::InternalError("MockExecutable failed as requested.");
    }
    ExecutionProfile* profile = run_options->run_options().execution_profile();
    if (profile != nullptr) {
      profile->set_compute_time_ns(duration_ns_);
    }
    return ExecutionOutput(ShapeUtil::MakeTupleShape({}),
                           ShapeUtil::MakeTupleShape({}),
                           run_options->run_options().allocator(),
                           run_options->run_options().device_ordinal());
  }

 private:
  int duration_ns_;
  bool should_fail_;
};

absl::StatusOr<ScopedShapedBuffer> CreateTestBuffer(
    se::DeviceMemoryAllocator* allocator, se::StreamExecutor* stream_exec,
    se::Stream* stream, int32_t value) {
  Shape test_shape = ShapeUtil::MakeShape(S32, {});
  TF_ASSIGN_OR_RETURN(auto* transfer_manager, TransferManager::GetForPlatform(
                                                  stream_exec->GetPlatform()));
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer output,
      transfer_manager->AllocateScopedShapedBuffer(
          test_shape, allocator, stream_exec->device_ordinal()));
  Literal literal = LiteralUtil::CreateR0<int32_t>(value);
  TF_RETURN_IF_ERROR(
      transfer_manager->TransferLiteralToDevice(stream, literal, output));
  return output;
}

class GpuProfilerTest : public HloHardwareIndependentTestBase {
 public:
  GpuProfilerTest() {
    se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
    std::vector<se::StreamExecutor*> executors =
        PlatformUtil::GetStreamExecutors(platform).value();
    stream_exec_ = executors[0];
  }
  se::StreamExecutor* stream_exec_;
};

TEST_F(GpuProfilerTest, ProfileWithSharedBuffersWithoutOutputBuffer) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  std::vector<std::unique_ptr<Executable>> executables;
  executables.push_back(std::make_unique<MockExecutable>(module, 1000));
  executables.push_back(std::make_unique<MockExecutable>(module, 2000));

  ProfileOptions options;
  options.should_populate_output_buffer = false;
  auto profiler = GpuProfiler::Create(stream_exec_, options);
  TF_ASSERT_OK_AND_ASSIGN(auto profiles, profiler->ProfileWithSharedBuffers(
                                             std::move(executables)));
  EXPECT_EQ(profiles.size(), 2);
  TF_ASSERT_OK(profiles[0].status());
  TF_ASSERT_OK(profiles[1].status());
  EXPECT_THAT(profiles,
              ElementsAre(IsOkAndHolds(Field(&ProfileResult::duration,
                                             absl::Nanoseconds(1000))),
                          IsOkAndHolds(Field(&ProfileResult::duration,
                                             absl::Nanoseconds(2000)))));
  EXPECT_THAT(profiles,
              ElementsAre(IsOkAndHolds(Field(&ProfileResult::output_buffer,
                                             Eq(std::nullopt))),
                          IsOkAndHolds(Field(&ProfileResult::output_buffer,
                                             Eq(std::nullopt)))));
}

TEST_F(GpuProfilerTest, ProfileWithSharedBuffers) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  std::vector<std::unique_ptr<Executable>> executables;
  executables.push_back(std::make_unique<MockExecutable>(module, 1));

  auto profiler = GpuProfiler::Create(stream_exec_, ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(auto profiles, profiler->ProfileWithSharedBuffers(
                                             std::move(executables)));
  EXPECT_THAT(profiles, ElementsAre(IsOkAndHolds(Field(
                            &ProfileResult::output_buffer, Ne(std::nullopt)))));
}

TEST_F(GpuProfilerTest, FailingExecutablesReturnStatus) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  std::vector<std::unique_ptr<Executable>> executables;
  executables.push_back(std::make_unique<MockExecutable>(module, 1000));
  executables.push_back(
      std::make_unique<MockExecutable>(module, 2000, /*should_fail=*/true));
  executables.push_back(std::make_unique<MockExecutable>(module, 3000));

  auto profiler = GpuProfiler::Create(stream_exec_, ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(auto profiles, profiler->ProfileWithSharedBuffers(
                                             std::move(executables)));
  EXPECT_EQ(profiles.size(), 3);
  TF_ASSERT_OK(profiles[0].status());
  EXPECT_FALSE(profiles[1].ok());
  TF_ASSERT_OK(profiles[2].status());
  EXPECT_THAT(profiles[0], IsOkAndHolds(Field(&ProfileResult::duration,
                                              absl::Nanoseconds(1000))));
  EXPECT_THAT(profiles[2], IsOkAndHolds(Field(&ProfileResult::duration,
                                              absl::Nanoseconds(3000))));
}

TEST_F(GpuProfilerTest, CreateInputBuffersAndProfile) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);

  auto profiler = GpuProfiler::Create(stream_exec_, ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  TF_ASSERT_OK_AND_ASSIGN(ProfileResult profile,
                          profiler->Profile(&mock_executable, *buffers));
  EXPECT_EQ(profile.duration, absl::Nanoseconds(1000));
}

class GpuProfilerTestWithRedzonePadding
    : public GpuProfilerTest,
      public ::testing::WithParamInterface<int> {};

TEST_P(GpuProfilerTestWithRedzonePadding, CheckInputBuffers) {
  constexpr absl::string_view kHloModule = R"(
    HloModule module
    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));
  MockExecutable mock_executable(module, 1000);
  ProfileOptions options;
  options.redzone_padding_bytes = GetParam();
  auto profiler = GpuProfiler::Create(stream_exec_, options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<InputBuffers> buffers,
                          profiler->CreateInputBuffers(&mock_executable));
  TF_EXPECT_OK(profiler->CheckInputBuffers(*buffers));
}

INSTANTIATE_TEST_SUITE_P(GpuProfilerTestWithRedzonePadding,
                         GpuProfilerTestWithRedzonePadding,
                         ::testing::Values(0, 1024));

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreSame) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options);

  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kOk));
}

TEST_F(GpuProfilerTest, CheckOutputBufferWhenBuffersAreDifferent) {
  ProfileOptions options;
  auto profiler = GpuProfiler::Create(stream_exec_, options);
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_exec_->CreateStream());
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
          stream_exec_);
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer output,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/1));
  TF_ASSERT_OK_AND_ASSIGN(ScopedShapedBuffer reference,
                          CreateTestBuffer(allocator.get(), stream_exec_,
                                           stream.get(), /*value=*/2));
  EXPECT_THAT(profiler->CheckOutputBuffer(output, reference, /*rtol=*/0.0),
              StatusIs(absl::StatusCode::kInternal));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
