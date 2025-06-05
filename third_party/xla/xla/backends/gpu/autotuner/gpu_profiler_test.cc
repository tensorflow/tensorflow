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

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/executable.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

using ::testing::ElementsAre;
using ::testing::Field;

class MockExecutable : public Executable {
 public:
  explicit MockExecutable(std::shared_ptr<HloModule> module, int duration_ns)
      : Executable(module), duration_ns_(duration_ns) {}
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
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
};

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
  executables.push_back(std::make_unique<MockExecutable>(module, 1000));
  executables.push_back(std::make_unique<MockExecutable>(module, 2000));

  auto profiler = GpuProfiler::Create(stream_exec_, ProfileOptions());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<ProfileResult> profiles,
      profiler->ProfileWithSharedBuffers(std::move(executables)));
  EXPECT_THAT(
      profiles,
      ElementsAre(Field(&ProfileResult::duration, absl::Nanoseconds(1000)),
                  Field(&ProfileResult::duration, absl::Nanoseconds(2000))));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
