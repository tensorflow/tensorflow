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

#include "xla/service/executable.h"

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_execution_profile.h"
#include "xla/service/service_executable_run_options.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

class TestExecutable : public Executable {
 public:
  explicit TestExecutable(std::shared_ptr<HloModule> module)
      : Executable{std::move(module)} {}

  explicit TestExecutable(HloModuleFactory hlo_module_factory)
      : Executable{std::move(hlo_module_factory)} {}

  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments) override {
    return absl::UnimplementedError("Not needed for this test.");
  }
};

class ExecutableTest : public HloHardwareIndependentTestBase {};

TEST_F(ExecutableTest, HloProtoGetterIsThreadCompatible) {
  // Executable::hlo_proto() is doing some lazy initialization of a
  // part of `hlo_proto_`. This test ensures that this is done in a
  // thread-compatible way.
  // Note that this test needs to run with --config=tsan to reliably
  // detect any potential data races.
  constexpr absl::string_view kHloModule = R"(
    HloModule module

    ENTRY main {
      ROOT c = s32[] constant(1)
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHloModule));

  TestExecutable executable(module);

  auto proto = std::make_unique<HloProto>();
  executable.set_hlo_proto(std::move(proto));

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test",
                                 /*num_threads=*/2);
    for (int i = 0; i < 2; ++i) {
      pool.Schedule([&] { executable.hlo_proto()->SerializeAsString(); });
    }
  }
}

constexpr absl::string_view kTrivialHloModule = R"(
  HloModule module

  ENTRY main {
    ROOT c = s32[] constant(1)
  }
)";

TEST_F(ExecutableTest, BuildsModuleWithFactoryOnDemand) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kTrivialHloModule));
  std::vector<std::string> reasons;
  TestExecutable executable([&](absl::string_view reason)
                                -> absl::StatusOr<std::shared_ptr<HloModule>> {
    reasons.push_back(std::string(reason));
    return module;
  });
  EXPECT_FALSE(executable.has_module());
  EXPECT_EQ(executable.shared_module(), nullptr);
  EXPECT_TRUE(reasons.empty());

  EXPECT_THAT(executable.GetOrCreateHloModule("first"), IsOkAndHolds(module));
  EXPECT_THAT(executable.GetOrCreateHloModule("second"), IsOkAndHolds(module));

  EXPECT_THAT(reasons, ElementsAre("first"));
  EXPECT_TRUE(executable.has_module());
  EXPECT_EQ(&executable.module(), module.get());
  EXPECT_EQ(executable.shared_module(), module);
  EXPECT_EQ(executable.name(), "module");
}

TEST_F(ExecutableTest, FactoryErrorIsSticky) {
  int num_calls = 0;
  TestExecutable executable(
      [&](absl::string_view) -> absl::StatusOr<std::shared_ptr<HloModule>> {
        ++num_calls;
        return absl::InternalError("broken");
      });

  EXPECT_THAT(executable.GetOrCreateHloModule("first"),
              StatusIs(absl::StatusCode::kInternal, "broken"));
  EXPECT_THAT(executable.GetOrCreateHloModule("second"),
              StatusIs(absl::StatusCode::kInternal, "broken"));
  EXPECT_EQ(num_calls, 1);
  EXPECT_FALSE(executable.has_module());
}

TEST_F(ExecutableTest, FactoryMustNotReturnNull) {
  TestExecutable executable(
      [](absl::string_view) -> absl::StatusOr<std::shared_ptr<HloModule>> {
        return nullptr;
      });

  EXPECT_THAT(executable.GetOrCreateHloModule("test"),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_FALSE(executable.has_module());
}

TEST_F(ExecutableTest, GetOrCreateHloModuleReturnsExistingModule) {
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kTrivialHloModule));
  TestExecutable executable(module);

  EXPECT_THAT(executable.GetOrCreateHloModule("test"), IsOkAndHolds(module));
}

TEST_F(ExecutableTest, GetOrCreateHloModuleFailsWithoutModuleAndFactory) {
  TestExecutable executable(std::shared_ptr<HloModule>(nullptr));

  EXPECT_THAT(executable.GetOrCreateHloModule("test"),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutableTest, ConcurrentCallersBuildModuleOnce) {
  // Readers either see no module or the fully built one. Note that this test
  // needs to run with --config=tsan to reliably detect data races.
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kTrivialHloModule));
  std::atomic<int> num_calls = 0;
  TestExecutable executable(
      [&](absl::string_view) -> absl::StatusOr<std::shared_ptr<HloModule>> {
        ++num_calls;
        return module;
      });

  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "test",
                                 /*num_threads=*/4);
    for (int i = 0; i < 3; ++i) {
      pool.Schedule([&] {
        for (int j = 0; j < 1000; ++j) {
          if (executable.has_module()) {
            EXPECT_EQ(executable.module().name(), "module");
          }
          std::shared_ptr<HloModule> shared_module = executable.shared_module();
          if (shared_module != nullptr) {
            EXPECT_EQ(shared_module->name(), "module");
          }
        }
      });
    }
    for (int i = 0; i < 4; ++i) {
      pool.Schedule([&] {
        EXPECT_THAT(executable.GetOrCreateHloModule("concurrent"),
                    IsOkAndHolds(module));
      });
    }
  }

  EXPECT_EQ(num_calls, 1);
  EXPECT_EQ(executable.shared_module(), module);
}

}  // namespace
}  // namespace xla
