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

#include "xla/backends/autotuner/autotuner.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status_matchers.h"

namespace xla {
namespace {

// Use one of existing gpu backend config protos as a test config.
using TestConfig = gpu::CustomFusionConfig;

class MockCodegenBackend : public CodegenBackend {
 public:
  MOCK_METHOD(absl::string_view, name, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>,
              GetSupportedConfigs,
              (const HloInstruction& instr,
               stream_executor::StreamExecutor* stream_executor),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<BackendConfig>>, GetDefaultConfig,
              (const HloInstruction& instr), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, Compile,
              (const HloInstruction& instr, const BackendConfig& config),
              (override));
  MOCK_METHOD(absl::Status, ApplyConfig,
              (HloInstruction & instr, const BackendConfig& config),
              (override));
};

class MockExecutor : public Executor {
 public:
  MOCK_METHOD(absl::StatusOr<ProfileResult>, Profile,
              (std::unique_ptr<Executable> executable), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<ProfileResult>>,
              ProfileWithSharedBuffers,
              (absl::Span<std::unique_ptr<Executable>> executables),
              (override));
};

using ::testing::_;
using ::testing::Return;
using tsl::testing::IsOk;
using tsl::testing::StatusIs;

TEST(AutotunerTest, NoCodegenBackend) {
  xla::Autotuner autotuner(nullptr, nullptr, xla::AutotuneConfig());
  EXPECT_THAT(autotuner.Autotune(nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AutotunerTest, AutotuneWithNoValidConfigs) {
  auto backend = std::make_unique<MockCodegenBackend>();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  TestConfig config;
  config.set_name("test_config");
  configs.push_back(std::make_unique<TestConfig>(config));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(absl::InternalError("test error")));
  xla::Autotuner autotuner(nullptr, std::make_unique<MockExecutor>(),
                           xla::AutotuneConfig());
  autotuner.RegisterCodegenBackend(std::move(backend));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner.Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(AutotunerTest, AutotuneAppliesBestConfig) {
  auto backend = std::make_unique<MockCodegenBackend>();
  std::vector<std::unique_ptr<BackendConfig>> configs;
  TestConfig config1, config2;
  config1.set_name("test_config_1");
  configs.push_back(std::make_unique<TestConfig>(config1));
  config2.set_name("test_config_2");
  configs.push_back(std::make_unique<TestConfig>(config2));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, testing::EqualsProto(config1))).Times(1);

  auto executor = std::make_unique<MockExecutor>();
  EXPECT_CALL(*executor.get(), Profile)
      .WillOnce(Return(ProfileResult{absl::Seconds(1)}))
      .WillOnce(Return(ProfileResult{absl::Seconds(2)}));

  xla::Autotuner autotuner(nullptr, std::move(executor), xla::AutotuneConfig());
  autotuner.RegisterCodegenBackend(std::move(backend));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner.Autotune(dummy_instr.get()), IsOk());
}

}  // namespace
}  // namespace xla
