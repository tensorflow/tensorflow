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
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

// Use one of existing gpu backend config protos as a test config.
using TestConfig = gpu::CustomFusionConfig;

MATCHER_P(ConfigMatcher, name, "") {
  const TestConfig& test_config = static_cast<const TestConfig&>(arg);
  return test_config.name() == name;
}

MATCHER_P(InstructionMatcher, opcode, "") { return arg.opcode() == opcode; }

std::unique_ptr<TestConfig> GetTestConfig(std::string name) {
  TestConfig config;
  config.set_name(name);
  return std::make_unique<TestConfig>(config);
}

class MockCodegenBackend : public CodegenBackend {
 public:
  MOCK_METHOD(absl::string_view, name, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>,
              GetSupportedConfigs, (const HloInstruction& instr), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<BackendConfig>>, GetDefaultConfig,
              (const HloInstruction& instr), (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, Compile,
              (const HloInstruction& instr, const BackendConfig& config),
              (override));
  MOCK_METHOD(absl::Status, ApplyConfig,
              (HloInstruction & instr, const BackendConfig& config),
              (override));
};

class MockProfiler : public Profiler {
 public:
  MOCK_METHOD(absl::StatusOr<std::vector<ProfileResult>>,
              ProfileWithSharedBuffers,
              (std::vector<std::unique_ptr<Executable>> executables),
              (override));
};

using ::testing::_;
using ::testing::Return;
using tsl::testing::IsOk;
using tsl::testing::StatusIs;

absl::StatusOr<std::unique_ptr<Autotuner>> SetupAutotunerWithExpectations(
    HloOpcode instr_to_autotune,
    std::pair<HloOpcode, int> instr_to_apply_config_and_count) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend,
              GetSupportedConfigs(InstructionMatcher(instr_to_autotune)))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  HloOpcode instr_to_apply_config = instr_to_apply_config_and_count.first;
  int count = instr_to_apply_config_and_count.second;
  EXPECT_CALL(*backend,
              ApplyConfig(InstructionMatcher(instr_to_apply_config), _))
      .Times(count);

  auto profiler = std::make_unique<MockProfiler>();
  std::vector<ProfileResult> profile_results = {{absl::Seconds(1)},
                                                {absl::Seconds(1)}};
  EXPECT_CALL(*profiler, ProfileWithSharedBuffers)
      .WillOnce(Return(profile_results));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  return Autotuner::Create(std::move(backends), std::move(profiler),
                           AutotuneConfig());
}

constexpr absl::string_view kHlo = R"(
  HloModule test_module
  
  ENTRY main {
    p0 = f32[] parameter(0)
    add = f32[] add(p0, p0)
    add_2 = f32[] add(p0, add)
    ROOT copy = f32[] copy(add_2)
  }
  )";

class AutotunerTest : public HloHardwareIndependentTestBase {};

TEST_F(AutotunerTest, NoCodegenBackend) {
  auto autotuner = Autotuner::Create({}, nullptr, AutotuneConfig());
  EXPECT_THAT(autotuner, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AutotunerTest, AutotuneWithNoValidConfigs) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(absl::InternalError("test error")));

  auto profiler = std::make_unique<MockProfiler>();
  std::vector<ProfileResult> profile_results;
  EXPECT_CALL(*profiler, ProfileWithSharedBuffers)
      .WillOnce(Return(profile_results));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), AutotuneConfig()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, AutotuneAppliesBestConfig) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .Times(1);

  auto profiler = std::make_unique<MockProfiler>();
  std::vector<ProfileResult> profile_results = {{absl::Seconds(1)},
                                                {absl::Seconds(2)}};
  EXPECT_CALL(*profiler, ProfileWithSharedBuffers)
      .WillOnce(Return(profile_results));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), AutotuneConfig()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleFindsNoInstructionsToAutotune) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs).Times(0);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), nullptr, AutotuneConfig()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(autotuner->Autotune(
                  module.get(), [](const HloInstruction& _) { return false; }),
              IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleFollowsFilter) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kCopy;
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instr_to_autotune=*/HloOpcode::kCopy,
          /*instr_to_apply_config_and_count=*/{HloOpcode::kCopy, 1}));

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune), IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleWithDuplicateInstructions) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));

  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kAdd;
  };
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instr_to_autotune=*/HloOpcode::kAdd,
          /*instr_to_apply_config_and_count=*/{HloOpcode::kAdd, 2}));

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune), IsOk());
}

}  // namespace
}  // namespace xla
