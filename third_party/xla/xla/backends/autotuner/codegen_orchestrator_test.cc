/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/autotuner/codegen_orchestrator.h"

#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/mock_codegen_backend.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/hlo.pb.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::ByMove;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::SizeIs;

std::unique_ptr<Executable> RegisterSpillingExecutable(int spilled = 8) {
  gpu::GpuExecutable::Params params;
  params.executable =
      std::make_unique<gpu::ThunkExecutor>(gpu::ThunkSequence{});
  params.buffer_assignment_proto = BufferAssignmentProto();
  params.buffer_allocations_debug_summary = "dummy";
  KernelStats kernel_stats;
  kernel_stats.store_bytes_spilled = spilled;
  kernel_stats.load_bytes_spilled = spilled;
  params.module_stats = {{"test_config_2", kernel_stats}};
  return gpu::GpuExecutable::Create(std::move(params)).value();
}

class CodegenOrchestratorTest : public HloHardwareIndependentTestBase {};

TEST_F(CodegenOrchestratorTest, NoCodegenBackendReturnsInvalidArgument) {
  auto orchestrator = CodegenOrchestrator::Create({}, {});
  EXPECT_THAT(orchestrator, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(CodegenOrchestratorTest, ReordersBackendsSoCorrectOnesAreFirst) {
  auto wrong_backend_1 = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*wrong_backend_1, name())
      .WillRepeatedly(Return("wrong_backend_1"));

  auto correct_backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*correct_backend_1, CanProduceWrongResults())
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*correct_backend_1, name())
      .WillRepeatedly(Return("correct_backend_1"));

  auto wrong_backend_2 = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*wrong_backend_2, name())
      .WillRepeatedly(Return("wrong_backend_2"));

  auto correct_backend_2 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*correct_backend_2, CanProduceWrongResults())
      .WillRepeatedly(Return(false));
  EXPECT_CALL(*correct_backend_2, name())
      .WillRepeatedly(Return("correct_backend_2"));

  auto* c1_ptr = correct_backend_1.get();
  auto* c2_ptr = correct_backend_2.get();
  auto* w1_ptr = wrong_backend_1.get();
  auto* w2_ptr = wrong_backend_2.get();

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(wrong_backend_1));
  backends.push_back(std::move(correct_backend_1));
  backends.push_back(std::move(wrong_backend_2));
  backends.push_back(std::move(correct_backend_2));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  ASSERT_THAT(orchestrator->codegen_backends(), SizeIs(4));
  EXPECT_EQ(orchestrator->codegen_backends()[0].get(), c1_ptr);
  EXPECT_EQ(orchestrator->codegen_backends()[1].get(), c2_ptr);
  EXPECT_EQ(orchestrator->codegen_backends()[2].get(), w1_ptr);
  EXPECT_EQ(orchestrator->codegen_backends()[3].get(), w2_ptr);
}

TEST_F(CodegenOrchestratorTest, GetSupportedConfigsAggregatesFromAllBackends) {
  std::vector<std::unique_ptr<BackendConfig>> configs_1;
  configs_1.push_back(GetTestConfig("test_config_1"));
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs_1)));
  EXPECT_CALL(*backend_1, name()).WillRepeatedly(Return("backend_1"));

  std::vector<std::unique_ptr<BackendConfig>> configs_2;
  configs_2.push_back(GetTestConfig("test_config_2"));
  auto backend_2 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_2, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs_2)));
  EXPECT_CALL(*backend_2, name()).WillRepeatedly(Return("backend_2"));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend_1));
  backends.push_back(std::move(backend_2));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  ASSERT_OK_AND_ASSIGN(auto supported,
                       orchestrator->GetSupportedConfigs(*dummy_instr));
  ASSERT_THAT(supported, SizeIs(2));
  EXPECT_THAT(*supported[0].backend_config, ConfigMatcher("test_config_1"));
  EXPECT_THAT(*supported[1].backend_config, ConfigMatcher("test_config_2"));
}

TEST_F(CodegenOrchestratorTest,
       GetSupportedConfigsToleratesPartialBackendFailure) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));

  auto good_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*good_backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*good_backend, name()).WillRepeatedly(Return("good_backend"));

  auto bad_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend, GetSupportedConfigs(_))
      .WillOnce(Return(absl::InternalError("backend error")));
  EXPECT_CALL(*bad_backend, name()).WillRepeatedly(Return("bad_backend"));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(good_backend));
  backends.push_back(std::move(bad_backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  ASSERT_OK_AND_ASSIGN(auto supported,
                       orchestrator->GetSupportedConfigs(*dummy_instr));
  ASSERT_THAT(supported, SizeIs(1));
  EXPECT_THAT(*supported[0].backend_config, ConfigMatcher("test_config_1"));
}

TEST_F(CodegenOrchestratorTest, GetSupportedConfigsFailsWhenAllBackendsFail) {
  auto bad_backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend_1, GetSupportedConfigs(_))
      .WillOnce(Return(absl::InternalError("error 1")));
  EXPECT_CALL(*bad_backend_1, name()).WillRepeatedly(Return("bad_backend_1"));

  auto bad_backend_2 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend_2, GetSupportedConfigs(_))
      .WillOnce(Return(absl::InternalError("error 2")));
  EXPECT_CALL(*bad_backend_2, name()).WillRepeatedly(Return("bad_backend_2"));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(bad_backend_1));
  backends.push_back(std::move(bad_backend_2));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(orchestrator->GetSupportedConfigs(*dummy_instr),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(CodegenOrchestratorTest, GetDefaultConfigReturnsFirstAvailable) {
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetDefaultConfig(_))
      .WillOnce(Return(absl::NotFoundError("no default")));
  EXPECT_CALL(*backend_1, name()).WillRepeatedly(Return("backend_1"));

  auto backend_2 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_2, GetDefaultConfig(_))
      .WillOnce(Return(ByMove(GetTestConfig("default"))));
  EXPECT_CALL(*backend_2, name()).WillRepeatedly(Return("backend_2"));

  auto* backend_2_ptr = backend_2.get();
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend_1));
  backends.push_back(std::move(backend_2));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  ASSERT_OK_AND_ASSIGN(auto default_config,
                       orchestrator->GetDefaultConfig(*dummy_instr));
  EXPECT_THAT(*default_config.backend_config, ConfigMatcher("default"));
  EXPECT_EQ(default_config.codegen_backend, backend_2_ptr);
}

TEST_F(CodegenOrchestratorTest, GetDefaultConfigFailsWhenNoBackendProvides) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetDefaultConfig(_))
      .WillOnce(Return(absl::NotFoundError("no default")));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(orchestrator->GetDefaultConfig(*dummy_instr),
              StatusIs(absl::StatusCode::kNotFound));
}


TEST_F(CodegenOrchestratorTest, ConfigsWithRegisterSpillingAreAllowed) {
  CodegenOrchestrator::Options options;
  options.allow_reg_spills_fn = [](const HloInstruction&, autotuner::Backend) {
    return true;
  };

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::UNSPECIFIED_BACKEND));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(RegisterSpillingExecutable()));

  CodegenOrchestrator::Config config{backend.get(),
                                     GetTestConfig("test_config_1")};

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator, CodegenOrchestrator::Create(
                                              std::move(backends), options));

  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(orchestrator->Compile(*dummy_instr, config), IsOk());
}

TEST_F(CodegenOrchestratorTest, ConfigsWithRegisterSpillingAreFiltered) {
  CodegenOrchestrator::Options options;
  options.allow_reg_spills_fn = [](const HloInstruction&, autotuner::Backend) {
    return false;
  };

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::UNSPECIFIED_BACKEND));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(RegisterSpillingExecutable()));

  CodegenOrchestrator::Config config{backend.get(),
                                     GetTestConfig("test_config_1")};

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator, CodegenOrchestrator::Create(
                                              std::move(backends), options));

  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(orchestrator->Compile(*dummy_instr, config),
              StatusIs(absl::StatusCode::kResourceExhausted));
}

TEST_F(CodegenOrchestratorTest, CompileAllExecutesAcrossBackends) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::UNSPECIFIED_BACKEND));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(absl::InternalError("compile error")));

  std::vector<CodegenOrchestrator::Config> configs;
  configs.push_back({backend.get(), GetTestConfig("test_config_1")});
  configs.push_back({backend.get(), GetTestConfig("test_config_2")});

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));

  auto candidates_future =
      orchestrator->CompileAll(*dummy_instr, std::move(configs), &thread_pool);
  ASSERT_OK_AND_ASSIGN(auto candidates, std::move(candidates_future).Await());
  ASSERT_THAT(candidates, SizeIs(2));
  EXPECT_THAT(candidates[0].executable, IsOk());
  EXPECT_THAT(candidates[1].executable,
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Compilation failed: compile error")));
}

TEST_F(CodegenOrchestratorTest, ApplyConfigDelegatesToBackend) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::OkStatus()));

  CodegenOrchestrator::Config config{backend.get(),
                                     GetTestConfig("test_config_1")};

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(orchestrator->ApplyConfig(*dummy_instr, config), IsOk());
}

}  // namespace
}  // namespace xla
