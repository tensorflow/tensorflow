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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

// Use one of existing gpu backend config protos as a test config.
using TestConfig = gpu::CustomFusionConfig;

MATCHER_P(ConfigMatcher, name, "") {
  TestConfig test_config;
  arg.UnpackTo(&test_config);
  return test_config.name() == name;
}

MATCHER_P(InstructionMatcher, opcode, "") { return arg.opcode() == opcode; }

std::unique_ptr<google::protobuf::Any> GetTestConfig(std::string name) {
  TestConfig config;
  config.set_name(name);
  auto any = std::make_unique<google::protobuf::Any>();
  any->PackFrom(config);
  return any;
}

AutotuneConfig GetTestAutotuneConfig() {
  AutotuneConfig config;
  config.check_buffers = false;
  return config;
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
  MOCK_METHOD(bool, CanProduceWrongResults, (), (const, override));
};

class MockCodegenBackendWithWrongResults : public MockCodegenBackend {
 public:
  bool CanProduceWrongResults() const override { return true; }
};

class MockProfiler : public Profiler {
 public:
  MOCK_METHOD(absl::StatusOr<ProfileResult>, Profile,
              (Executable * executable, const InputBuffers& buffers),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<InputBuffers>>, CreateInputBuffers,
              (const Executable* executable), (override));
  MOCK_METHOD(absl::Status, CheckInputBuffers, (InputBuffers & buffers),
              (override));
  MOCK_METHOD(absl::Status, CheckOutputBuffer,
              (ScopedShapedBuffer & output, ScopedShapedBuffer& reference,
               float rtol),
              (override));
};

class MockAutotunerCache : public AutotunerCacheInterface {
 public:
  MOCK_METHOD(std::optional<AutotunerCacheInterface::Config>, Lookup,
              (const HloInstruction* instr), (override));
  MOCK_METHOD(absl::Status, Insert,
              (const HloInstruction* instr,
               AutotunerCacheInterface::Config& best_config),
              (override));
};

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::Return;
using tsl::proto_utils::ToDurationProto;

se::DeviceDescription CreateDummyDeviceDescription() {
  se::DeviceDescription desc;
  desc.set_name("test_device");
  return desc;
}

absl::StatusOr<std::unique_ptr<Autotuner>> SetupAutotunerWithExpectations(
    HloOpcode instr_to_autotune,
    std::pair<HloOpcode, int> instr_to_apply_config_and_count) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillRepeatedly(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _))
      .WillRepeatedly(Return(absl::OkStatus()));

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
      .Times(count)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  return Autotuner::Create(std::move(backends), std::move(profiler),
                           GetTestAutotuneConfig(), std::move(cache_manager));
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

class AutotunerTest : public HloHardwareIndependentTestBase {
 public:
  AutotunerTest() { config_ = GetTestAutotuneConfig(); }
  AutotuneConfig config_;
};

TEST_F(AutotunerTest, NoCodegenBackend) {
  auto device_description = CreateDummyDeviceDescription();
  auto autotuner = Autotuner::Create({}, nullptr, config_,
                                     std::make_unique<MockAutotunerCache>());
  EXPECT_THAT(autotuner, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(AutotunerTest, NoCacheManager) {
  auto device_description = CreateDummyDeviceDescription();
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<MockCodegenBackend>());
  auto autotuner =
      Autotuner::Create(std::move(backends), nullptr, config_, nullptr);
  EXPECT_THAT(autotuner, IsOk());
}

TEST_F(AutotunerTest, AutotuneButNoSupportedConfigs) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .Times(1)
      .WillOnce(Return(std::vector<std::unique_ptr<BackendConfig>>()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();

  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, AutotuneButNoCompiledConfigs) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("invalid_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(absl::InternalError("test error")));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();
  EXPECT_CALL(*profiler, Profile(_, _)).Times(0);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, AutotuneAppliesBestConfigAndSkipsNonCompilableConfig) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("non_compilable_config"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(absl::InternalError("test error")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), absl_testing::IsOk());
}

TEST_F(AutotunerTest, AutotuneAppliesBestConfigUsingThreadPool) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager), &thread_pool));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), absl_testing::IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleFindsNoInstructionsToAutotune) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs).Times(0);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  auto device_description = CreateDummyDeviceDescription();
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), nullptr, config_,
                        std::make_unique<MockAutotunerCache>()));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(autotuner->Autotune(
                  module.get(), [](const HloInstruction& _) { return false; }),
              absl_testing::IsOk());
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

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune),
              absl_testing::IsOk());
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

TEST_F(AutotunerTest, CacheHit) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  AutotunerCacheInterface::Config config;
  config.codegen_backend_name = "mock_backend";
  TestConfig test_config;
  GetTestConfig("test_config_2")->UnpackTo(&test_config);
  config.backend_config.PackFrom(test_config);

  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(config));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs).Times(0);
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1);
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));

  auto profiler = std::make_unique<MockProfiler>();

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneWithBufferCheck) {
  config_.check_buffers = true;

  std::vector<std::unique_ptr<BackendConfig>> configs_1;
  configs_1.push_back(GetTestConfig("test_config_1"));
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs_1)));
  EXPECT_CALL(*backend_1, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  std::vector<std::unique_ptr<BackendConfig>> configs_2;
  configs_2.push_back(GetTestConfig("wrong_results_config"));
  auto backend_2 = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*backend_2, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs_2)));
  EXPECT_CALL(*backend_2, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  EXPECT_CALL(*backend_1, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer output_1(Shape(), nullptr, 0),
      output_2(Shape(), nullptr, 0), output_3(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(output_1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(output_2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(output_3)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("Don't match")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend_1));
  backends.push_back(std::move(backend_2));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneWithScratchBytesOptimization) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("config_more_time_less_scratch"));
  configs.push_back(GetTestConfig("config_less_time_more_scratch"));
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend_1, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  EXPECT_CALL(*backend_1,
              ApplyConfig(_, ConfigMatcher("config_more_time_less_scratch")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(2),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/100,
      })))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(1),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/200,
      })));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend_1));
  config_.optimize_scratch_bytes = true;
  config_.scratch_bytes_window_size_us = 2;
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, ExpectAllInstructionsInCache) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).Times(0);

  config_.expect_all_instructions_in_cache = true;

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs).Times(0);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends), nullptr, config_,
                                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(AutotunerTest, DumpLogsToFile) {
  config_.dump_logs_to = tsl::io::JoinPath(tsl::testing::TmpDir(), "dump.log");

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2),
                                      /*output_buffer=*/std::nullopt,
                                      /*scratch_bytes=*/100})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  TF_ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());

  std::string content;
  EXPECT_THAT(tsl::ReadFileToString(tsl::Env::Default(), config_.dump_logs_to,
                                    &content),
              absl_testing::IsOk());
  AutotuningLogs actual_logs;
  EXPECT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(content, &actual_logs));

  AutotuningLogs expected_logs;
  AutotuningLog* log = expected_logs.add_logs();
  log->mutable_instr()->PackFrom(dummy_instr->ToProto());
  AutotuneResult* result_1 = log->add_results();
  *result_1->mutable_run_time() = ToDurationProto(absl::Seconds(2));
  result_1->set_scratch_bytes(100);
  AutotuneResult* result_2 = log->add_results();
  *result_2->mutable_run_time() = ToDurationProto(absl::Seconds(1));

  EXPECT_THAT(actual_logs, tsl::proto_testing::EqualsProto(expected_logs));
}

}  // namespace
}  // namespace xla
