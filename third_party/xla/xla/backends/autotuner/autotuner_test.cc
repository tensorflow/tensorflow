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
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/testing/temporary_directory.h"
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
MATCHER_P(InstrPtrMatcher, opcode, "") { return arg->opcode() == opcode; }

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
               const AutotunerCacheInterface::Config& best_config),
              (override));
  MOCK_METHOD(absl::StatusOr<std::string>, Serialize,
              (absl::Span<const HloInstruction* const> instructions),
              (override));
  MOCK_METHOD(absl::Status, Deserialize, (absl::string_view serialized_cache),
              (override));
};

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::AtMost;
using ::testing::ByMove;
using ::testing::MatchesRegex;
using ::testing::Return;
using ::testing::UnorderedElementsAre;
using tsl::proto_utils::ToDurationProto;

se::DeviceDescription CreateDummyDeviceDescription() {
  se::DeviceDescription desc;
  desc.set_name("test_device");
  return desc;
}

absl::StatusOr<std::unique_ptr<Autotuner>> SetupAutotunerWithExpectations(
    std::vector<HloOpcode> instrs_to_autotune,
    std::vector<std::pair<HloOpcode, int>> instrs_to_apply_config_and_count,
    std::unique_ptr<MockAutotunerCache> cache = nullptr,
    bool dump_hlos = false) {
  auto backend = std::make_unique<MockCodegenBackend>();
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  for (const auto& instr_to_autotune : instrs_to_autotune) {
    std::vector<std::unique_ptr<BackendConfig>> configs;
    // Best config is just by notion here since profiler time is same for all.
    configs.push_back(GetTestConfig("best_config"));
    configs.push_back(GetTestConfig("another_config"));
    EXPECT_CALL(*backend,
                GetSupportedConfigs(InstructionMatcher(instr_to_autotune)))
        .WillOnce(Return(std::move(configs)));
  }
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .Times(instrs_to_autotune.size())
      .WillRepeatedly([] { return std::make_unique<InputBuffers>(); });
  EXPECT_CALL(*backend, Compile(_, _))
      .Times(2 * instrs_to_autotune.size())
      .WillRepeatedly([] { return std::unique_ptr<Executable>(); });
  EXPECT_CALL(*profiler, Profile(_, _))
      .Times(2 * instrs_to_autotune.size())
      .WillRepeatedly([] { return ProfileResult({absl::Seconds(1)}); });

  for (const auto& [instr_to_apply_config, count] :
       instrs_to_apply_config_and_count) {
    EXPECT_CALL(*backend,
                ApplyConfig(InstructionMatcher(instr_to_apply_config), _))
        .Times(count)
        .WillRepeatedly(Return(absl::OkStatus()));
  }
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  AutotuneConfig config = GetTestAutotuneConfig();
  config.dump_hlos = dump_hlos;
  return Autotuner::Create(std::move(backends), std::move(profiler), config,
                           std::move(cache));
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

  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
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
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), nullptr, config_,
                        std::make_unique<MockAutotunerCache>()));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  EXPECT_THAT(autotuner->Autotune(
                  module.get(), [](const HloInstruction& _) { return false; }),
              absl_testing::IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleFollowsFilter) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kCopy;
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kCopy},
          /*instrs_to_apply_config_and_count=*/{{HloOpcode::kCopy, 1}}));

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune),
              absl_testing::IsOk());
}

TEST_F(AutotunerTest, AutotuneModuleWithDuplicateInstructions) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kAdd;
  };
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
          /*instrs_to_apply_config_and_count=*/{{HloOpcode::kAdd, 2}}));

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune), IsOk());
}

TEST_F(AutotunerTest, AutotuneButOneBackendFails) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config"));

  auto good_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*good_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*good_backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*good_backend, ApplyConfig(_, ConfigMatcher("test_config")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  auto bad_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend, GetSupportedConfigs)
      .WillOnce(Return(absl::InternalError("test error")));

  auto profiler = std::make_unique<MockProfiler>();
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(good_backend));
  backends.push_back(std::move(bad_backend));
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), absl_testing::IsOk());
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
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneWithBufferCheckFiltersWrongResults) {
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
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneSkipsBufferCheckWhenNoReferenceOutput) {
  config_.check_buffers = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));
  auto backend = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer output_1(Shape(), nullptr, 0),
      output_2(Shape(), nullptr, 0), output_3(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(output_1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::nullopt})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _)).Times(0);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneWithScratchBytesOptimization) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("config_most_time_less_scratch"));
  configs.push_back(GetTestConfig("config_less_time_less_scratch"));
  configs.push_back(GetTestConfig("config_least_time_most_scratch"));
  configs.push_back(GetTestConfig("config_more_time_less_scratch"));
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend_1, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  EXPECT_CALL(*backend_1,
              ApplyConfig(_, ConfigMatcher("config_less_time_less_scratch")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(7),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/100,
      })))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(3),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/100,
      })))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(2),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/200,
      })))
      .WillOnce(Return(ProfileResult({
          /*duration=*/absl::Microseconds(6),
          /*output_buffer=*/std::nullopt,
          /*scratch_bytes=*/100,
      })));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend_1));
  config_.optimize_scratch_bytes = true;
  config_.scratch_bytes_window_size_us = 8;
  ASSERT_OK_AND_ASSIGN(
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

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(backends), nullptr, config_,
                                         std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(AutotunerTest, DumpLogsToFile) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  config_.dump_logs_to = tsl::io::JoinPath(temp_dir.path(), "dump.log");

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
  ASSERT_OK_AND_ASSIGN(
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
  result_1->mutable_other()->set_name("mock_backend");
  *result_1->mutable_other()->mutable_config() =
      *GetTestConfig("test_config_1");
  *result_1->mutable_run_time() = ToDurationProto(absl::Seconds(2));
  result_1->set_scratch_bytes(100);
  AutotuneResult* result_2 = log->add_results();
  result_2->mutable_other()->set_name("mock_backend");
  *result_2->mutable_other()->mutable_config() =
      *GetTestConfig("test_config_2");
  *result_2->mutable_run_time() = ToDurationProto(absl::Seconds(1));

  EXPECT_THAT(actual_logs, tsl::proto_testing::EqualsProto(expected_logs));
}

TEST_F(AutotunerTest, ExcludeCublasConfig) {
  config_.exclude_cublas_config = true;
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("Cublas_fission"));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();
  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, SelectFirstConfig) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

std::unique_ptr<Executable> RegisterSpillingExecutable() {
  gpu::GpuExecutable::Params params;
  params.executable = std::make_unique<gpu::SequentialThunk>(
      gpu::Thunk::ThunkInfo{}, gpu::ThunkSequence{});
  KernelStats kernel_stats;
  kernel_stats.store_bytes_spilled = 8;
  kernel_stats.load_bytes_spilled = 8;
  params.module_stats = {{"test_config_2", kernel_stats}};
  return gpu::GpuExecutable::Create(std::move(params)).value();
}

TEST_F(AutotunerTest, ConfigsWithRegisterSpillingAreAllowed) {
  config_.allow_reg_spills = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(RegisterSpillingExecutable()));
  EXPECT_CALL(*backend, ApplyConfig(_, _))
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  // Expect both configs to be profiled, as we allow register spilling.
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

TEST_F(AutotunerTest, ConfigsWithRegisterSpillingAreFiltered) {
  config_.allow_reg_spills = false;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));
  configs.push_back(GetTestConfig("test_config_3"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(RegisterSpillingExecutable()));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_3")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, _))
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  // Out of 3 configs, expect only 2 to be profiled as one spilled registers.
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

TEST_F(AutotunerTest, SelectFirstConfigStopsAfterFirstSuccess) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));
  configs.push_back(GetTestConfig("test_config_3"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2"))).Times(0);
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_3"))).Times(0);

  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

TEST_F(AutotunerTest, SelectFirstConfigFirstConfigFails) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::InternalError("test error")));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

TEST_F(AutotunerTest, SelectFirstConfigAllConfigsFail) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::InternalError("test error")));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(absl::InternalError("test error")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr),
              StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, UseDefaultConfig) {
  config_.use_default_config = true;

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_)).Times(0);
  EXPECT_CALL(*backend, GetDefaultConfig(_))
      .WillOnce(Return(ByMove(GetTestConfig("default"))));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("default")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), /*profiler=*/nullptr, config_,
                        /*cache=*/nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
}

TEST_F(AutotunerTest, UseDefaultConfigUnimplemented) {
  config_.use_default_config = true;

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs(_)).Times(0);
  EXPECT_CALL(*backend, GetDefaultConfig(_))
      .Times(AtMost(1))
      .WillRepeatedly(
          [] { return absl::UnimplementedError("not implemented"); });
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), /*profiler=*/nullptr, config_,
                        /*cache=*/nullptr));
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_DEATH(autotuner->Autotune(dummy_instr).IgnoreError(),
               "GetDefaultConfig is not implemented for mock_backend");
}

class MockKeyValueStore : public KeyValueStoreInterface {
 public:
  MOCK_METHOD(absl::Status, Set,
              (absl::string_view key, absl::string_view value), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, Get,
              (absl::string_view key, absl::Duration timeout), (override));
  MOCK_METHOD(absl::StatusOr<std::string>, TryGet, (absl::string_view key),
              (override));
  MOCK_METHOD(std::shared_ptr<tsl::CallOptions>, AsyncGet,
              (absl::string_view key,
               tsl::CoordinationServiceAgent::StatusOrValueCallback done),
              (override));
};

AutotunerCacheInterface::Config GetCacheConfig(absl::string_view name) {
  AutotunerCacheInterface::Config config;
  config.codegen_backend_name = "mock_backend";
  config.backend_config = *GetTestConfig(std::string(name));
  return config;
};

TEST_F(AutotunerTest, ShardedAutotuning) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  constexpr int kShardCount = 2;
  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kAdd ||
           instruction.opcode() == HloOpcode::kCopy;
  };
  auto kv_store = std::make_shared<MockKeyValueStore>();
  auto cache = std::make_unique<MockAutotunerCache>();

  // Shard 0 autotunes kCopy instructions, updates the cache and serializes the
  // result to a string "kCopy_autotune_result".
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kCopy)))
      .WillOnce(Return(std::nullopt))                    // During autotuning.
      .WillOnce(Return(GetCacheConfig("best_config")));  // Config application.
  EXPECT_CALL(*cache, Insert(InstrPtrMatcher(HloOpcode::kCopy), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Serialize(_)).WillOnce(Return("kCopy_autotune_result"));
  // Stores the serialized results to the KV store if it does not exist.
  EXPECT_CALL(*kv_store, TryGet(testing::HasSubstr("_0")))
      .WillOnce(Return(absl::NotFoundError("not found")));
  EXPECT_CALL(*kv_store, Set(testing::HasSubstr("_0"), "kCopy_autotune_result"))
      .WillOnce(Return(absl::OkStatus()));

  // Shard 0 reads the KV store entry for shard 1 and updates the current cache.
  EXPECT_CALL(*kv_store, Get(testing::HasSubstr("_1"), _))
      .WillOnce(Return("kAdd_autotune_result"));
  EXPECT_CALL(*cache, Deserialize("kAdd_autotune_result"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kAdd)))
      .WillOnce(Return(GetCacheConfig("best_config")));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kCopy},
          /*instrs_to_apply_config_and_count=*/
          {{HloOpcode::kCopy, 1}, {HloOpcode::kAdd, 2}}, std::move(cache)));

  MultiProcessKeyValueStore sharding_kv_store;
  sharding_kv_store.key_value_store = kv_store;
  sharding_kv_store.process_count = kShardCount;
  sharding_kv_store.process_index = 0;
  EXPECT_THAT(
      autotuner->Autotune(module.get(), should_autotune, sharding_kv_store),
      IsOk());
}

TEST_F(AutotunerTest, DumpHlos) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  auto module = ParseAndReturnVerifiedModule(kHlo).value();
  module->mutable_config().mutable_debug_options().set_xla_dump_to(
      dump_dir.path());
  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kCopy ||
           instruction.opcode() == HloOpcode::kAdd;
  };

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kCopy, HloOpcode::kAdd},
          // One apply config call per instruction is expected for dumping HLOs.
          /*instrs_to_apply_config_and_count=*/
          {{HloOpcode::kCopy, 2}, {HloOpcode::kAdd, 3}},
          /*cache=*/nullptr,
          /*dump_hlos=*/true));

  EXPECT_THAT(autotuner->Autotune(module.get(), should_autotune), IsOk());

  std::vector<std::string> files;
  EXPECT_THAT(tsl::Env::Default()->GetChildren(dump_dir.path(), &files),
              IsOk());
  EXPECT_THAT(files.size(), 4);
  EXPECT_THAT(
      files,
      UnorderedElementsAre(
          MatchesRegex(".*\\.test_module\\.autotuner_0\\.copy\\.before\\.txt"),
          MatchesRegex(".*\\.test_module\\.autotuner_0\\.copy\\.after\\.txt"),
          MatchesRegex(".*\\.test_module\\.autotuner_1\\.add\\.after\\.txt"),
          MatchesRegex(".*\\.test_module\\.autotuner_1\\.add\\.before\\.txt")));
}

TEST(AutotuneConfigTest, ToString) {
  AutotuneConfig config;
  config.check_buffers = true;
  config.relative_tolerance = 1e-4;
  config.crash_on_check_failure = false;
  config.optimize_scratch_bytes = true;
  config.scratch_bytes_window_size_us = 10;
  config.expect_all_instructions_in_cache = false;
  config.dump_logs_to = "/tmp/log";
  config.exclude_cublas_config = true;
  config.select_first_config = false;
  config.use_default_config = true;
  config.dump_hlos = false;
  config.allow_reg_spills = false;

  std::string expected =
      "{\n"
      "  \"check_buffers\": true,\n"
      "  \"relative_tolerance\": 0.000100,\n"
      "  \"crash_on_check_failure\": false,\n"
      "  \"optimize_scratch_bytes\": true,\n"
      "  \"scratch_bytes_window_size_us\": 10,\n"
      "  \"expect_all_instructions_in_cache\": false,\n"
      "  \"dump_logs_to\": \"/tmp/log\",\n"
      "  \"exclude_cublas_config\": true,\n"
      "  \"select_first_config\": false,\n"
      "  \"use_default_config\": true,\n"
      "  \"dump_hlos\": false,\n"
      "  \"allow_reg_spills\": false\n"
      "}";
  EXPECT_EQ(config.ToString(), expected);
}

}  // namespace
}  // namespace xla
