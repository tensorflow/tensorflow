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

#include <atomic>
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
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/kernel_stats.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

// Use one of existing gpu backend config protos as a test config.
using TestConfig = gpu::CustomFusionConfig;
using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::AtMost;
using ::testing::ByMove;
using ::testing::MatchesRegex;
using ::testing::Return;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

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
  MOCK_METHOD(autotuner::Backend, backend, (), (const, override));
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
              (const Executable* executable, const HloInstruction* instr),
              (override));
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
  MOCK_METHOD(CacheStats, GetCacheStats, (), (const, override));
};

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
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
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

std::unique_ptr<Executable> RegisterSpillingExecutable(int spilled = 8) {
  gpu::GpuExecutable::Params params;
  params.executable =
      std::make_unique<gpu::ThunkExecutor>(gpu::ThunkSequence{});
  KernelStats kernel_stats;
  kernel_stats.store_bytes_spilled = spilled;
  kernel_stats.load_bytes_spilled = spilled;
  params.module_stats = {{"test_config_2", kernel_stats}};
  return gpu::GpuExecutable::Create(std::move(params)).value();
}

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

TEST_F(AutotunerTest, AutotuneSingleSupportedConfig) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("only_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _)).Times(0);
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("only_config")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).Times(0);
  EXPECT_CALL(*profiler, Profile(_, _)).Times(0);

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), absl_testing::IsOk());
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
  configs.push_back(GetTestConfig("invalid_config_1"));
  configs.push_back(GetTestConfig("invalid_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillRepeatedly([](const HloInstruction&, const BackendConfig&) {
        return absl::InternalError("test error");
      });

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
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
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

  std::unique_ptr<Executable> executable1 = RegisterSpillingExecutable(0);
  Executable* exec1 = executable1.get();
  std::unique_ptr<Executable> executable2 = RegisterSpillingExecutable(0);
  Executable* exec2 = executable2.get();

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::move(executable1)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(std::move(executable2)));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  auto device_description = CreateDummyDeviceDescription();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(testing::Pointer(exec1), _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})));
  EXPECT_CALL(*profiler, Profile(testing::Pointer(exec2), _))
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
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto good_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*good_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*good_backend, Compile(_, _))
      .WillRepeatedly([](const HloInstruction&, const BackendConfig&) {
        return std::unique_ptr<Executable>();
      });
  EXPECT_CALL(*good_backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));
  auto bad_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend, GetSupportedConfigs)
      .WillOnce(Return(absl::InternalError("test error")));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

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
  config.codegen_backend = autotuner::Backend::UNSPECIFIED_BACKEND;
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

  // Trusted config wins: its cluster is the only trusted one, so it wins
  // regardless of the untrusted config's runtime.
  EXPECT_CALL(*backend_1, ApplyConfig(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer output_1(Shape(), nullptr, 0),
      output_2(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  // Unified clustering path: one Profile per candidate (no pre-loop
  // GetReferenceOutput). Trusted backend profiled first, untrusted second.
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(output_1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(output_2)})));
  // One CheckOutputBuffer call: untrusted's output vs the trusted cluster's
  // representative. Mismatch -> no cluster -> demoted.
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
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

TEST_F(AutotunerTest, AutotuneClustersOutputsWhenAllBackendsUntrusted) {
  config_.check_buffers = true;

  // Three untrustworthy configs. Outputs split into a 2-member majority
  // cluster and a 1-member minority cluster. The minority member is the
  // globally fastest, but the majority cluster's fastest should be picked —
  // proving that cluster size beats raw runtime when no backend is trusted.
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("majority_slow"));
  configs.push_back(GetTestConfig("majority_fast"));
  configs.push_back(GetTestConfig("minority_fastest"));
  auto backend = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _)).Times(3).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("majority_fast")))
      .WillOnce(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_a1(Shape(), nullptr, 0), out_a2(Shape(), nullptr, 0),
      out_b(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(3), std::move(out_a1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(out_a2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_b)})));
  // Cluster assignment compares:
  //   config 0 -> no clusters yet, creates cluster A (no call).
  //   config 1 -> vs cluster A -> match, joins A.
  //   config 2 -> vs cluster A -> mismatch, creates cluster B.
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("minority")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneTrustedClusterWinsOverLargerUntrustedCluster) {
  config_.check_buffers = true;

  // One trusted config vs three untrusted configs whose outputs agree with
  // each other (but not with trusted). The untrusted cluster is 3x larger
  // and globally fastest, but the trusted cluster still wins.
  std::vector<std::unique_ptr<BackendConfig>> configs_1;
  configs_1.push_back(GetTestConfig("trusted_config"));
  auto backend_1 = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend_1, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs_1)));
  EXPECT_CALL(*backend_1, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  std::vector<std::unique_ptr<BackendConfig>> configs_2;
  configs_2.push_back(GetTestConfig("untrusted_1"));
  configs_2.push_back(GetTestConfig("untrusted_2"));
  configs_2.push_back(GetTestConfig("untrusted_3"));
  auto backend_2 = std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*backend_2, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs_2)));
  EXPECT_CALL(*backend_2, Compile(_, _)).Times(3).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  EXPECT_CALL(*backend_1, ApplyConfig(_, ConfigMatcher("trusted_config")))
      .WillOnce(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted(Shape(), nullptr, 0),
      out_u1(Shape(), nullptr, 0), out_u2(Shape(), nullptr, 0),
      out_u3(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(10), std::move(out_trusted)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u3)})));
  // Once a trusted cluster exists, untrusted outputs only compare against
  // trusted-backed clusters and cannot create loser clusters.
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("u1 vs c0")))
      .WillOnce(Return(absl::InternalError("u2 vs c0")))
      .WillOnce(Return(absl::InternalError("u3 vs c0")));

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

TEST_F(AutotunerTest, AutotuneProfilesTrustedFirstPreservesCandidateOrder) {
  config_.check_buffers = true;

  std::vector<std::unique_ptr<BackendConfig>> untrusted_configs;
  untrusted_configs.push_back(GetTestConfig("untrusted_first"));
  auto untrusted_backend =
      std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*untrusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(untrusted_configs)));
  std::unique_ptr<Executable> untrusted_executable =
      RegisterSpillingExecutable(0);
  Executable* untrusted_exec = untrusted_executable.get();
  EXPECT_CALL(*untrusted_backend, Compile(_, _))
      .WillOnce(Return(std::move(untrusted_executable)));

  std::vector<std::unique_ptr<BackendConfig>> trusted_configs;
  trusted_configs.push_back(GetTestConfig("trusted_second"));
  auto trusted_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*trusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(trusted_configs)));
  EXPECT_CALL(*trusted_backend, CanProduceWrongResults())
      .WillRepeatedly(Return(false));
  std::unique_ptr<Executable> trusted_executable =
      RegisterSpillingExecutable(0);
  Executable* trusted_exec = trusted_executable.get();
  EXPECT_CALL(*trusted_backend, Compile(_, _))
      .WillOnce(Return(std::move(trusted_executable)));

  // The untrusted config is first in candidate order. It still wins the tie
  // after both outputs join the trusted cluster, proving that profiling order
  // does not leak into PickBestConfig tie-breaking.
  EXPECT_CALL(*untrusted_backend,
              ApplyConfig(_, ConfigMatcher("untrusted_first")))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*trusted_backend, ApplyConfig(_, _)).Times(0);

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted(Shape(), nullptr, 0),
      out_untrusted(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  {
    ::testing::InSequence sequence;
    EXPECT_CALL(*profiler, Profile(testing::Pointer(trusted_exec), _))
        .WillOnce(
            Return(ProfileResult({absl::Seconds(1), std::move(out_trusted)})));
    EXPECT_CALL(*profiler, Profile(testing::Pointer(untrusted_exec), _))
        .WillOnce(Return(
            ProfileResult({absl::Seconds(1), std::move(out_untrusted)})));
  }
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(untrusted_backend));
  backends.push_back(std::move(trusted_backend));
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneUntrustedVotesForTrustedCluster) {
  config_.check_buffers = true;

  std::vector<std::unique_ptr<BackendConfig>> trusted_configs;
  trusted_configs.push_back(GetTestConfig("trusted_a"));
  trusted_configs.push_back(GetTestConfig("trusted_b"));
  auto trusted_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*trusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(trusted_configs)));
  EXPECT_CALL(*trusted_backend, Compile(_, _)).Times(2).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  std::vector<std::unique_ptr<BackendConfig>> untrusted_configs;
  untrusted_configs.push_back(GetTestConfig("untrusted_matches_b"));
  untrusted_configs.push_back(GetTestConfig("untrusted_matches_none"));
  auto untrusted_backend =
      std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*untrusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(untrusted_configs)));
  EXPECT_CALL(*untrusted_backend, Compile(_, _)).Times(2).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  EXPECT_CALL(*untrusted_backend,
              ApplyConfig(_, ConfigMatcher("untrusted_matches_b")))
      .WillOnce(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted_a(Shape(), nullptr, 0),
      out_trusted_b(Shape(), nullptr, 0), out_untrusted_b(Shape(), nullptr, 0),
      out_untrusted_none(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(10), std::move(out_trusted_a)})))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(9), std::move(out_trusted_b)})))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(1), std::move(out_untrusted_b)})))
      .WillOnce(Return(
          ProfileResult({absl::Seconds(1), std::move(out_untrusted_none)})));
  // trusted_b creates the second trusted cluster. The first untrusted config
  // votes for it, making it the largest trusted-backed cluster. The second
  // untrusted config matches no trusted cluster and is demoted.
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("trusted_b vs trusted_a")))
      .WillOnce(Return(absl::InternalError("untrusted_b vs trusted_a")))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("untrusted_none vs trusted_a")))
      .WillOnce(Return(absl::InternalError("untrusted_none vs trusted_b")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(trusted_backend));
  backends.push_back(std::move(untrusted_backend));
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::make_unique<MockAutotunerCache>()));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(autotuner->Autotune(dummy_instr.get()), IsOk());
}

TEST_F(AutotunerTest, AutotuneClustersUntrustedWhenTrustedReferenceFails) {
  config_.check_buffers = true;

  std::vector<std::unique_ptr<BackendConfig>> trusted_configs;
  trusted_configs.push_back(GetTestConfig("trusted_fails"));
  auto trusted_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*trusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(trusted_configs)));
  EXPECT_CALL(*trusted_backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  std::vector<std::unique_ptr<BackendConfig>> untrusted_configs;
  untrusted_configs.push_back(GetTestConfig("majority_slow"));
  untrusted_configs.push_back(GetTestConfig("majority_fast"));
  untrusted_configs.push_back(GetTestConfig("minority_fastest"));
  auto untrusted_backend =
      std::make_unique<MockCodegenBackendWithWrongResults>();
  EXPECT_CALL(*untrusted_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(untrusted_configs)));
  EXPECT_CALL(*untrusted_backend, Compile(_, _)).Times(3).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  EXPECT_CALL(*untrusted_backend,
              ApplyConfig(_, ConfigMatcher("majority_fast")))
      .WillOnce(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_a1(Shape(), nullptr, 0), out_a2(Shape(), nullptr, 0),
      out_b(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(absl::InternalError("trusted failed")))
      .WillOnce(Return(ProfileResult({absl::Seconds(3), std::move(out_a1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(out_a2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_b)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("minority")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(trusted_backend));
  backends.push_back(std::move(untrusted_backend));
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
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
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
  configs.push_back(GetTestConfig("test_config_failure"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(absl::InternalError("failed to compile")))
      .WillOnce(Return(std::unique_ptr<Executable>()));
  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("test_config_2")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());

  std::string content;
  EXPECT_THAT(tsl::ReadFileToString(tsl::Env::Default(), config_.dump_logs_to,
                                    &content),
              absl_testing::IsOk());
  AutotuningLogs actual_logs;
  EXPECT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(content, &actual_logs));

  auto expected_logs = ParseTextProtoOrDie<AutotuningLogs>(R"pb(
    logs {
      results {
        other {
          name: "mock_backend"
          config {
            [type.googleapis.com/xla.gpu.CustomFusionConfig] {
              name: "test_config_failure"
            }
          }
        }
        run_time { seconds: 0 nanos: 0 }
        failure {
          kind: DISQUALIFIED
          msg: "INTERNAL: Compilation failed: failed to compile"
        }
      }
      results {
        other {
          name: "mock_backend"
          config {
            [type.googleapis.com/xla.gpu.CustomFusionConfig] {
              name: "test_config_1"
            }
          }
        }
        run_time { seconds: 2 nanos: 0 }
        scratch_bytes: 100
      }
      results {
        other {
          name: "mock_backend"
          config {
            [type.googleapis.com/xla.gpu.CustomFusionConfig] {
              name: "test_config_2"
            }
          }
        }
        run_time { seconds: 1 nanos: 0 }
      }
    }
  )pb");
  expected_logs.mutable_logs(0)->mutable_instr()->PackFrom(
      dummy_instr->ToProto());

  EXPECT_THAT(actual_logs, EqualsProto(expected_logs));
}

class AutotunerTestWithBackendName
    : public AutotunerTest,
      public ::testing::WithParamInterface<std::string> {};

TEST_P(AutotunerTestWithBackendName, ExcludeCublasConfig) {
  config_.exclude_cublas_config = true;
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return(GetParam()));
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();
  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr),
              StatusIs(absl::StatusCode::kInternal));
}

INSTANTIATE_TEST_SUITE_P(ExcludeCublasConfigInstance,
                         AutotunerTestWithBackendName,
                         ::testing::Values("CUBLAS_FISSION",
                                           "CUBLASLT_FISSION"));

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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(autotuner->Autotune(dummy_instr), absl_testing::IsOk());
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
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner, Autotuner::Create(std::move(backends),
                                        std::move(profiler), config_, nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
  config.codegen_backend = autotuner::Backend::UNSPECIFIED_BACKEND;
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

  // Shard 0 autotunes kAdd instructions, updates the cache and serializes the
  // result to a string "kAdd_autotune_result".
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kAdd)))
      .WillOnce(Return(std::nullopt))                    // During autotuning.
      .WillOnce(Return(GetCacheConfig("best_config")));  // Config application.
  EXPECT_CALL(*cache, Insert(InstrPtrMatcher(HloOpcode::kAdd), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Serialize(_)).WillOnce(Return("kAdd_autotune_result"));
  // Stores the serialized results to the KV store if it does not exist.
  EXPECT_CALL(*kv_store, TryGet(testing::HasSubstr("_0")))
      .WillOnce(Return(absl::NotFoundError("not found")));
  EXPECT_CALL(*kv_store, Set(testing::HasSubstr("_0"), "kAdd_autotune_result"))
      .WillOnce(Return(absl::OkStatus()));

  // Shard 0 reads the KV store entry for shard 1 and updates the current cache.
  EXPECT_CALL(*kv_store, Get(testing::HasSubstr("_1"), _))
      .WillOnce(Return("kCopy_autotune_result"));
  EXPECT_CALL(*cache, Deserialize("kCopy_autotune_result"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kCopy)))
      .WillOnce(Return(GetCacheConfig("best_config")));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
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

TEST_F(AutotunerTest, ShardedAutotuningTolerateLostSetRace) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  constexpr int kShardCount = 2;
  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kAdd ||
           instruction.opcode() == HloOpcode::kCopy;
  };
  auto kv_store = std::make_shared<MockKeyValueStore>();
  auto cache = std::make_unique<MockAutotunerCache>();

  // Same setup as ShardedAutotuning: shard 0 autotunes kAdd and serializes the
  // result.
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kAdd)))
      .WillOnce(Return(std::nullopt))
      .WillOnce(Return(GetCacheConfig("best_config")));
  EXPECT_CALL(*cache, Insert(InstrPtrMatcher(HloOpcode::kAdd), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Serialize(_)).WillOnce(Return("kAdd_autotune_result"));

  // The KV store reports the slot as empty, so the shard tries to Set...
  EXPECT_CALL(*kv_store, TryGet(testing::HasSubstr("_0")))
      .WillOnce(Return(absl::NotFoundError("not found")));
  // ...but a peer wins the race and the underlying store rejects our write
  // with AlreadyExists. The autotuner must treat this as success.
  EXPECT_CALL(*kv_store, Set(testing::HasSubstr("_0"), "kAdd_autotune_result"))
      .WillOnce(Return(absl::AlreadyExistsError("lost the race")));

  // Shard 0 still reads the KV store entry for shard 1 and applies configs
  // exactly as in the non-racy case.
  EXPECT_CALL(*kv_store, Get(testing::HasSubstr("_1"), _))
      .WillOnce(Return("kCopy_autotune_result"));
  EXPECT_CALL(*cache, Deserialize("kCopy_autotune_result"))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kCopy)))
      .WillOnce(Return(GetCacheConfig("best_config")));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
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

TEST_F(AutotunerTest, ShardedAutotuningPropagatesNonRaceSetError) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  constexpr int kShardCount = 2;
  auto should_autotune = [](const HloInstruction& instruction) {
    return instruction.opcode() == HloOpcode::kAdd ||
           instruction.opcode() == HloOpcode::kCopy;
  };
  auto kv_store = std::make_shared<MockKeyValueStore>();
  auto cache = std::make_unique<MockAutotunerCache>();

  // Shard 0 autotunes kAdd and serializes the result, exactly as in the happy
  // path.
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kAdd)))
      .WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache, Insert(InstrPtrMatcher(HloOpcode::kAdd), _))
      .WillOnce(Return(absl::OkStatus()));
  EXPECT_CALL(*cache, Serialize(_)).WillOnce(Return("kAdd_autotune_result"));

  // The KV store reports the slot as empty, so the shard tries to Set...
  EXPECT_CALL(*kv_store, TryGet(testing::HasSubstr("_0")))
      .WillOnce(Return(absl::NotFoundError("not found")));
  // ...and the underlying store fails for an unrelated reason. This must NOT
  // be silently swallowed: only AlreadyExists is treated as a lost race.
  EXPECT_CALL(*kv_store, Set(testing::HasSubstr("_0"), "kAdd_autotune_result"))
      .WillOnce(Return(absl::InternalError("disk on fire")));

  // Because Autotune returns early, we must not see any peer reads, cache
  // deserialization or config application. Leaving these expectations unset
  // (no EXPECT_CALL) means the mocks would only warn on unexpected calls; we
  // make the contract explicit by asserting they never happen.
  EXPECT_CALL(*kv_store, Get(_, _)).Times(0);
  EXPECT_CALL(*cache, Deserialize(_)).Times(0);
  EXPECT_CALL(*cache, Lookup(InstrPtrMatcher(HloOpcode::kCopy))).Times(0);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Autotuner> autotuner,
      SetupAutotunerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
          // No ApplyConfig calls expected: Autotune bails out before step 6.
          /*instrs_to_apply_config_and_count=*/{}, std::move(cache)));

  MultiProcessKeyValueStore sharding_kv_store;
  sharding_kv_store.key_value_store = kv_store;
  sharding_kv_store.process_count = kShardCount;
  sharding_kv_store.process_index = 0;
  EXPECT_THAT(
      autotuner->Autotune(module.get(), should_autotune, sharding_kv_store),
      StatusIs(absl::StatusCode::kInternal,
               testing::HasSubstr("disk on fire")));
}

TEST_F(AutotunerTest, DumpHlos) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
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
          MatchesRegex(".*\\.test_module\\.autotuner_0\\.add\\.before\\.txt"),
          MatchesRegex(".*\\.test_module\\.autotuner_0\\.add\\.after\\.txt"),
          MatchesRegex(".*\\.test_module\\.autotuner_1\\.copy\\.after\\.txt"),
          MatchesRegex(
              ".*\\.test_module\\.autotuner_1\\.copy\\.before\\.txt")));
}

class CountingDestructorExecutable : public Executable {
 public:
  explicit CountingDestructorExecutable(std::atomic<int>* destroy_count)
      : Executable(nullptr), destroy_count_(destroy_count) {}
  ~CountingDestructorExecutable() override {
    absl::SleepFor(absl::Milliseconds(10));
    ++(*destroy_count_);
  }
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions*,
      std::vector<ExecutionInput>) override {
    return absl::UnimplementedError("unused in test");
  }

 private:
  std::atomic<int>* destroy_count_;
};

TEST_F(AutotunerTest, ProfileAllUnloadsCandidatesBeforeReleasingProfilerLock) {
  std::atomic<int> executable_destroy_count{0};

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::UNSPECIFIED_BACKEND));
  EXPECT_CALL(*backend, GetSupportedConfigs(_)).WillRepeatedly([]() {
    std::vector<std::unique_ptr<BackendConfig>> out;
    out.push_back(GetTestConfig("config_1"));
    out.push_back(GetTestConfig("config_2"));
    return out;
  });
  EXPECT_CALL(*backend, Compile(_, _))
      .WillRepeatedly([&executable_destroy_count]() {
        return std::make_unique<CountingDestructorExecutable>(
            &executable_destroy_count);
      });
  EXPECT_CALL(*backend, ApplyConfig(_, _))
      .WillRepeatedly(Return(absl::OkStatus()));

  std::atomic<int> create_input_buffers_call_count{0};
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillRepeatedly([&executable_destroy_count,
                       &create_input_buffers_call_count](
                          const Executable*, const HloInstruction*) {
        int call = ++create_input_buffers_call_count;
        if (call == 2) {
          EXPECT_EQ(executable_destroy_count.load(), 2)
              << "First run's executables must be destroyed before second run "
                 "acquires profiler lock (avoids delay kernel timeouts)";
        }
        return std::make_unique<InputBuffers>();
      });
  EXPECT_CALL(*profiler, Profile(_, _)).WillRepeatedly([] {
    return ProfileResult({absl::Seconds(1)});
  });

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  auto cache = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache, Lookup(_)).WillRepeatedly(Return(std::nullopt));
  EXPECT_CALL(*cache, Insert(_, _)).WillRepeatedly(Return(absl::OkStatus()));

  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler), config_,
                        std::move(cache)));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  absl::Status status1;
  absl::Status status2;
  std::unique_ptr<tsl::Thread> t1(
      tsl::Env::Default()->StartThread({}, "autotuner-test-t1", [&]() {
        status1 = autotuner->Autotune(
            module->entry_computation()->GetInstructionWithName("add"));
      }));
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread({}, "autotuner-test-t2", [&]() {
        status2 = autotuner->Autotune(
            module->entry_computation()->GetInstructionWithName("copy"));
      }));
  t1.reset();
  t2.reset();
  ASSERT_OK(status1);
  ASSERT_OK(status2);
}

TEST(AutotuneConfigTest, ToString) {
  AutotuneConfig config;
  config.check_buffers = true;
  config.relative_tolerance = 1e-4;
  config.crash_on_check_failure = false;
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
