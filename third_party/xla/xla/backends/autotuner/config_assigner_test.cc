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

#include "xla/backends/autotuner/config_assigner.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/mock_codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/distributed_runtime/call_options.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/testing/temporary_directory.h"

namespace xla {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::ByMove;
using ::testing::MatchesRegex;
using ::testing::Return;
using ::testing::UnorderedElementsAre;

MATCHER_P(InstructionMatcher, opcode, "") { return arg.opcode() == opcode; }
MATCHER_P(InstrPtrMatcher, opcode, "") { return arg->opcode() == opcode; }

ConfigAssigner::Options GetTestConfigAssignerOptions() {
  ConfigAssigner::Options config;
  return config;
}

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

absl::StatusOr<std::unique_ptr<ConfigAssigner>> CreateConfigAssigner(
    std::vector<std::unique_ptr<CodegenBackend>> codegen_backends,
    std::unique_ptr<Profiler> profiler,
    ConfigAssigner::Options assigner_options,
    std::unique_ptr<AutotunerCacheInterface> cache,
    std::optional<Autotuner::Options> autotuner_options = std::nullopt,
    tsl::thread::ThreadPool* thread_pool = nullptr,
    CodegenOrchestrator::Options orchestrator_options = {}) {
  ABSL_ASSIGN_OR_RETURN(auto orchestrator,
                   CodegenOrchestrator::Create(std::move(codegen_backends),
                                               orchestrator_options));

  if (cache == nullptr) {
    cache = std::make_unique<NoOpAutotunerCache>();
  }

  std::unique_ptr<Autotuner> autotuner = nullptr;
  if (profiler != nullptr) {
    Autotuner::Options opts;
    if (autotuner_options.has_value()) {
      opts = *autotuner_options;
    } else {
      opts.correctness_check_options.enable_correctness_check = false;
    }

    std::vector<std::unique_ptr<Profiler>> profilers;
    profilers.push_back(std::move(profiler));
    ABSL_ASSIGN_OR_RETURN(
        autotuner, Autotuner::Create(*orchestrator, std::move(profilers), opts,
                                     thread_pool));
  }

  return ConfigAssigner::Create(assigner_options, std::move(cache),
                                std::move(orchestrator), std::move(autotuner));
}

absl::StatusOr<std::unique_ptr<ConfigAssigner>>
SetupConfigAssignerWithExpectations(
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
  ConfigAssigner::Options config = GetTestConfigAssignerOptions();
  config.dump_hlos = dump_hlos;
  return CreateConfigAssigner(std::move(backends), std::move(profiler), config,
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

class ConfigAssignerTest : public HloHardwareIndependentTestBase {
 public:
  ConfigAssignerTest() { config_ = GetTestConfigAssignerOptions(); }
  ConfigAssigner::Options config_;
};

TEST_F(ConfigAssignerTest, NoCacheManager) {
  auto device_description = CreateDummyDeviceDescription();
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<MockCodegenBackend>());
  auto config_assigner =
      CreateConfigAssigner(std::move(backends), nullptr, config_, nullptr);
  EXPECT_THAT(config_assigner, IsOk());
}


TEST_F(ConfigAssignerTest, CacheHit) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  AutotunerCacheInterface::Config config;
  config.codegen_backend = autotuner::Backend::UNSPECIFIED_BACKEND;
  config.backend_config = *GetTestConfig("test_config_2");

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
      auto config_assigner,
      CreateConfigAssigner(std::move(backends), std::move(profiler), config_,
                           std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr.get()), IsOk());
}

TEST_F(ConfigAssignerTest, ExpectAllInstructionsInCache) {
  auto cache_manager = std::make_unique<MockAutotunerCache>();
  EXPECT_CALL(*cache_manager, Lookup(_)).WillOnce(Return(std::nullopt));
  EXPECT_CALL(*cache_manager, Insert(_, _)).Times(0);

  config_.expect_all_instructions_in_cache = true;

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs).Times(0);
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(auto config_assigner,
                       CreateConfigAssigner(std::move(backends), nullptr,
                                            config_, std::move(cache_manager)));
  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr.get()),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ConfigAssignerTest, SelectFirstConfig) {
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
      auto config_assigner,
      CreateConfigAssigner(std::move(backends), std::move(profiler), config_,
                           nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr), absl_testing::IsOk());
}

TEST_F(ConfigAssignerTest, SelectFirstConfigPicksFirstCompilable) {
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
      auto config_assigner,
      CreateConfigAssigner(std::move(backends), std::move(profiler), config_,
                           nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr), absl_testing::IsOk());
}

TEST_F(ConfigAssignerTest,
       SelectFirstConfigFallsBackToDefaultIfNoSupportedConfigCompiles) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));

  auto backend = std::make_unique<MockCodegenBackend>();

  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::InternalError("test error")));

  EXPECT_CALL(*backend, GetDefaultConfig(_))
      .WillOnce(Return(ByMove(GetTestConfig("default"))));

  EXPECT_CALL(*backend, ApplyConfig(_, ConfigMatcher("default")))
      .Times(1)
      .WillRepeatedly(Return(absl::OkStatus()));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  auto profiler = std::make_unique<MockProfiler>();

  ASSERT_OK_AND_ASSIGN(
      auto config_assigner,
      CreateConfigAssigner(std::move(backends), std::move(profiler), config_,
                           nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr), absl_testing::IsOk());
}

TEST_F(ConfigAssignerTest,
       SelectFirstConfigFailsWhenNothingCompilesAndNoDefault) {
  config_.select_first_config = true;

  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(absl::InternalError("test error")));
  EXPECT_CALL(*backend, GetDefaultConfig(_))
      .WillOnce(Return(absl::NotFoundError("no default")));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));

  ASSERT_OK_AND_ASSIGN(
      auto config_assigner,
      CreateConfigAssigner(std::move(backends), /*profiler=*/nullptr, config_,
                           /*cache=*/nullptr));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();
  EXPECT_THAT(config_assigner->AssignConfig(dummy_instr),
              StatusIs(absl::StatusCode::kInternal));
}

class MockKeyValueStore : public KeyValueStoreInterface {
 public:
  MOCK_METHOD(absl::Status, Set,
              (absl::string_view key, absl::string_view value), (override));
  MOCK_METHOD(absl::Status, Delete, (absl::string_view key), (override));
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

TEST_F(ConfigAssignerTest, ShardedAutotuning) {
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
      std::unique_ptr<ConfigAssigner> config_assigner,
      SetupConfigAssignerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
          /*instrs_to_apply_config_and_count=*/
          {{HloOpcode::kCopy, 1}, {HloOpcode::kAdd, 2}}, std::move(cache)));

  MultiProcessKeyValueStore sharding_kv_store;
  sharding_kv_store.key_value_store = kv_store;
  sharding_kv_store.process_count = kShardCount;
  sharding_kv_store.process_index = 0;
  EXPECT_THAT(config_assigner->AssignConfigs(module.get(), should_autotune,
                                             sharding_kv_store),
              IsOk());
}

TEST_F(ConfigAssignerTest, ShardedAutotuningTolerateLostSetRace) {
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
  // with AlreadyExists. The config_assigner must treat this as success.
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
      std::unique_ptr<ConfigAssigner> config_assigner,
      SetupConfigAssignerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
          /*instrs_to_apply_config_and_count=*/
          {{HloOpcode::kCopy, 1}, {HloOpcode::kAdd, 2}}, std::move(cache)));

  MultiProcessKeyValueStore sharding_kv_store;
  sharding_kv_store.key_value_store = kv_store;
  sharding_kv_store.process_count = kShardCount;
  sharding_kv_store.process_index = 0;
  EXPECT_THAT(config_assigner->AssignConfigs(module.get(), should_autotune,
                                             sharding_kv_store),
              IsOk());
}

TEST_F(ConfigAssignerTest, ShardedAutotuningPropagatesNonRaceSetError) {
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
      std::unique_ptr<ConfigAssigner> config_assigner,
      SetupConfigAssignerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kAdd},
          // No ApplyConfig calls expected: Autotune bails out before step 6.
          /*instrs_to_apply_config_and_count=*/{}, std::move(cache)));

  MultiProcessKeyValueStore sharding_kv_store;
  sharding_kv_store.key_value_store = kv_store;
  sharding_kv_store.process_count = kShardCount;
  sharding_kv_store.process_index = 0;
  EXPECT_THAT(config_assigner->AssignConfigs(module.get(), should_autotune,
                                             sharding_kv_store),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("disk on fire")));
}

TEST_F(ConfigAssignerTest, DumpHlos) {
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
      std::unique_ptr<ConfigAssigner> config_assigner,
      SetupConfigAssignerWithExpectations(
          /*instrs_to_autotune=*/{HloOpcode::kCopy, HloOpcode::kAdd},
          // One apply config call per instruction is expected for dumping HLOs.
          /*instrs_to_apply_config_and_count=*/
          {{HloOpcode::kCopy, 2}, {HloOpcode::kAdd, 3}},
          /*cache=*/nullptr,
          /*dump_hlos=*/true));

  EXPECT_THAT(config_assigner->AssignConfigs(module.get(), should_autotune),
              IsOk());

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

TEST(ConfigAssignerOptionsTest, ToString) {
  ConfigAssigner::Options config;
  config.expect_all_instructions_in_cache = false;

  config.select_first_config = true;
  config.dump_hlos = false;

  std::string expected =
      "{\n"
      "  \"expect_all_instructions_in_cache\": false,\n"
      "  \"select_first_config\": true,\n"
      "  \"dump_hlos\": false\n"
      "}";
  EXPECT_EQ(config.ToString(), expected);
}

}  // namespace
}  // namespace xla
