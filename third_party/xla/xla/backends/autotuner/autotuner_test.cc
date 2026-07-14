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

#include "xla/backends/autotuner/autotuner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/service/executable.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using absl_testing::StatusIs;
using ::testing::_;
using ::testing::Return;

int64_t GetAlgorithmId(absl::string_view name) {
  static const auto* kConfigMap =
      new absl::flat_hash_map<absl::string_view, int64_t>({
          {"best_config", 1},
          {"another_config", 2},
          {"only_config", 3},
      });
  if (auto it = kConfigMap->find(name); it != kConfigMap->end()) {
    return it->second;
  }
  LOG(FATAL) << "Unknown config name: " << name;
  return 0;
}

MATCHER_P(ConfigMatcher, name, "") {
  if (!arg.has_gemm()) {
    return false;
  }
  return arg.gemm().algorithm() == GetAlgorithmId(name);
}

MATCHER_P(InstructionMatcher, opcode, "") { return arg.opcode() == opcode; }

std::unique_ptr<BackendConfig> GetTestConfig(absl::string_view name) {
  auto config = std::make_unique<BackendConfig>();
  config->mutable_gemm()->set_algorithm(GetAlgorithmId(name));
  return config;
}

class MockCodegenBackend : public CodegenBackend {
 public:
  MOCK_METHOD(absl::string_view, name, (), (const, override));
  MOCK_METHOD(autotuner::Backend, backend, (), (const, override));
  MOCK_METHOD(std::string, version, (), (const, override));
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

class AutotunerTest : public HloHardwareIndependentTestBase {
 protected:
  Autotuner::Options options_;
};

TEST_F(AutotunerTest, NullOrchestrator) {
  auto profiler = std::make_unique<MockProfiler>();
  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));

  std::unique_ptr<CodegenOrchestrator> null_orchestrator = nullptr;
  auto autotuner = Autotuner::Create(std::move(null_orchestrator),
                                     std::move(profilers), options_);
  EXPECT_THAT(autotuner, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, EmptyProfilers) {
  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::make_unique<MockCodegenBackend>());
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  auto autotuner = Autotuner::Create(std::move(orchestrator), {}, options_);
  EXPECT_THAT(autotuner, StatusIs(absl::StatusCode::kInternal));
}

TEST_F(AutotunerTest, AutotuneSingleSupportedConfig) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("only_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _)).WillOnce([] {
    return std::unique_ptr<Executable>();
  });

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  Shape shape = ShapeUtil::MakeShape(F32, {});
  EXPECT_CALL(*profiler, Profile(_, _)).WillOnce([shape] {
    ProfileResult result;
    result.duration = absl::Microseconds(100);
    result.output_buffer = ScopedShapedBuffer(shape, nullptr, 0);
    return result;
  });

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(orchestrator),
                                         std::move(profilers), options_));

  auto dummy_instr = HloInstruction::CreateConstant(LiteralUtil::CreateR0(1));

  constexpr absl::string_view kHlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT copy = f32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(
      auto results,
      autotuner->TuneConfigs(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kCopy;
      }));
  EXPECT_FALSE(results.empty());
  EXPECT_EQ(results[0].config.backend_config->gemm().algorithm(),
            GetAlgorithmId("only_config"));
}

TEST_F(AutotunerTest, AutotuneMultipleConfigsSelectsBest) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("best_config"));
  configs.push_back(GetTestConfig("another_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _)).Times(2).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  Shape shape = ShapeUtil::MakeShape(F32, {});
  EXPECT_CALL(*profiler, Profile(_, _))
      .Times(2)
      .WillOnce([shape] {
        ProfileResult result;
        result.duration = absl::Microseconds(100);
        result.output_buffer = ScopedShapedBuffer(shape, nullptr, 0);
        return result;
      })
      .WillOnce([shape] {
        ProfileResult result;
        result.duration = absl::Microseconds(500);
        result.output_buffer = ScopedShapedBuffer(shape, nullptr, 0);
        return result;
      });

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(orchestrator),
                                         std::move(profilers), options_));

  constexpr absl::string_view kHlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT copy = f32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(
      auto results,
      autotuner->TuneConfigs(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kCopy;
      }));
  EXPECT_FALSE(results.empty());
  EXPECT_EQ(results[0].config.backend_config->gemm().algorithm(),
            GetAlgorithmId("best_config"));
}

TEST_F(AutotunerTest, AutotuneToleratesOnlyNoSupportedConfigs) {
  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  // Return empty supported configs list showing "No supported configs found"
  // (NotFoundError).
  EXPECT_CALL(*backend, GetSupportedConfigs).WillRepeatedly([] {
    return std::vector<std::unique_ptr<BackendConfig>>();
  });

  auto profiler = std::make_unique<MockProfiler>();

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(orchestrator),
                                         std::move(profilers), options_));

  constexpr absl::string_view kHlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT copy = f32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  // Should fail by default (without tolerating the NotFound error).
  EXPECT_FALSE(autotuner
                   ->TuneConfigs(
                       *module,
                       [](const HloInstruction& instr) {
                         return instr.opcode() == HloOpcode::kCopy;
                       },
                       /*tolerate_no_supported_configs=*/false)
                   .ok());

  // Should succeed (returning empty results) if requested.
  ASSERT_OK_AND_ASSIGN(auto results,
                       autotuner->TuneConfigs(
                           *module,
                           [](const HloInstruction& instr) {
                             return instr.opcode() == HloOpcode::kCopy;
                           },
                           /*tolerate_no_supported_configs=*/true));
  EXPECT_TRUE(results.empty());
}

TEST_F(AutotunerTest, AutotuneCompileErrorWithNoSupportedConfigsTolerance) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("best_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  // Return compilation failure (InternalError)
  EXPECT_CALL(*backend, Compile(_, _)).WillRepeatedly([] {
    return absl::InternalError("Failed compilation");
  });

  auto profiler = std::make_unique<MockProfiler>();

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(orchestrator),
                                         std::move(profilers), options_));

  constexpr absl::string_view kHlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT copy = f32[] copy(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  // Should fail even if tolerate_no_supported_configs is true since the
  // failure is a compile error (InternalError), not a missing configs error.
  EXPECT_FALSE(autotuner
                   ->TuneConfigs(
                       *module,
                       [](const HloInstruction& instr) {
                         return instr.opcode() == HloOpcode::kCopy;
                       },
                       /*tolerate_no_supported_configs=*/true)
                   .ok());
}

TEST_F(AutotunerTest, AutotuneMultipleDevicesRoundRobin) {
  std::vector<std::unique_ptr<BackendConfig>> configs0;
  configs0.push_back(GetTestConfig("best_config"));

  std::vector<std::unique_ptr<BackendConfig>> configs1;
  configs1.push_back(GetTestConfig("another_config"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs(_))
      .WillOnce(Return(std::move(configs0)))
      .WillOnce(Return(std::move(configs1)));
  EXPECT_CALL(*backend, Compile(_, _)).WillRepeatedly([] {
    return std::unique_ptr<Executable>();
  });

  auto profiler0 = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler0, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler0, Profile(_, _)).WillOnce([] {
    ProfileResult result;
    result.duration = absl::Microseconds(100);
    result.output_buffer =
        ScopedShapedBuffer(ShapeUtil::MakeShape(F32, {2}), nullptr, 0);
    return result;
  });

  auto profiler1 = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler1, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler1, Profile(_, _)).WillOnce([] {
    ProfileResult result;
    result.duration = absl::Microseconds(200);
    result.output_buffer =
        ScopedShapedBuffer(ShapeUtil::MakeShape(F32, {4}), nullptr, 0);
    return result;
  });

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler0));
  profilers.push_back(std::move(profiler1));

  ASSERT_OK_AND_ASSIGN(auto autotuner,
                       Autotuner::Create(std::move(orchestrator),
                                         std::move(profilers), options_));

  constexpr absl::string_view kHlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2] parameter(0)
      p1 = f32[4] parameter(1)
      copy0 = f32[2] copy(p0)
      copy1 = f32[4] copy(p1)
      ROOT tuple = (f32[2], f32[4]) tuple(copy0, copy1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  ASSERT_OK_AND_ASSIGN(
      auto results,
      autotuner->TuneConfigs(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kCopy;
      }));
  EXPECT_EQ(results.size(), 2);
}

}  // namespace
}  // namespace xla
