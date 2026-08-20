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
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/text_format.h"
#include "xla/autotune_results.pb.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/mock_codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

using absl_testing::IsOk;
using absl_testing::StatusIs;
using ::testing::_;
using ::testing::Return;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

class FakeExecutable : public Executable {
 public:
  FakeExecutable() : Executable(nullptr) {}
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions*,
      std::vector<ExecutionInput>) override {
    return absl::UnimplementedError("unused in test");
  }
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

TEST_F(AutotunerTest, AutotuneAppliesBestConfigAndSkipsNonCompilableConfig) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("non_compilable_config"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, _))
      .WillOnce(Return(std::unique_ptr<Executable>()))
      .WillOnce(Return(absl::InternalError("test error")))
      .WillOnce(Return(std::unique_ptr<Executable>()));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));
  options_.correctness_check_options.enable_correctness_check = false;

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
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].config.backend_config->gemm().algorithm(),
            GetAlgorithmId("test_config_2"));
}

TEST_F(AutotunerTest, AutotuneAppliesBestConfigUsingThreadPool) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  std::unique_ptr<Executable> executable1 = std::make_unique<FakeExecutable>();
  Executable* exec1 = executable1.get();
  std::unique_ptr<Executable> executable2 = std::make_unique<FakeExecutable>();
  Executable* exec2 = executable2.get();

  auto backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*backend, name()).WillRepeatedly(Return("mock_backend"));
  EXPECT_CALL(*backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_1")))
      .WillOnce(Return(std::move(executable1)));
  EXPECT_CALL(*backend, Compile(_, ConfigMatcher("test_config_2")))
      .WillOnce(Return(std::move(executable2)));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(testing::Pointer(exec1), _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})));
  EXPECT_CALL(*profiler, Profile(testing::Pointer(exec2), _))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));
  options_.correctness_check_options.enable_correctness_check = false;

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 2);
  ASSERT_OK_AND_ASSIGN(
      auto autotuner,
      Autotuner::Create(std::move(orchestrator), std::move(profilers), options_,
                        &thread_pool));

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
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].config.backend_config->gemm().algorithm(),
            GetAlgorithmId("test_config_2"));
}

TEST_F(AutotunerTest, AutotuneToleratesSingleBackendFailure) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("test_config_1"));
  configs.push_back(GetTestConfig("test_config_2"));

  auto good_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*good_backend, name()).WillRepeatedly(Return("good_backend"));
  EXPECT_CALL(*good_backend, GetSupportedConfigs)
      .WillOnce(Return(std::move(configs)));
  EXPECT_CALL(*good_backend, Compile(_, _))
      .WillRepeatedly([](const HloInstruction&, const BackendConfig&) {
        return std::unique_ptr<Executable>();
      });

  auto bad_backend = std::make_unique<MockCodegenBackend>();
  EXPECT_CALL(*bad_backend, name()).WillRepeatedly(Return("bad_backend"));
  EXPECT_CALL(*bad_backend, GetSupportedConfigs)
      .WillOnce(Return(absl::InternalError("backend error")));

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1)})));

  std::vector<std::unique_ptr<CodegenBackend>> backends;
  backends.push_back(std::move(good_backend));
  backends.push_back(std::move(bad_backend));
  ASSERT_OK_AND_ASSIGN(auto orchestrator,
                       CodegenOrchestrator::Create(std::move(backends), {}));

  std::vector<std::unique_ptr<Profiler>> profilers;
  profilers.push_back(std::move(profiler));
  options_.correctness_check_options.enable_correctness_check = false;

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
  ASSERT_FALSE(results.empty());
  EXPECT_EQ(results[0].config.backend_config->gemm().algorithm(),
            GetAlgorithmId("test_config_2"));
}

TEST_F(AutotunerTest, DumpLogsToFile) {
  ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory temp_dir,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  options_.dump_logs_to = tsl::io::JoinPath(temp_dir.path(), "dump.log");
  options_.correctness_check_options.enable_correctness_check = false;

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
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  auto dummy_instr = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(
      auto results,
      autotuner->TuneConfigs(*module, [](const HloInstruction& instr) {
        return instr.opcode() == HloOpcode::kCopy;
      }));
  EXPECT_FALSE(results.empty());
  EXPECT_OK(autotuner->DumpTuningLogs());

  std::string content;
  EXPECT_THAT(tsl::ReadFileToString(tsl::Env::Default(), options_.dump_logs_to,
                                    &content),
              IsOk());
  AutotuningLogs actual_logs;
  EXPECT_TRUE(
      tsl::protobuf::TextFormat::ParseFromString(content, &actual_logs));

  auto expected_logs = ParseTextProtoOrDie<AutotuningLogs>(absl::StrFormat(
      R"pb(
        logs {
          results {
            gemm { algorithm: %d }
            run_time { seconds: 2 nanos: 0 }
            scratch_bytes: 100
          }
          results {
            gemm { algorithm: %d }
            run_time { seconds: 1 nanos: 0 }
          }
          results {
            gemm { algorithm: %d }
            run_time { seconds: 0 nanos: 0 }
            failure {
              kind: DISQUALIFIED
              msg: "INTERNAL: Compilation failed: failed to compile"
            }
          }
        }
      )pb",
      GetAlgorithmId("test_config_1"), GetAlgorithmId("test_config_2"),
      GetAlgorithmId("test_config_failure")));
  expected_logs.mutable_logs(0)->mutable_instr()->PackFrom(
      dummy_instr->ToProto());

  EXPECT_THAT(actual_logs, EqualsProto(expected_logs));
}

TEST_F(AutotunerTest, AutotuneCompileErrorWithNoSupportedConfigs) {
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

  // Should fail as there are no supported configs.
  EXPECT_FALSE(autotuner
                   ->TuneConfigs(*module,
                                 [](const HloInstruction& instr) {
                                   return instr.opcode() == HloOpcode::kCopy;
                                 })
                   .ok());
}

TEST_F(AutotunerTest, AutotuneCompileErrorWithNoCompiledCandidates) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  configs.push_back(GetTestConfig("best_config"));
  configs.push_back(GetTestConfig("another_config"));

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

  EXPECT_FALSE(autotuner
                   ->TuneConfigs(*module,
                                 [](const HloInstruction& instr) {
                                   return instr.opcode() == HloOpcode::kCopy;
                                 })
                   .ok());
}

TEST_F(AutotunerTest, AutotuneMultipleDevicesRoundRobin) {
  std::vector<std::unique_ptr<BackendConfig>> configs0;
  configs0.push_back(GetTestConfig("best_config"));
  configs0.push_back(GetTestConfig("some_config"));

  std::vector<std::unique_ptr<BackendConfig>> configs1;
  configs1.push_back(GetTestConfig("another_config"));
  configs1.push_back(GetTestConfig("some_other_config"));

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
  EXPECT_CALL(*profiler0, Profile(_, _)).Times(2).WillRepeatedly([] {
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
  EXPECT_CALL(*profiler1, Profile(_, _)).Times(2).WillRepeatedly([] {
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
