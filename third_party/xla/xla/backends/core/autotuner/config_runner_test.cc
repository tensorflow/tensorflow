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

#include "xla/backends/core/autotuner/config_runner.h"

#include <atomic>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/core/autotuner/autotuning.pb.h"
#include "xla/backends/core/autotuner/backends.pb.h"
#include "xla/backends/core/autotuner/mock_codegen_backend.h"
#include "xla/backends/core/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/executable.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace {

using absl_testing::StatusIs;
using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::Return;
using ::testing::SizeIs;

class CountingDestructorExecutable : public Executable {
 public:
  explicit CountingDestructorExecutable(
      std::atomic<int>* destroy_count = nullptr)
      : Executable(nullptr), destroy_count_(destroy_count) {}
  ~CountingDestructorExecutable() override {
    if (destroy_count_ != nullptr) {
      absl::SleepFor(absl::Milliseconds(10));
      ++(*destroy_count_);
    }
  }
  absl::StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions*,
      std::vector<ExecutionInput>) override {
    return absl::UnimplementedError("unused in test");
  }

 private:
  std::atomic<int>* destroy_count_;
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
  MOCK_METHOD(void, RecordValidatedCandidate,
              (const InputBuffers& buffers, absl::Duration duration),
              (override));
};

class ConfigRunnerTest : public ::testing::Test {};

TEST_F(ConfigRunnerTest, NullProfilerReturnsInvalidArgument) {
  auto runner = ConfigRunner::Create(nullptr, {});
  EXPECT_THAT(runner, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ConfigRunnerTest, ProfileAllCorrectnessCheckDisabled) {
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::nullopt, 100})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::nullopt, 200})));

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = false;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_2")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(2));
  EXPECT_FALSE(profiles[0].failure.has_value());
  EXPECT_EQ(profiles[0].duration, absl::Seconds(2));
  EXPECT_EQ(profiles[0].scratch_bytes, 100);
  EXPECT_FALSE(profiles[1].failure.has_value());
  EXPECT_EQ(profiles[1].duration, absl::Seconds(1));
  EXPECT_EQ(profiles[1].scratch_bytes, 200);
}

TEST_F(ConfigRunnerTest, ProfileAllExecutionError) {
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(absl::InternalError("execution crash")));

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = false;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(1));
  ASSERT_TRUE(profiles[0].failure.has_value());
  EXPECT_EQ(profiles[0].failure->kind,
            ConfigRunner::FailureKind::kExecutionFailed);
}

TEST_F(ConfigRunnerTest, ProfileAllRedzoneCheckFailed) {
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult(
          {absl::Seconds(1), ScopedShapedBuffer(Shape(), nullptr, 0)})));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillOnce(Return(absl::InternalError("out of bounds buffer write")));

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(1));
  ASSERT_TRUE(profiles[0].failure.has_value());
  EXPECT_EQ(profiles[0].failure->kind,
            ConfigRunner::FailureKind::kRedzoneCheckFailed);
}

TEST_F(ConfigRunnerTest, ProfileAllWarmupRedzoneCheckFailed) {
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(absl::InternalError(
          "Autotuning candidate rejected: kernel wrote past its allocated "
          "buffer. Redzone mismatch in LHS redzone of buffer 0x30fd08000")));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillOnce(Return(absl::OkStatus()));

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(1));
  ASSERT_TRUE(profiles[0].failure.has_value());
  EXPECT_EQ(profiles[0].failure->kind,
            ConfigRunner::FailureKind::kRedzoneCheckFailed);
  EXPECT_EQ(profiles[0].ToProto().failure().kind(),
            AutotuneResult::REDZONE_MODIFIED);
  EXPECT_EQ(profiles[0].ToFailedConfigsProto().kind(),
            autotuner::FailedConfigs::REDZONE_CHECK_FAILED);
}

TEST_F(ConfigRunnerTest, ProfileAllExecutionErrorWithInputRedzoneModified) {
  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _)).WillOnce([] {
    return std::make_unique<InputBuffers>();
  });
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(absl::InternalError("execution crash")));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillOnce(Return(absl::InternalError("input redzone modified")));

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(1));
  ASSERT_TRUE(profiles[0].failure.has_value());
  EXPECT_EQ(profiles[0].failure->kind,
            ConfigRunner::FailureKind::kRedzoneCheckFailed);
}

TEST_F(ConfigRunnerTest, FiltersWrongResultsAgainstTrusted) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer output_1(Shape(), nullptr, 0),
      output_2(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(output_1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(output_2)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("Don't match")));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend trusted_backend;
  MockCodegenBackendWithWrongResults untrusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("wrong_results_config")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(2));
  EXPECT_FALSE(profiles[0].failure.has_value());
  ASSERT_TRUE(profiles[1].failure.has_value());
  EXPECT_EQ(profiles[1].failure->kind,
            ConfigRunner::FailureKind::kWrongResults);
  EXPECT_THAT(profiles[1].failure->message, HasSubstr("Don't match"));
}

TEST_F(ConfigRunnerTest, RecordsOnlyCandidatesThatPassCorrectnessChecks) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer output_1(Shape(), nullptr, 0),
      output_2(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(output_1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(output_2)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("Don't match")));
  // The faster candidate produces wrong results, so it must not become the
  // competitive baseline.
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(2)))
      .Times(1);
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(1)))
      .Times(0);

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend trusted_backend;
  MockCodegenBackendWithWrongResults untrusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("test_config_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("wrong_results_config")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(2));
  EXPECT_FALSE(profiles[0].failure.has_value());
  ASSERT_TRUE(profiles[1].failure.has_value());
}

TEST_F(ConfigRunnerTest, DoesNotRecordCandidatesOfExcludedBackends) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = false;

  auto profiler = std::make_unique<MockProfiler>();
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::nullopt})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::nullopt})));
  // The faster candidate belongs to an excluded backend and cannot be
  // selected, so it must not become the competitive baseline.
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(2)))
      .Times(1);
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(1)))
      .Times(0);

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend excluded_backend;
  EXPECT_CALL(excluded_backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::CUBLASLT_FISSION));
  MockCodegenBackend allowed_backend;
  EXPECT_CALL(allowed_backend, backend())
      .WillRepeatedly(Return(autotuner::Backend::TRITON));

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&excluded_backend, GetTestConfig("excluded_config")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&allowed_backend, GetTestConfig("allowed_config")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(
      auto profiles,
      runner->ProfileAll(std::move(candidates), /*instr=*/nullptr,
                         /*excluded_backends=*/
                         {autotuner::Backend::CUBLASLT_FISSION}));
  ASSERT_THAT(profiles, SizeIs(2));
  EXPECT_FALSE(profiles[0].failure.has_value());
  EXPECT_FALSE(profiles[1].failure.has_value());
}

TEST_F(ConfigRunnerTest,
       DefersRecordingUntrustedCandidatesUntilClusterIsCorroborated) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_a1(Shape(), nullptr, 0), out_b(Shape(), nullptr, 0),
      out_a2(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(3), std::move(out_a1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_b)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(out_a2)})));
  // `out_b` does not match cluster 0 and forms a singleton cluster; `out_a2`
  // matches cluster 0.
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("minority")))
      .WillOnce(Return(absl::OkStatus()));
  // Cluster 0 is corroborated once `out_a2` joins it, so both of its members
  // are reported. The singleton (later demoted) fastest candidate is never
  // reported, so it cannot lower the competitive baseline.
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(3)))
      .Times(1);
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(2)))
      .Times(1);
  EXPECT_CALL(*profiler, RecordValidatedCandidate(_, absl::Seconds(1)))
      .Times(0);

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackendWithWrongResults untrusted_backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_slow")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("minority_fastest")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_fast")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(3));
  EXPECT_FALSE(profiles[0].failure.has_value());
  ASSERT_TRUE(profiles[1].failure.has_value());
  EXPECT_EQ(profiles[1].failure->kind,
            ConfigRunner::FailureKind::kWrongResults);
  EXPECT_FALSE(profiles[2].failure.has_value());
}

TEST_F(ConfigRunnerTest, ClustersOutputsWhenAllBackendsUntrusted) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_a1(Shape(), nullptr, 0), out_a2(Shape(), nullptr, 0),
      out_b(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(ProfileResult({absl::Seconds(3), std::move(out_a1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(out_a2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_b)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("minority")));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackendWithWrongResults untrusted_backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_slow")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_fast")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("minority_fastest")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(3));
  // Majority cluster members (indices 0 and 1) have no failures; minority
  // member (index 2) is demoted.
  EXPECT_FALSE(profiles[0].failure.has_value());
  EXPECT_FALSE(profiles[1].failure.has_value());
  ASSERT_TRUE(profiles[2].failure.has_value());
  EXPECT_EQ(profiles[2].failure->kind,
            ConfigRunner::FailureKind::kWrongResults);
  EXPECT_THAT(profiles[2].failure->message, HasSubstr("minority"));
}

TEST_F(ConfigRunnerTest, TrustedClusterWinsOverLargerUntrustedCluster) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted(Shape(), nullptr, 0),
      out_u1(Shape(), nullptr, 0), out_u2(Shape(), nullptr, 0),
      out_u3(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(10), std::move(out_trusted)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_u3)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("u1 vs c0")))
      .WillOnce(Return(absl::InternalError("u2 vs c0")))
      .WillOnce(Return(absl::InternalError("u3 vs c0")));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend trusted_backend;
  MockCodegenBackendWithWrongResults untrusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("trusted_config")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_1")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_2")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_3")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(4));
  EXPECT_FALSE(profiles[0].failure.has_value());
  EXPECT_TRUE(profiles[1].failure.has_value());
  EXPECT_TRUE(profiles[2].failure.has_value());
  EXPECT_TRUE(profiles[3].failure.has_value());
}

TEST_F(ConfigRunnerTest, PreservesCandidateOrderWhenProfilingTrustedFirst) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted(Shape(), nullptr, 0),
      out_untrusted(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(1), std::move(out_trusted)})))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(1), std::move(out_untrusted)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackendWithWrongResults untrusted_backend;
  MockCodegenBackend trusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  // Untrusted is candidate 0, Trusted is candidate 1.
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_first")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("trusted_second")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(2));
  EXPECT_THAT(*profiles[0].config.backend_config,
              ConfigMatcher("untrusted_first"));
  EXPECT_FALSE(profiles[0].failure.has_value());
  EXPECT_THAT(*profiles[1].config.backend_config,
              ConfigMatcher("trusted_second"));
  EXPECT_FALSE(profiles[1].failure.has_value());
}

TEST_F(ConfigRunnerTest, UntrustedVotesForTrustedCluster) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_trusted_a(Shape(), nullptr, 0),
      out_trusted_b(Shape(), nullptr, 0), out_untrusted_b(Shape(), nullptr, 0),
      out_untrusted_none(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(10), std::move(out_trusted_a)})))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(9), std::move(out_trusted_b)})))
      .WillOnce(
          Return(ProfileResult({absl::Seconds(1), std::move(out_untrusted_b)})))
      .WillOnce(Return(
          ProfileResult({absl::Seconds(1), std::move(out_untrusted_none)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::InternalError("trusted_b vs trusted_a")))
      .WillOnce(Return(absl::InternalError("untrusted_b vs trusted_a")))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("untrusted_none vs trusted_a")))
      .WillOnce(Return(absl::InternalError("untrusted_none vs trusted_b")));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend trusted_backend;
  MockCodegenBackendWithWrongResults untrusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("trusted_a")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("trusted_b")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_matches_b")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("untrusted_matches_none")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(4));
  // trusted_a is demoted because trusted_b cluster has 2 votes.
  ASSERT_TRUE(profiles[0].failure.has_value());
  EXPECT_THAT(profiles[0].failure->message,
              HasSubstr("Output disagrees with winning cluster"));
  EXPECT_FALSE(profiles[1].failure.has_value());
  EXPECT_FALSE(profiles[2].failure.has_value());
  ASSERT_TRUE(profiles[3].failure.has_value());
  EXPECT_THAT(profiles[3].failure->message,
              HasSubstr("Output disagrees with winning cluster"));
  // When cluster 0 is not the winner, no diff report against cluster 0 is
  // appended to avoid misleading mismatch reports.
  EXPECT_THAT(profiles[3].failure->message,
              Not(HasSubstr("untrusted_none vs trusted_a")));
}

TEST_F(ConfigRunnerTest, ClustersUntrustedWhenTrustedReferenceFails) {
  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = true;

  auto profiler = std::make_unique<MockProfiler>();
  ScopedShapedBuffer out_a1(Shape(), nullptr, 0), out_a2(Shape(), nullptr, 0),
      out_b(Shape(), nullptr, 0);
  EXPECT_CALL(*profiler, CreateInputBuffers(_, _))
      .WillOnce(Return(std::make_unique<InputBuffers>()));
  EXPECT_CALL(*profiler, CheckInputBuffers(_))
      .WillRepeatedly(Return(absl::OkStatus()));
  EXPECT_CALL(*profiler, Profile(_, _))
      .WillOnce(Return(absl::InternalError("trusted failed")))
      .WillOnce(Return(ProfileResult({absl::Seconds(3), std::move(out_a1)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(2), std::move(out_a2)})))
      .WillOnce(Return(ProfileResult({absl::Seconds(1), std::move(out_b)})));
  EXPECT_CALL(*profiler, CheckOutputBuffer(_, _, _))
      .WillOnce(Return(absl::OkStatus()))
      .WillOnce(Return(absl::InternalError("minority")));

  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend trusted_backend;
  MockCodegenBackendWithWrongResults untrusted_backend;

  std::vector<ConfigRunner::ExecutableCandidate> candidates;
  candidates.push_back({
      /*config=*/{&trusted_backend, GetTestConfig("trusted_fails")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_slow")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("majority_fast")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });
  candidates.push_back({
      /*config=*/{&untrusted_backend, GetTestConfig("minority_fastest")},
      /*executable=*/std::make_unique<CountingDestructorExecutable>(),
  });

  ASSERT_OK_AND_ASSIGN(auto profiles,
                       runner->ProfileAll(std::move(candidates)));
  ASSERT_THAT(profiles, SizeIs(4));
  EXPECT_TRUE(profiles[0].failure.has_value());
  EXPECT_FALSE(profiles[1].failure.has_value());
  EXPECT_FALSE(profiles[2].failure.has_value());
  EXPECT_TRUE(profiles[3].failure.has_value());
}

TEST_F(ConfigRunnerTest,
       ProfileAllUnloadsCandidatesBeforeReleasingProfilerLock) {
  std::atomic<int> executable_destroy_count{0};
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

  ConfigRunner::CorrectnessCheckOptions options;
  options.enable_correctness_check = false;
  ASSERT_OK_AND_ASSIGN(auto runner,
                       ConfigRunner::Create(std::move(profiler), options));

  MockCodegenBackend backend;
  std::vector<ConfigRunner::ExecutableCandidate> candidates1;
  candidates1.push_back({
      /*config=*/{&backend, GetTestConfig("config_1")},
      /*executable=*/
      std::make_unique<CountingDestructorExecutable>(&executable_destroy_count),
  });
  candidates1.push_back({
      /*config=*/{&backend, GetTestConfig("config_2")},
      /*executable=*/
      std::make_unique<CountingDestructorExecutable>(&executable_destroy_count),
  });

  std::vector<ConfigRunner::ExecutableCandidate> candidates2;
  candidates2.push_back({
      /*config=*/{&backend, GetTestConfig("config_1")},
      /*executable=*/
      std::make_unique<CountingDestructorExecutable>(&executable_destroy_count),
  });
  candidates2.push_back({
      /*config=*/{&backend, GetTestConfig("config_2")},
      /*executable=*/
      std::make_unique<CountingDestructorExecutable>(&executable_destroy_count),
  });

  absl::Status status1;
  absl::Status status2;
  std::unique_ptr<tsl::Thread> t1(
      tsl::Env::Default()->StartThread({}, "runner-test-t1", [&]() {
        auto result = runner->ProfileAll(std::move(candidates1));
        status1 = result.status();
      }));
  std::unique_ptr<tsl::Thread> t2(
      tsl::Env::Default()->StartThread({}, "runner-test-t2", [&]() {
        auto result = runner->ProfileAll(std::move(candidates2));
        status2 = result.status();
      }));
  t1.reset();
  t2.reset();
  ASSERT_OK(status1);
  ASSERT_OK(status2);
}

}  // namespace
}  // namespace xla
