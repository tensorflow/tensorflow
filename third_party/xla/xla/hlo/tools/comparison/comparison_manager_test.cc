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

#include "xla/hlo/tools/comparison/comparison_manager.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_manager.pb.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::ElementsAre;
using ::testing::Optional;
using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::Partially;
using ::tsl::proto_testing::TreatingNaNsAsEqual;

constexpr auto kBaselineVariant =
    ComparisonOptions::COMPARISON_VARIANT_BASELINE;
constexpr auto kTargetVariant = ComparisonOptions::COMPARISON_VARIANT_TARGET;

class InMemoryComparisonManager : public ComparisonManager {
 public:
  explicit InMemoryComparisonManager(bool log_samples = false,
                                     absl::string_view hlo_module_dump_dir = "")
      : ComparisonManager(log_samples, hlo_module_dump_dir) {}

  absl::Status OnComparisonResult(
      absl::string_view hlo_module_name,
      const TensorComparisonResult& result) override {
    absl::MutexLock lock(mu_);
    results_[hlo_module_name].push_back(result);
    first_comparison_received_notification_.Notify();
    return absl::OkStatus();
  }

  absl::Status OnHloModuleFinished(absl::string_view hlo_module_name,
                                   const ComparisonStats& stats) override {
    absl::MutexLock lock(mu_);
    finished_stats_[hlo_module_name] = stats;
    if (!first_module_finished_notification_.HasBeenNotified()) {
      first_module_finished_notification_.Notify();
    }
    return absl::OkStatus();
  }

  std::vector<TensorComparisonResult> GetComparisonResults(
      absl::string_view hlo_module_name) {
    absl::MutexLock lock(mu_);
    auto it = results_.find(hlo_module_name);
    if (it == results_.end()) {
      return {};
    }
    return it->second;
  }

  std::optional<ComparisonStats> GetComparisonStats(
      absl::string_view hlo_module_name) {
    absl::MutexLock lock(mu_);
    auto it = finished_stats_.find(hlo_module_name);
    if (it == finished_stats_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  bool WaitForComparison(absl::Duration timeout = absl::Seconds(5)) {
    return first_comparison_received_notification_
        .WaitForNotificationWithTimeout(timeout);
  }

  bool WaitForModuleFinish(absl::Duration timeout = absl::Seconds(5)) {
    return first_module_finished_notification_.WaitForNotificationWithTimeout(
        timeout);
  }

 private:
  absl::Mutex mu_;
  absl::flat_hash_map<std::string, std::vector<TensorComparisonResult>> results_
      ABSL_GUARDED_BY(mu_);
  absl::flat_hash_map<std::string, ComparisonStats> finished_stats_
      ABSL_GUARDED_BY(mu_);
  // All test cases where we need such wait has only a single module so we just
  // need a single notification for the first module.
  absl::Notification first_comparison_received_notification_;
  absl::Notification first_module_finished_notification_;
};

class ComparisonManagerTest : public HloHardwareIndependentTestBase {
 protected:
  ComparisonManagerTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  tsl::thread::ThreadPool pool_{tsl::Env::Default(), "test_pool", 2};

  std::unique_ptr<HloModule> CreateSimpleHloModule(
      const absl::string_view name, const absl::string_view op_name = "add") {
    const std::string hlo_text = absl::StrFormat(
        R"(
HloModule %s
ENTRY main {
  param0 = f32[2,2] parameter(0)
  param1 = f32[2,2] parameter(1)
  ROOT %s = f32[2,2] add(param0, param1)
}
)",
        name, op_name);
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnVerifiedModule(hlo_text);
    CHECK_OK(module.status());
    return std::move(*module);
  }
};

TensorSummary CreateTensorSummary(absl::string_view hlo_module_name,
                                  absl::string_view instruction_name,
                                  ComparisonOptions::ComparisonVariant variant,
                                  float value = 1.0f,
                                  absl::Span<const float> samples = {}) {
  TensorSummary summary;
  summary.mutable_metadata()->set_hlo_module_name(hlo_module_name);
  summary.mutable_metadata()->set_comparison_variant(variant);
  TensorPosition* pos = summary.mutable_metadata()->add_original_positions();
  pos->set_instruction_name(instruction_name);

  summary.mutable_shape()->set_element_type(xla::F32);
  summary.mutable_shape()->add_dimensions(2);
  summary.mutable_shape()->add_dimensions(2);

  summary.set_mean(value);
  summary.set_stddev(0.0f);
  summary.set_min(value);
  summary.set_max(value);
  summary.set_checksum(absl::StrCat("checksum_", value));
  summary.mutable_samples()->Add(samples.begin(), samples.end());
  return summary;
}

ModuleStats CreateModuleStats(int num_recorded = 0) {
  ModuleStats stats;
  stats.set_num_tensors_recorded(num_recorded);
  return stats;
}

TEST_F(ComparisonManagerTest, RegisterHloModuleAndReadyWait) {
  tsl::thread::ThreadPool* queue = &pool_;
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("ModuleA", "add_op");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("ModuleA", "add_op_target");

  std::atomic<bool> baseline_registered = false;
  std::atomic<bool> target_registered = false;

  absl::Notification baseline_done, target_done;

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(20));  // Ensure main thread waits.
    ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
    baseline_registered = true;
    baseline_done.Notify();
  });

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(40));  // Ensure main thread waits.
    ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
    target_registered = true;
    target_done.Notify();
  });

  ASSERT_FALSE(baseline_registered.load());
  ASSERT_FALSE(target_registered.load());

  ASSERT_OK(manager.WaitUntilHloModuleIsReady("ModuleA"));
  baseline_done.WaitForNotification();
  target_done.WaitForNotification();

  EXPECT_TRUE(baseline_registered.load());
  EXPECT_TRUE(target_registered.load());
}

TEST_F(ComparisonManagerTest, RegisterHloModuleRunAndWaitUntilAnyRunning) {
  tsl::thread::ThreadPool* queue = &pool_;
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module = CreateSimpleHloModule("ModuleB");
  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("ModuleB"));

  std::atomic<bool> run_registered = false;

  absl::Notification run_done;

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(50));
    ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                           /*logical_device_id=*/0,
                                           /*run_id=*/101, "ModuleB"));
    run_registered = true;
    run_done.Notify();
  });

  ASSERT_FALSE(run_registered.load());
  ASSERT_OK(manager.WaitUntilAnyHloModuleIsRunning());

  run_done.WaitForNotification();

  // Wait for the registration thread to complete its assertion and update.
  // The WaitUntilAnyHloModuleIsRunning might unblock as soon as Notify is
  // called, potentially before run_registered is set to true.
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (!run_registered.load() && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  EXPECT_TRUE(run_registered.load()) << "Timeout waiting for run_registered.";
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareMatching) {
  InMemoryComparisonManager manager(/*log_samples=*/true,  // Test with samples
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("DiffModule", "add.baseline");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("DiffModule", "add.target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("DiffModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/201, "DiffModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/202, "DiffModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "DiffModule", "add.baseline", kBaselineVariant, 1.0f, {1.0f, 1.0f});
  TensorSummary summary_target = CreateTensorSummary(
      "DiffModule", "add.target", kTargetVariant, 1.0f, {1.0f, 1.0f});

  ASSERT_OK(
      manager.RecordTensor(kBaselineVariant, "DiffModule", summary_baseline));
  ASSERT_OK(manager.RecordTensor(kTargetVariant, "DiffModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("DiffModule"),
              ElementsAre(Partially(EqualsProto(R"pb(
                summary_match: true
                target_summary { samples: 1 samples: 1 }
              )pb"))));
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareNanMatches) {
  InMemoryComparisonManager manager(/*log_samples=*/true,  // Test with samples
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("DiffModule", "add.baseline");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("DiffModule", "add.target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("DiffModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/201, "DiffModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/202, "DiffModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "DiffModule", "add.baseline", kBaselineVariant,
      std::numeric_limits<float>::quiet_NaN(), std::vector<float>{1.0f, 1.0f});
  TensorSummary summary_target = CreateTensorSummary(
      "DiffModule", "add.target", kTargetVariant,
      std::numeric_limits<float>::quiet_NaN(), std::vector<float>{1.0f, 1.0f});

  ASSERT_OK(
      manager.RecordTensor(kBaselineVariant, "DiffModule", summary_baseline));
  ASSERT_OK(manager.RecordTensor(kTargetVariant, "DiffModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("DiffModule"),
              ElementsAre(Partially(EqualsProto(R"pb(
                summary_match: true
                target_summary { samples: 1 samples: 1 }
              )pb"))));
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareNanInSamplesMatches) {
  InMemoryComparisonManager manager(/*log_samples=*/true,  // Test with samples
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("DiffModule", "add.baseline");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("DiffModule", "add.target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("DiffModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/201, "DiffModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/202, "DiffModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "DiffModule", "add.baseline", kBaselineVariant, 1,
      std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f});
  TensorSummary summary_target = CreateTensorSummary(
      "DiffModule", "add.target", kTargetVariant, 1,
      std::vector<float>{std::numeric_limits<float>::quiet_NaN(), 1.0f});

  ASSERT_OK(
      manager.RecordTensor(kBaselineVariant, "DiffModule", summary_baseline));
  ASSERT_OK(manager.RecordTensor(kTargetVariant, "DiffModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("DiffModule"),
              ElementsAre(Partially(TreatingNaNsAsEqual(EqualsProto(R"pb(
                shape_match: true
                summary_match: true
                target_summary { samples: nan samples: 1 }
              )pb")))));
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareMismatchingShapes) {
  InMemoryComparisonManager manager(/*log_samples=*/true,  // Test with samples
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("DiffModule", "add.baseline");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("DiffModule", "add.target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("DiffModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/201, "DiffModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/202, "DiffModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "DiffModule", "add.baseline", kBaselineVariant,
      std::numeric_limits<float>::quiet_NaN(), std::vector<float>{1.0f, 1.0f});
  TensorSummary summary_target = CreateTensorSummary(
      "DiffModule", "add.target", kTargetVariant,
      std::numeric_limits<float>::quiet_NaN(), std::vector<float>{1.0f, 1.0f});
  summary_target.mutable_shape()->add_dimensions(3);

  ASSERT_OK(
      manager.RecordTensor(kBaselineVariant, "DiffModule", summary_baseline));
  ASSERT_OK(manager.RecordTensor(kTargetVariant, "DiffModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("DiffModule"),
              ElementsAre(Partially(EqualsProto(R"pb(
                summary_delta: { baseline_summary { samples: 1 samples: 1 } }
                shape_match: false
              )pb"))));
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareMismatchingSamples) {
  InMemoryComparisonManager manager(/*log_samples=*/true,  // Test with samples
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("DiffModule", "add.baseline");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("DiffModule", "add.target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("DiffModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/201, "DiffModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/202, "DiffModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "DiffModule", "add.baseline", kBaselineVariant,
      std::numeric_limits<float>::quiet_NaN(), std::vector<float>{1.0f, 2.0f});
  TensorSummary summary_target =
      CreateTensorSummary("DiffModule", "add.target", kTargetVariant,
                          std::numeric_limits<float>::quiet_NaN(),
                          std::vector<float>{1.0f, 2.0f, 3.0f});

  ASSERT_OK(
      manager.RecordTensor(kBaselineVariant, "DiffModule", summary_baseline));
  ASSERT_OK(manager.RecordTensor(kTargetVariant, "DiffModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("DiffModule"),
              ElementsAre(Partially(TreatingNaNsAsEqual(EqualsProto(R"pb(
                summary_delta: {
                  baseline_summary { samples: 1 samples: 2 }
                  sample_delta_mean: nan
                  sample_delta_stddev: nan
                  sample_delta_max: nan
                }
              )pb")))));
}

TEST_F(ComparisonManagerTest, RecordTensorAndCompareMismatching) {
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("MismatchModule", "op_base");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("MismatchModule", "op_target");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_baseline));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_target));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("MismatchModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/301, "MismatchModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/302, "MismatchModule"));

  TensorSummary summary_baseline =
      CreateTensorSummary("MismatchModule", "op_base", kBaselineVariant, 1.0f);
  TensorSummary summary_target =
      CreateTensorSummary("MismatchModule", "op_target", kTargetVariant, 2.0f);

  ASSERT_OK(manager.RecordTensor(kBaselineVariant, "MismatchModule",
                                 summary_baseline));
  ASSERT_OK(
      manager.RecordTensor(kTargetVariant, "MismatchModule", summary_target));

  ASSERT_TRUE(manager.WaitForComparison()) << "Comparison was not triggered";

  EXPECT_THAT(manager.GetComparisonResults("MismatchModule"),
              ElementsAre(Partially(TreatingNaNsAsEqual(EqualsProto(R"pb(
                summary_delta: {
                  baseline_summary { mean: 1 }
                  sample_delta_mean: nan
                  sample_delta_stddev: nan
                  sample_delta_max: nan
                }
              )pb")))));
}

TEST_F(ComparisonManagerTest, RecordTensorUnmatchedAndFinish) {
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module =
      CreateSimpleHloModule("UnmatchedModule", "op_unmatched");
  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("UnmatchedModule"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant,
                                         /*logical_device_id=*/0,
                                         /*run_id=*/401, "UnmatchedModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant,
                                         /*logical_device_id=*/1,
                                         /*run_id=*/402, "UnmatchedModule"));

  TensorSummary summary_baseline = CreateTensorSummary(
      "UnmatchedModule", "op_unmatched", kBaselineVariant, 1.0f);
  ASSERT_OK(manager.RecordTensor(kBaselineVariant, "UnmatchedModule",
                                 summary_baseline));

  ASSERT_OK(manager.FinishHloModuleRun(kBaselineVariant,
                                       /*logical_device_id=*/0,
                                       /*run_id=*/401, CreateModuleStats(1)));
  ASSERT_OK(manager.FinishHloModuleRun(kTargetVariant, /*logical_device_id=*/1,
                                       /*run_id=*/402, CreateModuleStats(0)));

  EXPECT_THAT(manager.GetComparisonStats("UnmatchedModule"),
              Optional(Partially(EqualsProto(R"pb(
                num_unmatched_baseline_tensors: 1
                target_stats { num_tensors_recorded: 0 }
                baseline_stats { num_tensors_recorded: 1 }
              )pb"))));

  ASSERT_TRUE(manager.WaitForModuleFinish()) << "Module finish not triggered";
}

TEST_F(ComparisonManagerTest, FinishHloModuleRunAndWaitUntilFinished) {
  tsl::thread::ThreadPool* queue = &pool_;
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());
  std::unique_ptr<HloModule> module =
      CreateSimpleHloModule("FinishModule", "op_finish");
  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("FinishModule"));

  uint64_t baseline_run_id = 501;
  uint64_t target_run_id = 502;
  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant, 0, baseline_run_id,
                                         "FinishModule"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant, 1, target_run_id,
                                         "FinishModule"));

  std::atomic<bool> baseline_finished_api_called = false;
  std::atomic<bool> target_finished_api_called = false;

  absl::Notification baseline_done, target_done;

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(50));
    ASSERT_OK(manager.FinishHloModuleRun(kBaselineVariant, 0, baseline_run_id,
                                         CreateModuleStats()));
    baseline_finished_api_called = true;
    baseline_done.Notify();
  });

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(100));
    ASSERT_OK(manager.FinishHloModuleRun(kTargetVariant, 1, target_run_id,
                                         CreateModuleStats()));
    target_finished_api_called = true;
    target_done.Notify();
  });

  ASSERT_FALSE(baseline_finished_api_called.load());
  ASSERT_FALSE(target_finished_api_called.load());

  baseline_done.WaitForNotification();
  target_done.WaitForNotification();

  EXPECT_TRUE(baseline_finished_api_called.load());
  EXPECT_TRUE(target_finished_api_called.load());
  EXPECT_THAT(manager.GetComparisonStats("FinishModule"),
              Optional(Partially(EqualsProto(R"pb(
                target_stats { num_tensors_recorded: 0 }
                baseline_stats { num_tensors_recorded: 0 }
              )pb"))));
}

TEST_F(ComparisonManagerTest, WaitUntilAllModulesAreFinished) {
  tsl::thread::ThreadPool* queue = &pool_;
  InMemoryComparisonManager manager(/*log_samples=*/false,
                                    /*hlo_module_dump_dir=*/testing::TempDir());

  std::unique_ptr<HloModule> module_x = CreateSimpleHloModule("ModuleX");
  std::unique_ptr<HloModule> module_y = CreateSimpleHloModule("ModuleY");

  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_x));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_x));
  ASSERT_OK(manager.RegisterHloModule(kBaselineVariant, *module_y));
  ASSERT_OK(manager.RegisterHloModule(kTargetVariant, *module_y));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("ModuleX"));
  ASSERT_OK(manager.WaitUntilHloModuleIsReady("ModuleY"));

  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant, 0, 601, "ModuleX"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant, 1, 602, "ModuleX"));
  ASSERT_OK(manager.RegisterHloModuleRun(kBaselineVariant, 2, 603, "ModuleY"));
  ASSERT_OK(manager.RegisterHloModuleRun(kTargetVariant, 3, 604, "ModuleY"));

  std::atomic<int> finished_counter = 0;

  absl::Notification baseline_x_done, target_x_done, baseline_y_done,
      target_y_done;

  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(20));
    ASSERT_OK(manager.FinishHloModuleRun(kBaselineVariant, 0, 601,
                                         CreateModuleStats()));
    ++finished_counter;
    baseline_x_done.Notify();
  });
  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(40));
    ASSERT_OK(manager.FinishHloModuleRun(kTargetVariant, 1, 602,
                                         CreateModuleStats()));
    ++finished_counter;
    target_x_done.Notify();
  });
  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(60));
    ASSERT_OK(manager.FinishHloModuleRun(kBaselineVariant, 2, 603,
                                         CreateModuleStats()));
    ++finished_counter;
    baseline_y_done.Notify();
  });
  queue->Schedule([&]() {
    absl::SleepFor(absl::Milliseconds(80));
    ASSERT_OK(manager.FinishHloModuleRun(kTargetVariant, 3, 604,
                                         CreateModuleStats()));
    ++finished_counter;
    target_y_done.Notify();
  });

  ASSERT_OK(manager.WaitUntilAllModulesAreFinished());

  baseline_x_done.WaitForNotification();
  target_x_done.WaitForNotification();
  baseline_y_done.WaitForNotification();
  target_y_done.WaitForNotification();

  EXPECT_EQ(finished_counter.load(), 4);
}

}  // namespace
}  // namespace xla::numerics::comparison
