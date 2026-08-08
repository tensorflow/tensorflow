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

#include "xla/hlo/tools/comparison/xla_job_comparator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::Eq;
using ::testing::SizeIs;

using FloatSummary = ::xla::comparison::FloatSummary;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;

struct CallbackResult {
  int replica_id;
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      pending_transformation;
  AbsoluteScopedTensorKey baseline_tensor_key;
  std::optional<OriginalTensorSummary> baseline_tensor_summary;
  AbsoluteScopedTensorKey target_tensor_key;
  std::optional<OriginalTensorSummary> target_tensor_summary;
};

RecoveredTensorSummaryProto CreateRecoveredTensorSummaryProto(
    const AbsoluteScopedTensorKey& key, absl::Span<const int64_t> dimensions) {
  RecoveredTensorSummaryProto proto;
  *proto.mutable_tensor_key() = key.ToProto();
  proto.mutable_original_tensor_summary()->mutable_dimensions()->Assign(
      dimensions.begin(), dimensions.end());
  return proto;
}

absl::Status WriteSummaries(
    absl::string_view filename,
    absl::Span<const RecoveredTensorSummaryProto> summaries) {
  riegeli::RecordWriter writer{riegeli::FdWriter(filename)};
  for (const auto& summary : summaries) {
    if (!writer.WriteRecord(summary)) {
      return writer.status();
    }
  }
  if (!writer.Close()) {
    return writer.status();
  }
  return absl::OkStatus();
}

// Helper to create a FloatSummary.
FloatSummary CreateSummary(float value) {
  return FloatSummary{
      /*block_summaries=*/{{/*block_indices=*/{},
                            /*min=*/value,
                            /*max=*/value,
                            /*mean=*/value,
                            /*stddev=*/0,
                            /*count=*/1}},
      /*split_spec=*/{},
  };
}

// Helper to create an OriginalTensorSummary.
OriginalTensorSummary CreateOriginalTensorSummary(
    float value, const std::vector<int64_t>& dims) {
  return OriginalTensorSummary{
      /*dimensions=*/dims,
      /*summaries=*/{CreateSummary(value)},
  };
}

class XlaJobComparatorTest : public ::xla::HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    callback_results_.clear();
    baseline_file_ = tsl::io::JoinPath(testing::TempDir(), "baseline.riegeli");
    target_file_ = tsl::io::JoinPath(testing::TempDir(), "target.riegeli");
  }

  void WriteDefaultRecoveredSummaries() {
    ASSERT_OK(WriteSummaries(
        baseline_file_,
        {CreateRecoveredTensorSummaryProto(
            AbsoluteScopedTensorKey::Create(TensorKey::Create("add_baseline")),
            {4, 4})}));
    ASSERT_OK(WriteSummaries(
        target_file_,
        {CreateRecoveredTensorSummaryProto(
            AbsoluteScopedTensorKey::Create(TensorKey::Create("add_target")),
            {4, 4})}));
  }

  std::unique_ptr<HloModule> CreateOriginalModule(absl::string_view name,
                                                  absl::string_view root_name) {
    const std::string hlo_text = absl::StrFormat(
        R"(
HloModule %s
ENTRY main {
  %%param0 = f32[4,4]{1,0} parameter(0)
  %%param1 = f32[4,4]{1,0} parameter(1)
  ROOT %s = f32[4,4]{1,0} add(%%param0, %%param1)
}
)",
        name, root_name);
    absl::StatusOr<std::unique_ptr<HloModule>> module =
        ParseAndReturnVerifiedModule(hlo_text);
    CHECK_OK(module.status());
    return std::move(*module);
  }

  std::vector<CallbackResult> callback_results_;
  std::string baseline_file_;
  std::string target_file_;
  XlaJobComparator::XlaJobComparatorCallback callback_ =
      [&](int replica_id,
          std::shared_ptr<const tensor_transformation::TensorTransformation>
              pending,
          AbsoluteScopedTensorKey baseline_key,
          OriginalTensorSummary const* baseline_summary,
          AbsoluteScopedTensorKey target_key,
          OriginalTensorSummary const* target_summary) {
        callback_results_.push_back(
            {replica_id, std::move(pending), baseline_key,
             baseline_summary ? std::make_optional(*baseline_summary)
                              : std::nullopt,
             target_key,
             target_summary ? std::make_optional(*target_summary)
                            : std::nullopt});
        return absl::OkStatus();
      };
};

TEST_F(XlaJobComparatorTest, OneReplicaTwoComputations) {
  std::unique_ptr<HloModule> baseline_original =
      CreateOriginalModule("baseline_original", "add_baseline");
  std::unique_ptr<HloModule> target_original =
      CreateOriginalModule("target_original", "add_target");

  int replica_count = 1;
  ASSERT_NO_FATAL_FAILURE(WriteDefaultRecoveredSummaries());
  ASSERT_OK_AND_ASSIGN(
      auto create_result,
      XlaJobComparator::Create(replica_count, baseline_original.get(),
                               target_original.get(), baseline_file_,
                               target_file_, std::move(callback_)));
  auto [comparator, creation_metrics, diff_results] = std::move(create_result);

  EXPECT_GT(creation_metrics.comparator_metrics.unchanged_tensor_pair_count, 0);

  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_baseline"));
  OriginalTensorSummary baseline_summary =
      CreateOriginalTensorSummary(1.0f, {4, 4});
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, 0, baseline_key, nullptr,
      baseline_summary));

  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_target"));
  OriginalTensorSummary target_summary =
      CreateOriginalTensorSummary(3.0f, {4, 4});
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, 0, target_key, nullptr, target_summary));

  ASSERT_OK(comparator.FinishComparison());

  ASSERT_THAT(callback_results_, SizeIs(1));
  EXPECT_EQ(callback_results_[0].replica_id, 0);
  EXPECT_EQ(
      callback_results_[0].baseline_tensor_key.tensor_key.instruction_name,
      "add_baseline");
  EXPECT_EQ(callback_results_[0].target_tensor_key.tensor_key.instruction_name,
            "add_target");
  ASSERT_TRUE(callback_results_[0].baseline_tensor_summary.has_value());
  EXPECT_THAT(callback_results_[0].baseline_tensor_summary->dimensions,
              Eq(std::vector<int64_t>{4, 4}));
  ASSERT_THAT(callback_results_[0].baseline_tensor_summary->summaries,
              SizeIs(1));
  ASSERT_TRUE(callback_results_[0].target_tensor_summary.has_value());
  EXPECT_THAT(callback_results_[0].target_tensor_summary->dimensions,
              Eq(std::vector<int64_t>{4, 4}));
  ASSERT_THAT(callback_results_[0].target_tensor_summary->summaries, SizeIs(1));
  // Values are merged by min/max across shards.
  ASSERT_THAT(callback_results_[0]
                  .baseline_tensor_summary->summaries[0]
                  .block_summaries,
              SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            1.0f);
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            1.0f);
  ASSERT_THAT(
      callback_results_[0].target_tensor_summary->summaries[0].block_summaries,
      SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            3.0f);
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            3.0f);

  std::vector<XlaJobComparator::ProcessingMetrics> processing_metrics =
      comparator.GetProcessingMetrics();
  ASSERT_THAT(processing_metrics, SizeIs(1));
  EXPECT_EQ(processing_metrics[0]
                .comparator_metrics.received_baseline_tensor_summaries,
            1);
  EXPECT_EQ(
      processing_metrics[0].comparator_metrics.received_target_tensor_summaries,
      1);
  EXPECT_EQ(processing_metrics[0].comparator_metrics.compared_pairs_count, 1);
}

TEST_F(XlaJobComparatorTest, TwoReplicasTwoComputations) {
  std::unique_ptr<HloModule> baseline_original =
      CreateOriginalModule("baseline_original", "add_baseline");
  std::unique_ptr<HloModule> target_original =
      CreateOriginalModule("target_original", "add_target");

  int replica_count = 2;
  ASSERT_NO_FATAL_FAILURE(WriteDefaultRecoveredSummaries());
  ASSERT_OK_AND_ASSIGN(
      auto create_result,
      XlaJobComparator::Create(replica_count, baseline_original.get(),
                               target_original.get(), baseline_file_,
                               target_file_, std::move(callback_)));
  auto [comparator, creation_metrics, diff_results] = std::move(create_result);

  EXPECT_GT(creation_metrics.comparator_metrics.unchanged_tensor_pair_count, 0);

  // Replica 0
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, 0,
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_baseline")),
      nullptr, CreateOriginalTensorSummary(1.0f, {4, 4})));
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, 0,
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_target")), nullptr,
      CreateOriginalTensorSummary(3.0f, {4, 4})));

  // Replica 1
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, 1,
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_baseline")),
      nullptr, CreateOriginalTensorSummary(5.0f, {4, 4})));
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, 1,
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_target")), nullptr,
      CreateOriginalTensorSummary(7.0f, {4, 4})));

  ASSERT_OK(comparator.FinishComparison());

  ASSERT_THAT(callback_results_, SizeIs(2));
  // Sort results by replica_id to make test deterministic.
  std::sort(
      callback_results_.begin(), callback_results_.end(),
      [](const auto& a, const auto& b) { return a.replica_id < b.replica_id; });

  EXPECT_EQ(callback_results_[0].replica_id, 0);
  EXPECT_EQ(
      callback_results_[0].baseline_tensor_key.tensor_key.instruction_name,
      "add_baseline");
  EXPECT_EQ(callback_results_[0].target_tensor_key.tensor_key.instruction_name,
            "add_target");
  ASSERT_TRUE(callback_results_[0].baseline_tensor_summary.has_value());
  ASSERT_THAT(callback_results_[0].baseline_tensor_summary->summaries,
              SizeIs(1));
  ASSERT_THAT(callback_results_[0]
                  .baseline_tensor_summary->summaries[0]
                  .block_summaries,
              SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            1.0f);
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            1.0f);
  ASSERT_TRUE(callback_results_[0].target_tensor_summary.has_value());
  ASSERT_THAT(callback_results_[0].target_tensor_summary->summaries, SizeIs(1));
  ASSERT_THAT(
      callback_results_[0].target_tensor_summary->summaries[0].block_summaries,
      SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            3.0f);
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            3.0f);

  EXPECT_EQ(callback_results_[1].replica_id, 1);
  EXPECT_EQ(
      callback_results_[1].baseline_tensor_key.tensor_key.instruction_name,
      "add_baseline");
  EXPECT_EQ(callback_results_[1].target_tensor_key.tensor_key.instruction_name,
            "add_target");
  ASSERT_TRUE(callback_results_[1].baseline_tensor_summary.has_value());
  ASSERT_THAT(callback_results_[1].baseline_tensor_summary->summaries,
              SizeIs(1));
  ASSERT_THAT(callback_results_[1]
                  .baseline_tensor_summary->summaries[0]
                  .block_summaries,
              SizeIs(1));
  EXPECT_EQ(callback_results_[1]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            5.0f);
  EXPECT_EQ(callback_results_[1]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            5.0f);
  ASSERT_TRUE(callback_results_[1].target_tensor_summary.has_value());
  ASSERT_THAT(callback_results_[1].target_tensor_summary->summaries, SizeIs(1));
  ASSERT_THAT(
      callback_results_[1].target_tensor_summary->summaries[0].block_summaries,
      SizeIs(1));
  EXPECT_EQ(callback_results_[1]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            7.0f);
  EXPECT_EQ(callback_results_[1]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            7.0f);

  std::vector<XlaJobComparator::ProcessingMetrics> processing_metrics =
      comparator.GetProcessingMetrics();
  ASSERT_THAT(processing_metrics, SizeIs(2));
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(processing_metrics[i]
                  .comparator_metrics.received_baseline_tensor_summaries,
              1);
    EXPECT_EQ(processing_metrics[i]
                  .comparator_metrics.received_target_tensor_summaries,
              1);
    EXPECT_EQ(processing_metrics[i].comparator_metrics.compared_pairs_count, 1);
  }
}

TEST_F(XlaJobComparatorTest, FinishComparisonWithPendingBaselineSummary) {
  std::unique_ptr<HloModule> baseline_original =
      CreateOriginalModule("baseline_original", "add_baseline");
  std::unique_ptr<HloModule> target_original =
      CreateOriginalModule("target_original", "add_target");

  int replica_count = 1;
  ASSERT_NO_FATAL_FAILURE(WriteDefaultRecoveredSummaries());
  ASSERT_OK_AND_ASSIGN(
      auto create_result,
      XlaJobComparator::Create(replica_count, baseline_original.get(),
                               target_original.get(), baseline_file_,
                               target_file_, std::move(callback_)));
  auto [comparator, creation_metrics, diff_results] = std::move(create_result);

  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, 0,
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_baseline")),
      nullptr, CreateOriginalTensorSummary(1.0f, {4, 4})));

  ASSERT_THAT(callback_results_, SizeIs(0));

  ASSERT_OK(comparator.FinishComparison());

  ASSERT_THAT(callback_results_, SizeIs(1));
  EXPECT_EQ(callback_results_[0].replica_id, 0);
  EXPECT_EQ(
      callback_results_[0].baseline_tensor_key.tensor_key.instruction_name,
      "add_baseline");
  EXPECT_EQ(callback_results_[0].target_tensor_key.tensor_key.instruction_name,
            "add_target");
  ASSERT_TRUE(callback_results_[0].baseline_tensor_summary.has_value());
  EXPECT_FALSE(callback_results_[0].target_tensor_summary.has_value());
  EXPECT_THAT(callback_results_[0].baseline_tensor_summary->dimensions,
              Eq(std::vector<int64_t>{4, 4}));
  ASSERT_THAT(callback_results_[0].baseline_tensor_summary->summaries,
              SizeIs(1));
  ASSERT_THAT(callback_results_[0]
                  .baseline_tensor_summary->summaries[0]
                  .block_summaries,
              SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            1.0f);
  EXPECT_EQ(callback_results_[0]
                .baseline_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            1.0f);
}

TEST_F(XlaJobComparatorTest, FinishComparisonWithPendingTargetSummary) {
  std::unique_ptr<HloModule> baseline_original =
      CreateOriginalModule("baseline_original", "add_baseline");
  std::unique_ptr<HloModule> target_original =
      CreateOriginalModule("target_original", "add_target");

  int replica_count = 1;
  ASSERT_NO_FATAL_FAILURE(WriteDefaultRecoveredSummaries());
  ASSERT_OK_AND_ASSIGN(
      auto create_result,
      XlaJobComparator::Create(replica_count, baseline_original.get(),
                               target_original.get(), baseline_file_,
                               target_file_, std::move(callback_)));
  auto [comparator, creation_metrics, diff_results] = std::move(create_result);

  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_target"));
  ASSERT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, 0, target_key, nullptr,
      CreateOriginalTensorSummary(3.0f, {4, 4})));

  ASSERT_THAT(callback_results_, SizeIs(0));

  ASSERT_OK(comparator.FinishComparison());

  ASSERT_THAT(callback_results_, SizeIs(1));
  EXPECT_EQ(callback_results_[0].replica_id, 0);
  EXPECT_EQ(
      callback_results_[0].baseline_tensor_key.tensor_key.instruction_name,
      "add_baseline");
  EXPECT_EQ(callback_results_[0].target_tensor_key.tensor_key.instruction_name,
            "add_target");
  EXPECT_FALSE(callback_results_[0].baseline_tensor_summary.has_value());
  ASSERT_TRUE(callback_results_[0].target_tensor_summary.has_value());
  EXPECT_THAT(callback_results_[0].target_tensor_summary->dimensions,
              Eq(std::vector<int64_t>{4, 4}));
  ASSERT_THAT(callback_results_[0].target_tensor_summary->summaries, SizeIs(1));
  ASSERT_THAT(
      callback_results_[0].target_tensor_summary->summaries[0].block_summaries,
      SizeIs(1));
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .min,
            3.0f);
  EXPECT_EQ(callback_results_[0]
                .target_tensor_summary->summaries[0]
                .block_summaries[0]
                .max,
            3.0f);
}

}  // namespace
}  // namespace xla::numerics::comparison
