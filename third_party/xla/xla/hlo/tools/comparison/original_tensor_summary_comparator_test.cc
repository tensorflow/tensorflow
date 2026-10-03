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

#include "xla/hlo/tools/comparison/original_tensor_summary_comparator.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_key_matcher.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/hlo/tools/hlo_diff/utils/bidirectional_map.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {
namespace {

using ::testing::_;
using ::testing::Eq;
using ::testing::IsNull;
using ::testing::MockFunction;
using ::testing::NotNull;

using TensorTransformation = tensor_transformation::TensorTransformation;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;
using FloatSummary = ::xla::comparison::FloatSummary;
using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;

OriginalTensorSummary CreateSummary(float val) {
  return OriginalTensorSummary{
      /*dimensions=*/{1},
      /*summaries=*/{FloatSummary{
          /*block_summaries=*/
          {{/*block_indices=*/{}, /*min=*/val, /*max=*/val, /*count=*/1}},
          /*split_spec=*/{}}}};
}

RecoveredTensorSummaryProto CreateSummaryProto(
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

TEST(OriginalTensorSummaryComparatorTest, BaselineThenTargetInvokesCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));

  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, Eq(baseline_key), _, Eq(target_key), _))
      .WillOnce([] { return absl::OkStatus(); });

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 1);
}

TEST(OriginalTensorSummaryComparatorTest, TargetThenBaselineInvokesCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, Eq(baseline_key), _, Eq(target_key), _))
      .WillOnce([] { return absl::OkStatus(); });

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 1);
}

TEST(OriginalTensorSummaryComparatorTest, OnlyBaselineDoesNotInvokeCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);
}

TEST(OriginalTensorSummaryComparatorTest, OnlyTargetDoesNotInvokeCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);
}

TEST(OriginalTensorSummaryComparatorTest, NoCallbackIfInstructionNotInBimap) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.2"));
  ASSERT_OK_AND_ASSIGN(
      auto key_matcher,
      OriginalTensorSummaryKeyMatcher::Create(bimap_ptr, {baseline_key}, {}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);
}

TEST(OriginalTensorSummaryComparatorTest, ScopedInstructionsAreTranslated) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.1", "scope.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.1")});
  AbsoluteScopedTensorKey target_key = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"), {ScopeInstruction::Create("scope.1.t")});
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));

  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, Eq(baseline_key), _, Eq(target_key), _))
      .WillOnce([] { return absl::OkStatus(); });

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 1);
}

TEST(OriginalTensorSummaryComparatorTest, WildcardMatching) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.1", "scope.1.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.2", "scope.2.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.3", "scope.3.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));

  // b_s1_wild matches t_s1_1 and t_s1_2
  AbsoluteScopedTensorKey b_s1_wild = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.1", -1)});
  AbsoluteScopedTensorKey t_s1_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.1.t", 1)});
  AbsoluteScopedTensorKey t_s1_2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.1.t", 2)});
  // t_s2_wild matches b_s2_1 and b_s2_2
  AbsoluteScopedTensorKey t_s2_wild = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.2.t", -1)});
  AbsoluteScopedTensorKey b_s2_1 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.2", 1)});
  AbsoluteScopedTensorKey b_s2_2 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.2", 2)});
  // b_s3_wild matches t_s3_wild
  AbsoluteScopedTensorKey b_s3_wild = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.3", -1)});
  AbsoluteScopedTensorKey t_s3_wild = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.3.t", -1)});

  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {b_s1_wild, b_s2_1, b_s2_2, b_s3_wild},
                           {t_s1_1, t_s1_2, t_s2_wild, t_s3_wild}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, b_s1_wild, nullptr, CreateSummary(1)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s1_wild), _, Eq(t_s1_1), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s1_1, nullptr, CreateSummary(2)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s1_wild), _, Eq(t_s1_2), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s1_2, nullptr, CreateSummary(3)));

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s2_wild, nullptr, CreateSummary(4)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s2_1), _, Eq(t_s2_wild), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, b_s2_1, nullptr, CreateSummary(5)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s2_2), _, Eq(t_s2_wild), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, b_s2_2, nullptr, CreateSummary(6)));

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, b_s3_wild, nullptr, CreateSummary(7)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s3_wild), _, Eq(t_s3_wild), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s3_wild, nullptr, CreateSummary(8)));
  // Wildcard pending comparisons should not be processed in FinishComparison.
  EXPECT_CALL(mock_callback, Call(_, _, IsNull(), _, NotNull())).Times(0);
  EXPECT_CALL(mock_callback, Call(_, _, NotNull(), _, IsNull())).Times(0);
  EXPECT_OK(comparator.FinishComparison());
}

TEST(OriginalTensorSummaryComparatorTest, NestedWildcardMatching) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.1", "scope.1.t", std::monostate{});
  hlo_diff_bimap.Insert("scope.2", "scope.2.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));

  // baseline scope.1#3/scope.2#* should match target scope.1.t#3/scope.2.t#5
  AbsoluteScopedTensorKey b_s1_3_s2_wild = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.1", 3),
                                    ScopeInstruction::Create("scope.2", -1)});
  AbsoluteScopedTensorKey t_s1_3_s2_5 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.1.t", 3),
       ScopeInstruction::Create("scope.2.t", 5)});
  // baseline scope.1#3/scope.2#* should NOT match target
  // scope.1.t#2/scope.2.t#5
  AbsoluteScopedTensorKey t_s1_2_s2_5 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1.t"),
      {ScopeInstruction::Create("scope.1.t", 2),
       ScopeInstruction::Create("scope.2.t", 5)});
  AbsoluteScopedTensorKey b_s1_2_s2_5 = AbsoluteScopedTensorKey::Create(
      TensorKey::Create("inst.1"), {ScopeInstruction::Create("scope.1", 2),
                                    ScopeInstruction::Create("scope.2", 5)});

  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {b_s1_3_s2_wild, b_s1_2_s2_5},
                           {t_s1_3_s2_5, t_s1_2_s2_5}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, b_s1_3_s2_wild, nullptr, CreateSummary(1)));
  EXPECT_CALL(mock_callback, Call(_, Eq(b_s1_3_s2_wild), _, Eq(t_s1_3_s2_5), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s1_3_s2_5, nullptr, CreateSummary(2)));

  EXPECT_CALL(mock_callback, Call(_, Eq(b_s1_3_s2_wild), _, Eq(t_s1_2_s2_5), _))
      .Times(0);
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, t_s1_2_s2_5, nullptr, CreateSummary(3)));

  EXPECT_CALL(mock_callback,
              Call(_, Eq(b_s1_2_s2_5), IsNull(), Eq(t_s1_2_s2_5), NotNull()))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.FinishComparison());
}

TEST(OriginalTensorSummaryComparatorTest,
     GetCommonContinuationWithDifferentObjectIdentity) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  using tensor_transformation::Broadcast;
  using tensor_transformation::Reshape;

  auto common_continuation1 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{4}});
  auto common_continuation2 = std::make_shared<const TensorTransformation>(
      Reshape{/*continuation=*/nullptr, /*output_dimensions=*/{4}});
  ASSERT_NE(common_continuation1, common_continuation2);
  ASSERT_EQ(*common_continuation1, *common_continuation2);

  auto baseline_transformation =
      std::make_shared<const TensorTransformation>(Reshape{
          /*continuation=*/common_continuation1, /*output_dimensions=*/{2, 2}});
  auto target_transformation = std::make_shared<const TensorTransformation>(
      Broadcast{/*continuation=*/common_continuation2,
                /*output_dimensions=*/{2, 2, 1},
                /*broadcast_dimensions=*/{0, 1}});

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key, baseline_transformation,
      CreateSummary(1)));

  EXPECT_CALL(mock_callback, Call(testing::Pointee(Eq(*common_continuation1)),
                                  Eq(baseline_key), _, Eq(target_key), _))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key, target_transformation,
      CreateSummary(2)));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 1);
}

TEST(OriginalTensorSummaryComparatorTest,
     FinishComparisonWithPendingBaselineInvokesCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);

  EXPECT_CALL(mock_callback,
              Call(_, Eq(baseline_key), NotNull(), Eq(target_key), IsNull()))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.FinishComparison());
}

TEST(OriginalTensorSummaryComparatorTest,
     FinishComparisonWithPendingTargetInvokesCallback) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  hlo_diff_bimap.Insert("inst.1", "inst.1.t", std::monostate{});

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  auto bimap_ptr = std::make_shared<
      const BidirectionalMap<std::string, std::string, std::monostate>>(
      std::move(hlo_diff_bimap));
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1"));
  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("inst.1.t"));
  ASSERT_OK_AND_ASSIGN(auto key_matcher,
                       OriginalTensorSummaryKeyMatcher::Create(
                           bimap_ptr, {baseline_key}, {target_key}));
  OriginalTensorSummaryComparator comparator(std::move(key_matcher),
                                             mock_callback.AsStdFunction());

  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator.ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator.GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);

  EXPECT_CALL(mock_callback,
              Call(_, Eq(baseline_key), IsNull(), Eq(target_key), NotNull()))
      .WillOnce([] { return absl::OkStatus(); });
  EXPECT_OK(comparator.FinishComparison());
}

class OriginalTensorSummaryComparatorWithHloTest
    : public HloHardwareIndependentTestBase {
 protected:
  OriginalTensorSummaryComparatorWithHloTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {
    baseline_file_ = tsl::io::JoinPath(testing::TempDir(), "baseline.riegeli");
    target_file_ = tsl::io::JoinPath(testing::TempDir(), "target.riegeli");
  }

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

  std::string baseline_file_;
  std::string target_file_;
};

TEST_F(OriginalTensorSummaryComparatorWithHloTest, CreateComparator) {
  std::unique_ptr<HloModule> module_baseline =
      CreateSimpleHloModule("ModuleA", "add_op");
  std::unique_ptr<HloModule> module_target =
      CreateSimpleHloModule("ModuleA", "add_op_target");

  auto p0 = AbsoluteScopedTensorKey::Create(TensorKey::Create("param0"));
  auto p1 = AbsoluteScopedTensorKey::Create(TensorKey::Create("param1"));
  auto add_op = AbsoluteScopedTensorKey::Create(TensorKey::Create("add_op"));
  auto add_op_target =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_op_target"));

  ASSERT_OK(WriteSummaries(
      baseline_file_,
      {CreateSummaryProto(p0, {2, 2}), CreateSummaryProto(p1, {2, 2}),
       CreateSummaryProto(add_op, {2, 2})}));
  ASSERT_OK(WriteSummaries(
      target_file_,
      {CreateSummaryProto(p0, {2, 2}), CreateSummaryProto(p1, {2, 2}),
       CreateSummaryProto(add_op_target, {2, 2})}));

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  ASSERT_OK_AND_ASSIGN(
      auto comparator_and_metrics,
      OriginalTensorSummaryComparator::Create(
          module_baseline.get(), module_target.get(), baseline_file_,
          target_file_, mock_callback.AsStdFunction()));
  auto& [comparator, creation_metrics, diff_results] = comparator_and_metrics;
  EXPECT_EQ(creation_metrics.baseline_tensor_count, 3);
  EXPECT_EQ(creation_metrics.target_tensor_count, 3);
  EXPECT_EQ(creation_metrics.unchanged_tensor_pair_count, 3);
  EXPECT_EQ(creation_metrics.changed_tensor_pair_count, 0);

  // The 'add_op' in baseline should be mapped to 'add_op_target' in target.
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_op"));
  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_OK(comparator->ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));

  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("add_op_target"));
  OriginalTensorSummary target_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/1,
                                                        /*max=*/4,
                                                        /*mean=*/2.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, Eq(baseline_key), _, Eq(target_key), _))
      .WillOnce([] { return absl::OkStatus(); });

  EXPECT_OK(comparator->ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 0);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 1);
}

TEST_F(OriginalTensorSummaryComparatorWithHloTest, IgnoreDifferentShapes) {
  const std::string hlo_baseline = R"(
HloModule m1
ENTRY main {
  param0 = f32[4]{0} parameter(0)
  ROOT out = f32[2,2]{1,0} reshape(f32[4]{0} %param0)
}
)";
  const std::string hlo_target = R"(
HloModule m2
ENTRY main {
  param0 = f32[4]{0} parameter(0)
  ROOT out = f32[4,1]{1,0} reshape(f32[4]{0} %param0)
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_baseline,
                       ParseAndReturnVerifiedModule(hlo_baseline));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module_target,
                       ParseAndReturnVerifiedModule(hlo_target));

  auto p0 = AbsoluteScopedTensorKey::Create(TensorKey::Create("param0"));
  auto out = AbsoluteScopedTensorKey::Create(TensorKey::Create("out"));
  ASSERT_OK(WriteSummaries(baseline_file_, {CreateSummaryProto(p0, {4}),
                                            CreateSummaryProto(out, {2, 2})}));
  ASSERT_OK(WriteSummaries(target_file_, {CreateSummaryProto(p0, {4}),
                                          CreateSummaryProto(out, {4, 1})}));

  MockFunction<absl::Status(
      std::shared_ptr<const TensorTransformation> pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>
      mock_callback;

  ASSERT_OK_AND_ASSIGN(
      auto comparator_and_metrics,
      OriginalTensorSummaryComparator::Create(
          module_baseline.get(), module_target.get(), baseline_file_,
          target_file_, mock_callback.AsStdFunction()));
  auto& [comparator, creation_metrics, diff_results] = comparator_and_metrics;

  // We expect 'out' to be a changed instruction, but with incompatible shapes,
  // so it shouldn't be added to bimap used for comparison.
  EXPECT_EQ(creation_metrics.baseline_tensor_count, 2);
  EXPECT_EQ(creation_metrics.target_tensor_count, 2);
  EXPECT_EQ(creation_metrics.unchanged_tensor_pair_count, 1);
  EXPECT_EQ(creation_metrics.changed_tensor_pair_count, 1);

  // The 'out' in baseline should not be mapped to 'out' in target because of
  // shape mismatch.
  AbsoluteScopedTensorKey baseline_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("out"));
  OriginalTensorSummary baseline_summary{
      /*dimensions=*/{2, 2},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_OK(comparator->ProcessOriginalTensorSummary(
      ComparisonVariant::kBaseline, baseline_key,
      /*pending_transformation=*/nullptr, baseline_summary));

  AbsoluteScopedTensorKey target_key =
      AbsoluteScopedTensorKey::Create(TensorKey::Create("out"));
  OriginalTensorSummary target_summary{
      /*dimensions=*/{4, 1},
      /*summaries=*/{FloatSummary{/*block_summaries=*/{{/*block_indices=*/{},
                                                        /*min=*/0,
                                                        /*max=*/3,
                                                        /*mean=*/1.5,
                                                        /*stddev=*/1,
                                                        /*count=*/4}},
                                  /*split_spec=*/{}}}};

  EXPECT_CALL(mock_callback, Call(_, _, _, _, _)).Times(0);

  EXPECT_OK(comparator->ProcessOriginalTensorSummary(
      ComparisonVariant::kTarget, target_key,
      /*pending_transformation=*/nullptr, target_summary));
  auto processing_metrics = comparator->GetProcessingMetrics();
  EXPECT_EQ(processing_metrics.received_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.received_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_baseline_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.untranslatable_target_tensor_summaries, 1);
  EXPECT_EQ(processing_metrics.compared_pairs_count, 0);
}

}  // namespace
}  // namespace xla::numerics::comparison
