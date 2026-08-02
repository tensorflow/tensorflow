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

#include "xla/hlo/tools/comparison/original_tensor_summary_key_matcher.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/types/span.h"
#include "riegeli/bytes/fd_writer.h"
#include "riegeli/records/record_writer.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/utils/bidirectional_map.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::numerics::comparison {
namespace {
using ::testing::Eq;
using ::testing::Optional;
using ::tsl::testing::StatusIs;
RecoveredTensorSummaryProto CreateSummary(
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
class OriginalTensorSummaryKeyMatcherTest : public ::testing::Test {
 protected:
  void SetUp() override {
    baseline_file_ = tsl::io::JoinPath(testing::TempDir(), "baseline.riegeli");
    target_file_ = tsl::io::JoinPath(testing::TempDir(), "target.riegeli");
    bimap_ = std::make_shared<
        BidirectionalMap<std::string, std::string, std::monostate>>();
  }
  absl::Status CreateMatcher(
      absl::Span<const RecoveredTensorSummaryProto> baseline_summaries,
      absl::Span<const RecoveredTensorSummaryProto> target_summaries) {
    CHECK_OK(WriteSummaries(baseline_file_, baseline_summaries));
    CHECK_OK(WriteSummaries(target_file_, target_summaries));
    ASSIGN_OR_RETURN(matcher_, OriginalTensorSummaryKeyMatcher::Create(
                                   bimap_, baseline_file_, target_file_));
    return absl::OkStatus();
  }
  void AddBimap(const std::string& baseline, const std::string& target) {
    bimap_->Insert(baseline, target, std::monostate());
  }
  std::string baseline_file_;
  std::string target_file_;
  std::shared_ptr<BidirectionalMap<std::string, std::string, std::monostate>>
      bimap_;
  std::shared_ptr<OriginalTensorSummaryKeyMatcher> matcher_;
};
TEST_F(OriginalTensorSummaryKeyMatcherTest, EmptyFiles) {
  ASSERT_OK(CreateMatcher({}, {}));
  EXPECT_EQ(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                      ComparisonVariant::kBaseline),
            std::nullopt);
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, EmptyTargetFile) {
  ASSERT_OK(CreateMatcher(
      {CreateSummary(ScopedTensorKey::FromString("a"), {2, 2})}, {}));
  EXPECT_EQ(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                      ComparisonVariant::kBaseline),
            std::nullopt);
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, SimpleBimapMatch) {
  AddBimap("a", "a_target");
  ASSERT_OK(CreateMatcher(
      {CreateSummary(ScopedTensorKey::FromString("a"), {2, 2})},
      {CreateSummary(ScopedTensorKey::FromString("a_target"), {2, 2})}));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                        ComparisonVariant::kBaseline),
              Optional(Eq(ScopedTensorKey::FromString("a_target"))));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a_target"),
                                        ComparisonVariant::kTarget),
              Optional(Eq(ScopedTensorKey::FromString("a"))));
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, BimapMatchWithScope) {
  AddBimap("scope1", "scope1_target");
  AddBimap("a", "a_target");
  auto baseline_key = ScopedTensorKey::FromString("scope1/a");
  auto target_key = ScopedTensorKey::FromString("scope1_target/a_target");
  ASSERT_OK(CreateMatcher({CreateSummary(baseline_key, {2, 2})},
                          {CreateSummary(target_key, {2, 2})}));
  EXPECT_THAT(
      matcher_->FindMatchingKey(baseline_key, ComparisonVariant::kBaseline),
      Optional(Eq(target_key)));
  EXPECT_THAT(matcher_->FindMatchingKey(target_key, ComparisonVariant::kTarget),
              Optional(Eq(baseline_key)));
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, BimapMatchWithIteration) {
  AddBimap("while1", "while1_target");
  AddBimap("a", "a_target");
  auto baseline_key = ScopedTensorKey::FromString("while1#1/a");
  auto target_key = ScopedTensorKey::FromString("while1_target#1/a_target");
  ASSERT_OK(CreateMatcher({CreateSummary(baseline_key, {2, 2})},
                          {CreateSummary(target_key, {2, 2})}));
  EXPECT_THAT(
      matcher_->FindMatchingKey(baseline_key, ComparisonVariant::kBaseline),
      Optional(Eq(target_key)));
  EXPECT_THAT(matcher_->FindMatchingKey(target_key, ComparisonVariant::kTarget),
              Optional(Eq(baseline_key)));
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, BimapMatchWithShapeIndex) {
  AddBimap("tuple1", "tuple1_target");
  auto baseline_key0 = ScopedTensorKey::FromString("tuple1", {0});
  auto baseline_key1 = ScopedTensorKey::FromString("tuple1", {1});
  auto target_key0 = ScopedTensorKey::FromString("tuple1_target", {0});
  auto target_key1 = ScopedTensorKey::FromString("tuple1_target", {1});
  ASSERT_OK(CreateMatcher({CreateSummary(baseline_key0, {2, 2}),
                           CreateSummary(baseline_key1, {3, 3})},
                          {CreateSummary(target_key0, {2, 2}),
                           CreateSummary(target_key1, {3, 3})}));
  EXPECT_THAT(
      matcher_->FindMatchingKey(baseline_key0, ComparisonVariant::kBaseline),
      Optional(Eq(target_key0)));
  EXPECT_THAT(
      matcher_->FindMatchingKey(baseline_key1, ComparisonVariant::kBaseline),
      Optional(Eq(target_key1)));
}
TEST_F(OriginalTensorSummaryKeyMatcherTest,
       BimapMatchWithShapeIndexDimMismatch) {
  AddBimap("tuple1", "tuple1_target");
  auto baseline_key0 = ScopedTensorKey::FromString("tuple1", {0});
  auto baseline_key1 = ScopedTensorKey::FromString("tuple1", {1});
  auto target_key0 = ScopedTensorKey::FromString("tuple1_target", {0});
  auto target_key1 = ScopedTensorKey::FromString("tuple1_target", {1});
  ASSERT_OK(CreateMatcher({CreateSummary(baseline_key0, {2, 2}),
                           CreateSummary(baseline_key1, {3, 3})},
                          {CreateSummary(target_key0, {2, 2}),
                           CreateSummary(target_key1, {4, 4})}));
  // tuple1{0} should match because dims are same.
  EXPECT_THAT(
      matcher_->FindMatchingKey(baseline_key0, ComparisonVariant::kBaseline),
      Optional(Eq(target_key0)));
  // tuple1{1} should not match because dims are different.
  EXPECT_EQ(
      matcher_->FindMatchingKey(baseline_key1, ComparisonVariant::kBaseline),
      std::nullopt);
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, HeuristicMatch) {
  ASSERT_OK(
      CreateMatcher({CreateSummary(ScopedTensorKey::FromString("a"), {2, 2})},
                    {CreateSummary(ScopedTensorKey::FromString("b"), {2, 2})}));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                        ComparisonVariant::kBaseline),
              Optional(Eq(ScopedTensorKey::FromString("b"))));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("b"),
                                        ComparisonVariant::kTarget),
              Optional(Eq(ScopedTensorKey::FromString("a"))));
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, HeuristicMatchAmbiguous) {
  ASSERT_OK(
      CreateMatcher({CreateSummary(ScopedTensorKey::FromString("a"), {2, 2})},
                    {CreateSummary(ScopedTensorKey::FromString("b"), {2, 2}),
                     CreateSummary(ScopedTensorKey::FromString("c"), {2, 2})}));
  EXPECT_EQ(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                      ComparisonVariant::kBaseline),
            std::nullopt);
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, HeuristicMatchLowSimilarity) {
  ASSERT_OK(
      CreateMatcher({CreateSummary(ScopedTensorKey::FromString("a"), {2, 2})},
                    {CreateSummary(ScopedTensorKey::FromString("b"), {3, 3})}));
  EXPECT_EQ(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                      ComparisonVariant::kBaseline),
            std::nullopt);
}
TEST_F(OriginalTensorSummaryKeyMatcherTest, HeuristicMatchTensorVsScope) {
  auto baseline_key = ScopedTensorKey::FromString("a");
  auto target_key = ScopedTensorKey::FromString("scope1/b");
  ASSERT_OK(CreateMatcher({CreateSummary(baseline_key, {2, 2})},
                          {CreateSummary(target_key, {2, 2})}));
  EXPECT_EQ(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                      ComparisonVariant::kBaseline),
            std::nullopt);
}

TEST_F(OriginalTensorSummaryKeyMatcherTest, CreateWithKeysSimpleBimapMatch) {
  AddBimap("a", "a_target");
  std::vector<AbsoluteScopedTensorKey> baseline_keys = {
      ScopedTensorKey::FromString("a")};
  std::vector<AbsoluteScopedTensorKey> target_keys = {
      ScopedTensorKey::FromString("a_target")};
  ASSERT_OK_AND_ASSIGN(matcher_, OriginalTensorSummaryKeyMatcher::Create(
                                     bimap_, baseline_keys, target_keys));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a"),
                                        ComparisonVariant::kBaseline),
              Optional(Eq(ScopedTensorKey::FromString("a_target"))));
  EXPECT_THAT(matcher_->FindMatchingKey(ScopedTensorKey::FromString("a_target"),
                                        ComparisonVariant::kTarget),
              Optional(Eq(ScopedTensorKey::FromString("a"))));
}

TEST_F(OriginalTensorSummaryKeyMatcherTest,
       CreateFailsWithNonExistentBaseline) {
  ASSERT_OK(WriteSummaries(target_file_, {}));
  EXPECT_THAT(
      OriginalTensorSummaryKeyMatcher::Create(
          bimap_,
          tsl::io::JoinPath(testing::TempDir(), "non_existent_baseline"),
          target_file_),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(OriginalTensorSummaryKeyMatcherTest, CreateFailsWithNonExistentTarget) {
  ASSERT_OK(WriteSummaries(baseline_file_, {}));
  EXPECT_THAT(OriginalTensorSummaryKeyMatcher::Create(
                  bimap_, baseline_file_,
                  tsl::io::JoinPath(testing::TempDir(), "non_existent_target")),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(OriginalTensorSummaryKeyMatcherTest, CreateFailsWithInvalidBaseline) {
  const std::string invalid_baseline_file =
      tsl::io::JoinPath(testing::TempDir(), "invalid_baseline.riegeli");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), invalid_baseline_file,
                                   "this is not riegeli format"));
  ASSERT_OK(WriteSummaries(target_file_, {}));
  EXPECT_THAT(OriginalTensorSummaryKeyMatcher::Create(
                  bimap_, invalid_baseline_file, target_file_),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(OriginalTensorSummaryKeyMatcherTest, CreateFailsWithInvalidTarget) {
  const std::string invalid_target_file =
      tsl::io::JoinPath(testing::TempDir(), "invalid_target.riegeli");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), invalid_target_file,
                                   "this is not riegeli format"));
  ASSERT_OK(WriteSummaries(baseline_file_, {}));
  EXPECT_THAT(OriginalTensorSummaryKeyMatcher::Create(bimap_, baseline_file_,
                                                      invalid_target_file),
              StatusIs(absl::StatusCode::kInvalidArgument));
}
}  // namespace
}  // namespace xla::numerics::comparison
