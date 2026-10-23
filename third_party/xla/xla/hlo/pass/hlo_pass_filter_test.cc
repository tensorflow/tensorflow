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

#include "xla/hlo/pass/hlo_pass_filter.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_ptr_field.h"

namespace xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;

TEST(HloPassFilterTest, ValidateSyntaxValid) {
  EXPECT_THAT(HloPassFilter::ValidateEntry(""), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp:0"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp:2"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("simplification/algsimp"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("simplification/algsimp:2"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("@0"), IsOk());
  EXPECT_THAT(HloPassFilter::ValidateEntry("@42"), IsOk());
}

TEST(HloPassFilterTest, ValidateSyntaxInvalid) {
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp:"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp:abc"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(HloPassFilter::ValidateEntry("algsimp:1:2"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(HloPassFilter::ValidateEntry("@"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(HloPassFilter::ValidateEntry("@abc"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

HloPassFilter::InvocationInfo MakeInvocation(absl::string_view pass_name,
                                             absl::string_view pipeline_name,
                                             int64_t pass_id,
                                             int64_t global_occurrence,
                                             int64_t pipeline_occurrence) {
  return HloPassFilter::InvocationInfo{pass_name, pipeline_name, pass_id,
                                       global_occurrence, pipeline_occurrence};
}

TEST(HloPassFilterTest, MatchesPlainName) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("algsimp"));
  EXPECT_TRUE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 5, 3, 1)));
  EXPECT_FALSE(
      filter.Matches(MakeInvocation("dce", "simplification", 5, 3, 1)));
  // A plain name also matches a pass whose parent pipeline has that name.
  EXPECT_TRUE(filter.Matches(MakeInvocation("dce", "algsimp", 5, 3, 1)));
}

TEST(HloPassFilterTest, ScopedNameDoesNotMatchPipelineName) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("opt/algsimp"));
  // Scoped spec does NOT fall back to pipeline_name matching.
  EXPECT_FALSE(filter.Matches(MakeInvocation("dce", "algsimp", 5, 3, 1)));
}

TEST(HloPassFilterTest, OccurrenceDoesNotMatchPipelineName) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("algsimp:0"));
  // Occurrence spec does NOT fall back to pipeline_name matching.
  EXPECT_FALSE(filter.Matches(MakeInvocation("dce", "algsimp", 5, 0, 0)));
}

TEST(HloPassFilterTest, MatchesGlobalOccurrence) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("algsimp:2"));
  EXPECT_FALSE(filter.empty());
  EXPECT_TRUE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 9, 2, 0)));
  // Wrong global occurrence.
  EXPECT_FALSE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 9, 1, 2)));
}

TEST(HloPassFilterTest, MatchesPipelineScope) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("simplification/algsimp"));

  EXPECT_TRUE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 5, 3, 1)));
  // Wrong parent pipeline.
  EXPECT_FALSE(filter.Matches(MakeInvocation("algsimp", "other", 5, 3, 1)));
}

TEST(HloPassFilterTest, MatchesScopedOccurrence) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromFlag("simplification/algsimp:2"));
  EXPECT_FALSE(filter.empty());
  // Matches on pipeline-scoped occurrence, ignoring the global occurrence.
  EXPECT_TRUE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 5, 7, 2)));
  // Wrong pipeline-scoped occurrence.
  EXPECT_FALSE(
      filter.Matches(MakeInvocation("algsimp", "simplification", 5, 2, 3)));
  // Wrong pipeline.
  EXPECT_FALSE(filter.Matches(MakeInvocation("algsimp", "other", 5, 2, 2)));
}

TEST(HloPassFilterTest, MatchesPassId) {
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter, HloPassFilter::FromFlag("@42"));
  EXPECT_FALSE(filter.empty());
  // pass_id match ignores name/pipeline/occurrence.
  EXPECT_TRUE(filter.Matches(MakeInvocation("anything", "anywhere", 42, 0, 0)));
  EXPECT_FALSE(
      filter.Matches(MakeInvocation("anything", "anywhere", 41, 0, 0)));
}

TEST(HloPassFilterTest, MatchesMultipleEntries) {
  ASSERT_OK_AND_ASSIGN(
      HloPassFilter filter,
      HloPassFilter::FromFlag("algsimp,dce:2,@42,simplification/reshape:1"));
  EXPECT_FALSE(filter.empty());
  EXPECT_TRUE(filter.Matches(MakeInvocation("algsimp", "any", 1, 0, 0)));
  EXPECT_TRUE(filter.Matches(MakeInvocation("dce", "any", 2, 2, 0)));
  EXPECT_TRUE(filter.Matches(MakeInvocation("foo", "bar", 42, 0, 0)));
  EXPECT_TRUE(
      filter.Matches(MakeInvocation("reshape", "simplification", 3, 0, 1)));
  EXPECT_FALSE(filter.Matches(MakeInvocation("dce", "any", 2, 1, 0)));
  EXPECT_FALSE(filter.Matches(MakeInvocation("other", "any", 99, 0, 0)));
}

TEST(HloPassFilterTest, FromRepeatedProtoFieldValid) {
  google::protobuf::RepeatedPtrField<std::string> entries;
  *entries.Add() = "algsimp";
  *entries.Add() = "dce:2";
  *entries.Add() = "@42";
  ASSERT_OK_AND_ASSIGN(HloPassFilter filter,
                       HloPassFilter::FromRepeatedProtoField(entries));
  EXPECT_FALSE(filter.empty());
  EXPECT_TRUE(filter.Matches(MakeInvocation("algsimp", "any", 1, 0, 0)));
  EXPECT_TRUE(filter.Matches(MakeInvocation("dce", "any", 2, 2, 0)));
  EXPECT_TRUE(filter.Matches(MakeInvocation("foo", "bar", 42, 0, 0)));
}

TEST(HloPassFilterTest, FromRepeatedProtoFieldInvalid) {
  google::protobuf::RepeatedPtrField<std::string> entries;
  *entries.Add() = "algsimp";
  *entries.Add() = "@notanumber";
  EXPECT_THAT(HloPassFilter::FromRepeatedProtoField(entries),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
