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
#include "xla/hlo/tools/hlo_diff/utils/text_diff.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"

namespace xla::hlo_diff {
namespace {

using ::testing::ElementsAre;
using ::testing::FieldsAre;
using ::testing::IsEmpty;

// Helper to make TextDiffChunk creation more concise in tests.
TextDiffChunk Unchanged(absl::string_view text) {
  return {TextDiffType::kUnchanged, std::string(text)};
}
TextDiffChunk Added(absl::string_view text) {
  return {TextDiffType::kAdded, std::string(text)};
}
TextDiffChunk Removed(absl::string_view text) {
  return {TextDiffType::kRemoved, std::string(text)};
}

class TextDiffTest : public ::testing::Test {};

TEST_F(TextDiffTest, IdenticalStrings) {
  absl::string_view left = "abcde";
  absl::string_view right = "abcde";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(FieldsAre(TextDiffType::kUnchanged, "abcde")));
}

TEST_F(TextDiffTest, EmptyStrings) {
  absl::string_view left = "";
  absl::string_view right = "";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, IsEmpty());
}

TEST_F(TextDiffTest, LeftEmpty) {
  absl::string_view left = "";
  absl::string_view right = "abc";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(FieldsAre(TextDiffType::kAdded, "abc")));
}

TEST_F(TextDiffTest, RightEmpty) {
  absl::string_view left = "abc";
  absl::string_view right = "";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(FieldsAre(TextDiffType::kRemoved, "abc")));
}

TEST_F(TextDiffTest, CompletelyDifferent) {
  absl::string_view left = "abc";
  absl::string_view right = "def";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(FieldsAre(TextDiffType::kRemoved, "abc"),
                                FieldsAre(TextDiffType::kAdded, "def")));
}

TEST_F(TextDiffTest, Additions) {
  absl::string_view left = "ace";
  absl::string_view right = "abcde";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(Unchanged("a"), Added("b"), Unchanged("c"),
                                Added("d"), Unchanged("e")));
}

TEST_F(TextDiffTest, Removals) {
  absl::string_view left = "abcde";
  absl::string_view right = "ace";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(Unchanged("a"), Removed("b"), Unchanged("c"),
                                Removed("d"), Unchanged("e")));
}

TEST_F(TextDiffTest, Changes) {
  absl::string_view left = "axc";
  absl::string_view right = "ayc";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(Unchanged("a"), Removed("x"), Added("y"),
                                Unchanged("c")));
}

TEST_F(TextDiffTest, MixedChanges) {
  absl::string_view left = "a-cdefg";
  absl::string_view right = "ab-dxfg";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  EXPECT_THAT(diff, ElementsAre(Unchanged("a"), Added("b"), Unchanged("-"),
                                Removed("c"), Unchanged("d"), Removed("e"),
                                Added("x"), Unchanged("fg")));
}

TEST_F(TextDiffTest, LongerCommonSubsequence) {
  absl::string_view left = "AGGTAB";
  absl::string_view right = "GXTXAYB";
  std::vector<TextDiffChunk> diff = ComputeTextDiff(left, right);
  // LCS is GTAB
  EXPECT_THAT(diff, ElementsAre(Removed("AG"), Unchanged("G"), Added("X"),
                                Unchanged("T"), Added("X"), Unchanged("A"),
                                Added("Y"), Unchanged("B")));
}

}  // namespace
}  // namespace xla::hlo_diff
