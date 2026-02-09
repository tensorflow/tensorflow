// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace tensorflow {
namespace text {
namespace {

using ::testing::ElementsAre;

TEST(WhitespaceTokenizerTest, TokenizeWithOffsets) {
  absl::string_view input("I heard the news today");
  std::vector<std::string> output_tokens;
  std::vector<int> output_start_offsets;
  std::vector<int> output_end_offsets;
  std::string config(BuildWhitespaceTokenizerConfig());
  WhitespaceTokenizer t(&config);
  t.Tokenize(input, &output_tokens, &output_start_offsets, &output_end_offsets);
  EXPECT_THAT(output_tokens, ElementsAre("I", "heard", "the", "news", "today"));
  EXPECT_THAT(output_start_offsets, ElementsAre(0, 2, 8, 12, 17));
  EXPECT_THAT(output_end_offsets, ElementsAre(1, 7, 11, 16, 22));
}

TEST(WhitespaceTokenizerTest, Tokenize) {
  absl::string_view input("I heard the news today");
  std::vector<std::string> output_tokens;
  std::string config = BuildWhitespaceTokenizerConfig();
  WhitespaceTokenizer t(&config);
  t.Tokenize(input, &output_tokens);
  EXPECT_THAT(output_tokens, ElementsAre("I", "heard", "the", "news", "today"));
}

TEST(WhitespaceTokenizerTest, Internationalization) {
  absl::string_view input("la灯 灯a 瀮b");
  std::vector<std::string> output_tokens;
  std::vector<int> output_start_offsets;
  std::vector<int> output_end_offsets;
  std::string config = BuildWhitespaceTokenizerConfig();
  WhitespaceTokenizer t(&config);
  t.Tokenize(input, &output_tokens, &output_start_offsets, &output_end_offsets);
  EXPECT_THAT(output_start_offsets, ElementsAre(0, 6, 11));
  EXPECT_THAT(output_end_offsets, ElementsAre(5, 10, 15));
}

TEST(WhitespaceTokenizerTest, InvalidCodepoint) {
  absl::string_view input("\xE3");
  std::vector<std::string> output_tokens;
  std::vector<int> output_start_offsets;
  std::vector<int> output_end_offsets;
  std::string config = BuildWhitespaceTokenizerConfig();
  WhitespaceTokenizer t(&config);
  t.Tokenize(input, &output_tokens, &output_start_offsets, &output_end_offsets);
  EXPECT_THAT(output_start_offsets, ElementsAre(0));
  EXPECT_THAT(output_end_offsets, ElementsAre(1));
}

TEST(WhitespaceTokenizerTest, MaxCodepoint) {
  // Create an artificially-small config so that we can test behavior with
  // codepoints at the upper edge of its range. This bitmap marks 0x00-0x3f as
  // whitespace.
  std::string config(8, '\xff');
  // Verify that reading one bit off the end of the bitmap returns
  // not-whitespace.
  WhitespaceTokenizerConfig cfg(config);
  EXPECT_FALSE(cfg.IsWhitespace(0x40));
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
