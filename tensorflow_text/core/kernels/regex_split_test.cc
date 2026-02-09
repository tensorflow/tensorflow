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

#include "tensorflow_text/core/kernels/regex_split.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "re2/re2.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {
namespace text {
namespace {

std::vector<absl::string_view> RunTest(const tstring& input,
                                       const tstring& regex,
                                       const tstring& delim_regex) {
  RE2 re2((absl::string_view(regex)));
  RE2 include_delim_re2((absl::string_view(delim_regex)));

  std::vector<int64_t> begin_offsets;
  std::vector<int64_t> end_offsets;
  std::vector<absl::string_view> tokens;

  RegexSplit(input, re2, true, include_delim_re2, &tokens, &begin_offsets,
             &end_offsets);
  return tokens;
}

TEST(RegexSplitTest, JapaneseAndWhitespace) {
  tstring regex = "(\\p{Hiragana}+|\\p{Katakana}+|\\s)";
  tstring delim_regex = "(\\p{Hiragana}+|\\p{Katakana}+)";
  tstring input = "He said フランスです";
  auto extracted_tokens = RunTest(input, regex, delim_regex);
  EXPECT_THAT(extracted_tokens, testing::ElementsAreArray({
                                    "He",
                                    "said",
                                    "フランス",
                                    "です",
                                }));
}

TEST(RegexSplitTest, Japanese) {
  tstring regex = "(\\p{Hiragana}+|\\p{Katakana}+)";
  tstring input = "He said フランスです";
  auto extracted_tokens = RunTest(input, regex, regex);
  EXPECT_THAT(extracted_tokens, testing::ElementsAreArray({
                                    "He said ",
                                    "フランス",
                                    "です",
                                }));
}

TEST(RegexSplitTest, ChineseHan) {
  tstring regex = "(\\p{Han})";
  tstring input = "敵人變盟友背後盤算";
  auto extracted_tokens = RunTest(input, regex, regex);
  EXPECT_THAT(extracted_tokens,
              testing::ElementsAreArray(
                  {"敵", "人", "變", "盟", "友", "背", "後", "盤", "算"}));
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
