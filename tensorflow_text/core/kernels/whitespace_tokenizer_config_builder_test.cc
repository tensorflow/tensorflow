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

#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "icu4c/source/common/unicode/appendable.h"
#include "icu4c/source/common/unicode/bytestream.h"
#include "icu4c/source/common/unicode/edits.h"
#include "icu4c/source/common/unicode/normalizer2.h"
#include "icu4c/source/common/unicode/schriter.h"
#include "icu4c/source/common/unicode/stringoptions.h"
#include "icu4c/source/common/unicode/stringpiece.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/ucnv.h"
#include "icu4c/source/common/unicode/ucnv_err.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/uniset.h"
#include "icu4c/source/common/unicode/unistr.h"
#include "icu4c/source/common/unicode/uset.h"
#include "icu4c/source/common/unicode/utf.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"

namespace tensorflow {
namespace text {
namespace {

TEST(WhitespaceTokenizerConfigBuilderTest, BuildWhitespaceString) {
  std::string result = BuildWhitespaceString();
  EXPECT_THAT(result, ::testing::HasSubstr(" "));
  EXPECT_THAT(result, ::testing::HasSubstr("\n"));
}

TEST(WhitespaceTokenizerConfigBuilderTest,
     BuildWhitespaceTokenizerConfig_AllWhitespacePresent) {
  std::string whitespaces = BuildWhitespaceString();
  icu::UnicodeString codepoints = icu::UnicodeString::fromUTF8(whitespaces);
  std::string config = BuildWhitespaceTokenizerConfig();
  // verify all whitepaces are present
  WhitespaceTokenizerConfig cfg(config);
  for (int i = 0; i < codepoints.length(); ++i) {
    EXPECT_TRUE(cfg.IsWhitespace(codepoints[i]));
  }
}

TEST(WhitespaceTokenizerConfigBuilderTest,
     BuildWhitespaceTokenizerConfig_MinSize) {
  std::string whitespaces = BuildWhitespaceString();
  icu::UnicodeString codepoints = icu::UnicodeString::fromUTF8(whitespaces);
  std::string config = BuildWhitespaceTokenizerConfig();
  // verify we are the minimum perfect hash
  auto largest_cp = codepoints[codepoints.length() - 1];
  EXPECT_EQ(config.length(), (largest_cp / 8) + 1);
}

TEST(WhitespaceTokenizerConfigBuilderTest,
     BuildWhitespaceTokenizerConfig_VerifyCount) {
  std::string whitespaces = BuildWhitespaceString();
  icu::UnicodeString codepoints = icu::UnicodeString::fromUTF8(whitespaces);
  std::string config = BuildWhitespaceTokenizerConfig();
  // verify we have the correct number of true values (rest will be false)
  int count = 0;
  WhitespaceTokenizerConfig cfg(config);
  for (int i = 0; i < config.length() * 8; ++i) {
    count += cfg.IsWhitespace(i) ? 1 : 0;
  }
  EXPECT_EQ(count, codepoints.length());
}

}  // namespace
}  // namespace text
}  // namespace tensorflow
