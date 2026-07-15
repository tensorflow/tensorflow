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

#include "xla/tools/hlo_dump/hlo_lexer.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/test.h"

namespace xla::numerics::debug_info {
namespace {

std::vector<Token> FilterSpaces(absl::Span<const Token> tokens) {
  std::vector<Token> filtered;
  for (const auto& tok : tokens) {
    if (tok.kind == TokKind::kText &&
        tok.value.find_first_not_of(" \t\n\r") == std::string::npos) {
      continue;
    }
    filtered.push_back(tok);
  }
  return filtered;
}

TEST(HloLexerTest, LexBasicInstruction) {
  std::vector<Token> tokens = LexHlo("%foo = f32[10] add(%p0, %p1)");
  std::vector<Token> filtered = FilterSpaces(tokens);

  ASSERT_EQ(filtered.size(), 12);
  EXPECT_EQ(filtered[0].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[0].value, "%foo");
  EXPECT_EQ(filtered[1].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[1].value, "=");
  EXPECT_EQ(filtered[2].kind, TokKind::kKeywordType);
  EXPECT_EQ(filtered[2].value, "f32");
  EXPECT_EQ(filtered[3].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[3].value, "[");
  EXPECT_EQ(filtered[4].kind, TokKind::kNumber);
  EXPECT_EQ(filtered[4].value, "10");
  EXPECT_EQ(filtered[5].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[5].value, "]");
  EXPECT_EQ(filtered[6].kind, TokKind::kNameFunction);
  EXPECT_EQ(filtered[6].value, "add");
  EXPECT_EQ(filtered[7].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[7].value, "(");
  EXPECT_EQ(filtered[8].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[8].value, "%p0");
  EXPECT_EQ(filtered[9].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[9].value, ",");
  EXPECT_EQ(filtered[10].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[10].value, "%p1");
  EXPECT_EQ(filtered[11].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[11].value, ")");
}

TEST(HloLexerTest, LexNameWithDot) {
  std::vector<Token> tokens = LexHlo("while.5/arg_tuple.1@{0}");
  std::vector<Token> filtered = FilterSpaces(tokens);

  // while.5 (kName)
  // / (kText)
  // arg_tuple.1 (kName)
  // @ (kText)
  // { (kPunctuation)
  // 0 (kNumber)
  // } (kPunctuation)

  ASSERT_EQ(filtered.size(), 7);
  EXPECT_EQ(filtered[0].kind, TokKind::kName);
  EXPECT_EQ(filtered[0].value, "while.5");
  EXPECT_EQ(filtered[1].kind, TokKind::kText);
  EXPECT_EQ(filtered[1].value, "/");
  EXPECT_EQ(filtered[2].kind, TokKind::kName);
  EXPECT_EQ(filtered[2].value, "arg_tuple.1");
  EXPECT_EQ(filtered[3].kind, TokKind::kText);
  EXPECT_EQ(filtered[3].value, "@");
  EXPECT_EQ(filtered[4].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[4].value, "{");
  EXPECT_EQ(filtered[5].kind, TokKind::kNumber);
  EXPECT_EQ(filtered[5].value, "0");
  EXPECT_EQ(filtered[6].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[6].value, "}");
}

TEST(HloLexerTest, LexCompactGte) {
  std::vector<Token> tokens =
      LexHlo("%foo = f32[10] add(%p0#0, %p1#1#0, %p2#-1)");
  std::vector<Token> filtered = FilterSpaces(tokens);

  ASSERT_EQ(filtered.size(), 14);
  EXPECT_EQ(filtered[0].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[0].value, "%foo");
  EXPECT_EQ(filtered[1].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[1].value, "=");
  EXPECT_EQ(filtered[2].kind, TokKind::kKeywordType);
  EXPECT_EQ(filtered[2].value, "f32");
  EXPECT_EQ(filtered[3].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[3].value, "[");
  EXPECT_EQ(filtered[4].kind, TokKind::kNumber);
  EXPECT_EQ(filtered[4].value, "10");
  EXPECT_EQ(filtered[5].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[5].value, "]");
  EXPECT_EQ(filtered[6].kind, TokKind::kNameFunction);
  EXPECT_EQ(filtered[6].value, "add");
  EXPECT_EQ(filtered[7].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[7].value, "(");
  EXPECT_EQ(filtered[8].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[8].value, "%p0#0");
  EXPECT_EQ(filtered[9].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[9].value, ",");
  EXPECT_EQ(filtered[10].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[10].value, "%p1#1#0");
  EXPECT_EQ(filtered[11].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[11].value, ",");
  EXPECT_EQ(filtered[12].kind, TokKind::kNameVariable);
  EXPECT_EQ(filtered[12].value, "%p2#-1");
  EXPECT_EQ(filtered[13].kind, TokKind::kPunctuation);
  EXPECT_EQ(filtered[13].value, ")");
}

}  // namespace
}  // namespace xla::numerics::debug_info
