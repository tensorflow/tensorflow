// Copyright 2026 The OpenXLA Authors.
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

#ifndef XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_LEXER_H_
#define XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_LEXER_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace xla::tools::debug_info {

// Rationale for having a dedicated lexer instead of using xla::HloParser (or
// xla::HloLexer):
// 1. HTML dumping must preserve the exact original HLO text layout, including
//    all whitespace, formatting, and comments, for correct rendering and
//    annotation mapping. The production parser/lexer discards whitespace and
//    comments entirely.
// 2. The HTML dumper needs to be robust to malformed or partial HLO text
//    without failing. Unlike xla::HloParser, which enforces full semantic
//    validity, this lexer simply partitions the source text sequentially and
//    gracefully.
enum class TokKind {
  kText,
  kComment,
  kCommentSpecial,
  kString,
  kKeyword,
  kKeywordType,
  kNameVariable,
  kNameFunction,
  kNameComputation,
  kNumber,
  kPunctuation,
  kName,
  kOther
};

struct Token {
  TokKind kind;
  std::string value;
};

std::vector<Token> LexHlo(absl::string_view hlo_text);
const char* TokKindToClass(TokKind kind);

}  // namespace xla::tools::debug_info

#endif  // XLA_TOOLS_DEBUG_INFO_HLO_DUMP_HLO_LEXER_H_
