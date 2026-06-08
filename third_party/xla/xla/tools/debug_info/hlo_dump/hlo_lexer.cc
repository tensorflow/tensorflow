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

#include "xla/tools/debug_info/hlo_dump/hlo_lexer.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "re2/re2.h"

namespace xla::tools::debug_info {

const char* TokKindToClass(TokKind kind) {
  switch (kind) {
    case TokKind::kComment:
      return "c1";
    case TokKind::kCommentSpecial:
      return "cs";
    case TokKind::kString:
      return "s";
    case TokKind::kKeyword:
      return "k";
    case TokKind::kKeywordType:
      return "kt";
    case TokKind::kNameVariable:
      return "nv";
    case TokKind::kNameFunction:
      return "nf";
    case TokKind::kNameComputation:
      return "nc";
    case TokKind::kNumber:
      return "m";
    case TokKind::kPunctuation:
      return "p";
    case TokKind::kName:
      return "n";
    default:
      return "";
  }
}

namespace {
static const LazyRE2 re_text = {"^\\s+"};
static const LazyRE2 re_comment = {"^[ ]*//.*"};
static const LazyRE2 re_comment_special1 = {"^(## BEGIN_GRAPH|## END_GRAPH)"};
static const LazyRE2 re_comment_special2 = {
    "^(\\[ScheduleSyncTensorsGraph\\].*)"};
static const LazyRE2 re_string = {"^(\"(?:[^\\\\\"]|\\\\.)*\")"};
static const LazyRE2 re_keyword = {
    "^\\b(true|false|inf|HloModule|ENTRY|ROOT|maximal|replicated|manual|last_"
    "tile_dim_replicate)\\b"};
static const LazyRE2 re_keyword_type = {
    "^\\b(pred|s8|s16|s32|s64|u8|u16|u32|u64|f16|f32|bf16|f64|c64|c128|tuple|"
    "opaque_type|token)\\b"};
static const LazyRE2 re_name_variable = {"^(%[a-zA-Z0-9_.-]+)"};
static const LazyRE2 re_name_computation = {"^(@[a-zA-Z0-9_.-]+)"};
static const LazyRE2 re_name_function = {"^([a-zA-Z_][a-zA-Z0-9_.-]*)(\\()"};
static const LazyRE2 re_number = {"^(-?inf|-?[0-9]+\\.[0-9]+|-?[0-9]+)"};
static const LazyRE2 re_punctuation = {"^([{},=()\\[\\]]|->)"};
static const LazyRE2 re_keyword_fusion = {"^\\b(fusion)\\b"};
static const LazyRE2 re_keyword_custom_call = {"^\\b(custom-call)\\b"};
static const LazyRE2 re_name = {"^([a-zA-Z_][a-zA-Z0-9_.-]*)"};
// This is placed before keyword checks to ensure that names which might start
// with a keyword but contain a dot (e.g., "tuple.1") are tokenized as a single
// kName rather than splitting at the dot.
static const LazyRE2 re_name_with_dot = {
    "^([a-zA-Z_][a-zA-Z0-9_.-]*\\.[a-zA-Z0-9_.-]*)"};
}  // namespace

std::vector<Token> LexHlo(absl::string_view hlo_text) {
  std::vector<Token> tokens;
  absl::string_view input = hlo_text;

  while (!input.empty()) {
    absl::string_view m1, m2;
    if (RE2::Consume(&input, *re_text, &m1)) {
      tokens.push_back({TokKind::kText, std::string(m1)});
    } else if (RE2::Consume(&input, *re_comment, &m1)) {
      tokens.push_back({TokKind::kComment, std::string(m1)});
    } else if (RE2::Consume(&input, *re_comment_special1, &m1)) {
      tokens.push_back({TokKind::kCommentSpecial, std::string(m1)});
    } else if (RE2::Consume(&input, *re_comment_special2, &m1)) {
      tokens.push_back({TokKind::kCommentSpecial, std::string(m1)});
    } else if (RE2::Consume(&input, *re_string, &m1)) {
      tokens.push_back({TokKind::kString, std::string(m1)});
    } else if (RE2::Consume(&input, *re_name_with_dot, &m1)) {
      tokens.push_back({TokKind::kName, std::string(m1)});
    } else if (RE2::Consume(&input, *re_keyword, &m1)) {
      tokens.push_back({TokKind::kKeyword, std::string(m1)});
    } else if (RE2::Consume(&input, *re_keyword_type, &m1)) {
      tokens.push_back({TokKind::kKeywordType, std::string(m1)});
    } else if (RE2::Consume(&input, *re_name_variable, &m1)) {
      tokens.push_back({TokKind::kNameVariable, std::string(m1)});
    } else if (RE2::Consume(&input, *re_name_computation, &m1)) {
      tokens.push_back({TokKind::kNameComputation, std::string(m1)});
    } else if (RE2::Consume(&input, *re_name_function, &m1, &m2)) {
      tokens.push_back({TokKind::kNameFunction, std::string(m1)});
      tokens.push_back({TokKind::kPunctuation, std::string(m2)});
    } else if (RE2::Consume(&input, *re_number, &m1)) {
      tokens.push_back({TokKind::kNumber, std::string(m1)});
    } else if (RE2::Consume(&input, *re_punctuation, &m1)) {
      tokens.push_back({TokKind::kPunctuation, std::string(m1)});
    } else if (RE2::Consume(&input, *re_keyword_fusion, &m1)) {
      tokens.push_back({TokKind::kKeyword, std::string(m1)});
    } else if (RE2::Consume(&input, *re_keyword_custom_call, &m1)) {
      tokens.push_back({TokKind::kKeyword, std::string(m1)});
    } else if (RE2::Consume(&input, *re_name, &m1)) {
      tokens.push_back({TokKind::kName, std::string(m1)});
    } else {
      tokens.push_back({TokKind::kText, std::string(input.substr(0, 1))});
      input.remove_prefix(1);
    }
  }
  return tokens;
}

}  // namespace xla::tools::debug_info
