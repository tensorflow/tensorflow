/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/parser/hlo_lexer.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "xla/primitive_util.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"

namespace xla {
namespace {

using absl::string_view;

constexpr int kEOF = -1;
constexpr int kError = -2;

// [a-zA-Z0-9_.-]
bool IsIdentifierChar(char c) {
  return absl::ascii_isalnum(static_cast<unsigned char>(c)) || c == '-' ||
         c == '.' || c == '_';
}

}  // namespace

int HloLexer::GetNextChar() {
  int current_char = PeekCurrentChar();
  if (current_char != kEOF && current_char != kError) {
    current_ptr_++;
  }
  return current_char;
}

int HloLexer::PeekCurrentChar() const {
  if (current_ptr_ == buf_.data() + buf_.size()) {
    return kEOF;
  }
  char current_char = *current_ptr_;
  if (current_char == 0) {
    // '\0' should not appear in the middle of the string.
    return kError;
  }
  return static_cast<unsigned char>(current_char);
}

bool HloLexer::CanDereference(const char* ptr) const {
  return (ptr < buf_.data() + buf_.size()) && ptr >= buf_.data();
}

absl::string_view HloLexer::StringViewFromPointers(const char* begin,
                                                   const char* end) const {
  CHECK(begin <= end);
  CHECK((begin == buf_.data() + buf_.size()) || CanDereference(begin));
  CHECK((end == buf_.data() + buf_.size()) || CanDereference(end));
  return absl::string_view(begin, end - begin);
}

TokKind HloLexer::LookAhead() {
  if (GetKind() == TokKind::kEof || GetKind() == TokKind::kError) {
    return GetKind();
  }

  const char* old_current_ptr = current_ptr_;
  TokenState old_token_state = token_state_;
  Lex();
  TokKind kind = GetKind();
  token_state_ = old_token_state;
  current_ptr_ = old_current_ptr;
  return kind;
}

TokKind HloLexer::LexToken() {
  while (true) {
    token_state_.token_start = current_ptr_;

    int current_char = GetNextChar();
    TokKind tmp;
    switch (current_char) {
      default:
        // [a-zA-Z_]
        if (absl::ascii_isalpha(static_cast<unsigned char>(current_char)) ||
            current_char == '_') {
          return LexIdentifier();
        }
        return TokKind::kError;
      case kEOF:
        // Hit the end of the input buffer.
        return TokKind::kEof;
      case kError:
        // Hit an invalid character in the input buffer.
        return TokKind::kError;
      case ' ':
      case '\t':
      case '\n':
      case '\r':
        // Ignore whitespace.
        continue;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case '-':
      case '?':
        if (current_char == '-' && PeekCurrentChar() == '>') {
          current_ptr_++;
          return TokKind::kArrow;
        }
        tmp = LexNumberOrPattern();
        if (tmp == TokKind::kError && current_char == '?') {
          return TokKind::kQuestionMark;
        }
        return tmp;
      case '=':
        return TokKind::kEqual;
      case '<':
        if (current_char == '<' && PeekCurrentChar() == '=') {
          current_ptr_++;
          return TokKind::kLeq;
        }
        return TokKind::kError;
      case ',':
        return TokKind::kComma;
      case '%':
        return LexPercent();
      case ':':
        return TokKind::kColon;
      case '*':
        return TokKind::kAsterisk;
      case '#':
        return TokKind::kOctothorp;
      case '+':
        return TokKind::kPlus;
      case '~':
        return TokKind::kTilde;
      case '[':
        return TokKind::kLsquare;
      case ']':
        return TokKind::kRsquare;
      case '{':
        return TokKind::kLbrace;
      case '}':
        return TokKind::kRbrace;
      case '(':
        return TokKind::kLparen;
      case ')':
        return TokKind::kRparen;
      case '/': {
        if (PeekCurrentChar() == '*') {
          // This is the start of a /*...*/ delimited comment. Save the current
          // location in case the comment is unterminated so the error message
          // will point to the beginning of the comment.
          const char* comment_start = current_ptr_;
          current_ptr_++;
          // Advance until '*/' is found.
          while (true) {
            int current = GetNextChar();
            if (current == '*' && PeekCurrentChar() == '/') {
              // End of comment.
              current_ptr_++;
              break;
            }
            if (current == kEOF) {
              // Unterminated comment.
              current_ptr_ = comment_start;
              return TokKind::kError;
            }
            if (current == kError) {
              return TokKind::kError;
            }
          }
          // Return no token for the comment. Keep lexing.
          continue;
        } else if (PeekCurrentChar() == '/') {
          // This is the start of a '//' delimited comment. Throw away
          // everything until end of line or file. The end-of-line character(s)
          // are left unlexed in the buffer which is harmless because these are
          // skipped later by the lexer. This approach enables support for
          // different end-of-line encodings.
          while (true) {
            int current = PeekCurrentChar();
            if (current == kEOF || current == '\n' || current == '\r') {
              break;
            }
            if (current == kError) {
              return TokKind::kError;
            }
            current_ptr_++;
          }
          continue;
        }
        // A lone '/' is an error.
        return TokKind::kError;
      }
      case '.':
        if (PeekCurrentChar() == '.') {
          current_ptr_++;
          if (PeekCurrentChar() == '.') {
            current_ptr_++;
            return TokKind::kDots;
          }
        }
        return TokKind::kError;
      case '"':
        return LexString();
    }
  }
}

std::optional<int64_t> HloLexer::LexNanPayload(absl::string_view& consumable) {
  static LazyRE2 payload_pattern = {R"(\(0x[0-9a-fA-F]+\))"};
  if (!RE2::Consume(&consumable, *payload_pattern)) {
    return std::nullopt;
  }
  auto slice = StringViewFromPointers(current_ptr_, consumable.data());
  current_ptr_ = consumable.data();
  CHECK(absl::StartsWith(slice, "(0x"));
  slice.remove_prefix(std::strlen("(0x"));
  CHECK(absl::EndsWith(slice, ")"));
  slice.remove_suffix(std::strlen(")"));
  uint64_t payload_value;
  if (tsl::strings::HexStringToUint64(slice, &payload_value)) {
    if (payload_value <= 0 || payload_value > NanPayloadBitMask<double>()) {
      LOG(ERROR) << "NaN payload out of range: " << payload_value;
      return std::nullopt;
    }
    return payload_value;
  }
  return std::nullopt;
}

// Lex a shape, name, keyword, attribute name, the dim labels pattern, and
// other identifiers.
//
// shape    ::= ([a-zA-Z0-9_]*[0-9]*)\[([0-9,]*)\](?:\s*{([0-9,]*)})?
// name     ::= [a-zA-Z_][a-zA-Z0-9_.-]*:
// keyword  ::= HloModule, ENTRY, ...
// attribute_name ::= condition, body, dimensions, ...
// dim_labels_pattern ::= [0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,}
// identifiers ::= other cases that match [a-zA-Z_][a-zA-Z0-9_.-]*
TokKind HloLexer::LexIdentifier() {
  while (IsIdentifierChar(PeekCurrentChar())) {
    current_ptr_++;
  }

  // If followed by ':', it's a name.
  if (PeekCurrentChar() == ':') {
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    current_ptr_++;  // skip ':'
    return TokKind::kName;
  }

  // If followed by '=', it's a attribute name.
  if (PeekCurrentChar() == '=') {
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    current_ptr_++;  // skip '='
    return TokKind::kAttributeName;
  }

  absl::string_view identifier =
      StringViewFromPointers(token_state_.token_start, current_ptr_);

  // Primitive type strings are reserved words. The exception is 'tuple' whose
  // type is represented using nested parentheses without the string 'tuple'.
  if (primitive_util::IsPrimitiveTypeName(identifier)) {
    PrimitiveType primitive_type =
        primitive_util::StringToPrimitiveType(identifier).value();
    if (primitive_type != TUPLE) {
      token_state_.primitive_type_val = primitive_type;
      return TokKind::kPrimitiveType;
    }
  }

  if (identifier == "nan") {
    std::optional<int64_t> payload;
    if (PeekCurrentChar() == '(') {
      absl::string_view consumable =
          StringViewFromPointers(current_ptr_, buf_.data() + buf_.size());
      payload = LexNanPayload(consumable);
      if (!payload.has_value()) {
        return TokKind::kError;
      }
    }
    token_state_.decimal_val = NanWithSignAndPayload<double>(
        /*sign=*/false, payload.value_or(QuietNanWithoutPayload<double>()));
    return TokKind::kDecimal;
  }

  // See if this is a keyword.
#define KEYWORD(STR)            \
  do {                          \
    if (identifier == #STR) {   \
      return TokKind::kw_##STR; \
    }                           \
  } while (false)

  KEYWORD(true);
  KEYWORD(false);
  KEYWORD(inf);
  KEYWORD(HloModule);
  KEYWORD(ENTRY);
  KEYWORD(ROOT);
  KEYWORD(maximal);
  KEYWORD(replicated);
  KEYWORD(manual);
  KEYWORD(last_tile_dim_replicate);
  KEYWORD(shard_as);
  KEYWORD(shard_like);
  KEYWORD(unknown);

#undef KEYWORD

  {
    absl::string_view consumable = StringViewFromPointers(
        token_state_.token_start, buf_.data() + buf_.size());
    static LazyRE2 dim_labels_pattern = {
        R"([0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,})"};
    if (RE2::Consume(&consumable, *dim_labels_pattern)) {
      current_ptr_ = consumable.data();
      token_state_.str_val.assign(token_state_.token_start, current_ptr_);
      return TokKind::kDimLabels;
    }
    static LazyRE2 sparsity_desc_pattern = {
        R"(([LR]\.[0-9]+@[0-9]+:[0-9]+_?)+)"};
    if (RE2::Consume(&consumable, *sparsity_desc_pattern)) {
      current_ptr_ = consumable.data();
      token_state_.str_val.assign(token_state_.token_start, current_ptr_);
      return TokKind::kSparsityDesc;
    }
  }

  token_state_.str_val = std::string(identifier);
  return TokKind::kIdent;
}

// Lex names after a % character.
// name ::= [a-zA-Z_][a-zA-Z0-9_.-]*
TokKind HloLexer::LexPercent() {
  const char* name_start = current_ptr_;
  if (absl::ascii_isalpha(static_cast<unsigned char>(PeekCurrentChar())) ||
      PeekCurrentChar() == '_') {
    current_ptr_++;
    while (IsIdentifierChar(PeekCurrentChar())) {
      current_ptr_++;
    }
    token_state_.str_val.assign(name_start, current_ptr_);
    return TokKind::kName;
  }
  return TokKind::kError;
}

// Lex integer and floating-point values, -inf, and patterns for dim labels,
// dxd (e.g. 1x2x3), and pad.
//
// fp with exp ::= [-]?([0-9]+|[0-9]+[.][0-9]*|[0-9]*[.][0-9]+)([eE][+-]?[0-9]+)
// fp without exp ::= [-]?([0-9]+[.][0-9]*|[0-9]*[.][0-9]+)
// dim_labels_pattern ::= [0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,}
// dxd_pattern ::= [0-9]+(x[0-9]+)+
// pad_pattern ::=
//   [-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?(x[-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?)*
// int ::=  [-]?[0-9]+
// negative inf ::= '-inf'
TokKind HloLexer::LexNumberOrPattern() {
  absl::string_view consumable = StringViewFromPointers(
      token_state_.token_start, buf_.data() + buf_.size());
  static LazyRE2 float_pattern = {
      R"([-]?((\d+|\d+[.]\d*|\d*[.]\d+)([eE][+-]?\d+))|[-]?(\d+[.]\d*|\d*[.]\d+))"};
  if (RE2::Consume(&consumable, *float_pattern)) {
    current_ptr_ = consumable.data();
    CHECK(absl::SimpleAtod(std::string(token_state_.token_start, current_ptr_),
                           &token_state_.decimal_val));
    return TokKind::kDecimal;
  }

  static LazyRE2 dim_labels_pattern = {
      R"([0-9bf?]{2,}_[0-9io?]{2,}->[0-9bf?]{2,})"};
  static LazyRE2 dxd_pattern = {R"([0-9]+(x[0-9]+)+)"};
  static LazyRE2 pad_pattern = {
      R"([-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?(x[-]?[0-9]+_[-]?[0-9]+(_[0-9]+)?)*)"};

  if (RE2::Consume(&consumable, *dim_labels_pattern)) {
    current_ptr_ = consumable.data();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kDimLabels;
  }

  if (RE2::Consume(&consumable, *dxd_pattern)) {
    current_ptr_ = consumable.data();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kDxD;
  }

  if (RE2::Consume(&consumable, *pad_pattern)) {
    current_ptr_ = consumable.data();
    token_state_.str_val.assign(token_state_.token_start, current_ptr_);
    return TokKind::kPad;
  }

  static LazyRE2 int_pattern = {R"([-]?\d+)"};
  if (RE2::Consume(&consumable, *int_pattern)) {
    current_ptr_ = consumable.data();
    auto slice = StringViewFromPointers(token_state_.token_start, current_ptr_);
    if (absl::SimpleAtoi(slice, &token_state_.int64_val)) {
      return TokKind::kInt;
    }
    uint64_t uint64_val;
    if (absl::SimpleAtoi(slice, &uint64_val)) {
      token_state_.int64_val = absl::bit_cast<int64_t>(uint64_val);
      return TokKind::kInt;
    }
    LOG(ERROR) << "Failed to parse int literal: " << slice;
    return TokKind::kError;
  }

  static LazyRE2 neg_inf = {"-inf"};
  if (RE2::Consume(&consumable, *neg_inf)) {
    current_ptr_ = consumable.data();
    return TokKind::kNegInf;
  }

  static LazyRE2 neg_nan = {"-nan"};
  if (RE2::Consume(&consumable, *neg_nan)) {
    current_ptr_ = consumable.data();

    std::optional<int64_t> payload;
    if (PeekCurrentChar() == '(') {
      payload = LexNanPayload(consumable);
      if (!payload.has_value()) {
        return TokKind::kError;
      }
    }
    token_state_.decimal_val = NanWithSignAndPayload<double>(
        /*sign=*/true, payload.value_or(QuietNanWithoutPayload<double>()));
    return TokKind::kDecimal;
  }

  return TokKind::kError;
}

std::pair<unsigned, unsigned> HloLexer::GetLineAndColumn(LocTy location) const {
  unsigned line_no = 1;
  const char* start = buf_.data();
  const char* ptr = start;
  if (line_no_cache_.last_query && CanDereference(line_no_cache_.last_query) &&
      line_no_cache_.last_query <= location) {
    ptr = line_no_cache_.last_query;
    line_no = line_no_cache_.line_no_of_query;
  }
  for (; ptr != location; ptr++) {
    CHECK_LT(ptr, buf_.data() + buf_.size());
    if (*ptr == '\n') {
      line_no++;
    }
  }

  // Update the line number cache.
  line_no_cache_.last_query = ptr;
  line_no_cache_.line_no_of_query = line_no;
  size_t line_offset = StringViewFromPointers(start, ptr).rfind('\n');
  if (line_offset == absl::string_view::npos) {
    line_offset = 0;
  }
  return {line_no, ptr - start - line_offset};
}

absl::string_view HloLexer::GetLine(LocTy loc) const {
  if (!CanDereference(loc)) {
    return "LINE OUT OF RANGE";
  }
  size_t line_start = StringViewFromPointers(buf_.data(), loc + 1).rfind('\n');
  const char* start = line_start == absl::string_view::npos
                          ? buf_.data()
                          : buf_.data() + line_start + 1;
  size_t line_end =
      StringViewFromPointers(loc, buf_.data() + buf_.size()).find('\n');
  const char* end = line_end == absl::string_view::npos
                        ? buf_.data() + buf_.size()
                        : loc + line_end;

  return StringViewFromPointers(start, end);
}

// Lexes quoted string with escaping characters. If matched, the quoted string
// will be unescaped and stored to token_state_.str_val.
TokKind HloLexer::LexString() {
  absl::string_view consumable = StringViewFromPointers(
      token_state_.token_start, buf_.data() + buf_.size());
  static LazyRE2 escaping_pattern = {R"("([^"\\]|\\.)*")"};
  if (RE2::Consume(&consumable, *escaping_pattern)) {
    current_ptr_ = consumable.data();
    absl::string_view raw =
        StringViewFromPointers(token_state_.token_start + 1, current_ptr_ - 1);
    std::string error;
    if (!absl::CUnescape(raw, &token_state_.str_val, &error)) {
      LOG(ERROR) << "Failed unescaping string: " << raw << ". error: " << error;
      return TokKind::kError;
    }
    return TokKind::kString;
  }
  return TokKind::kError;
}

TokKind HloLexer::LexJsonDict() {
  // We require that you've already lexed the open curly brace.
  if (GetKind() != TokKind::kLbrace) return TokKind::kError;

  absl::string_view orig = StringViewFromPointers(token_state_.token_start,
                                                  buf_.data() + buf_.size());
  absl::string_view str = orig;

  int64_t object_depth = 0;
  if (str.empty()) return TokKind::kError;

  if (str.front() != '{') return TokKind::kError;
  ++object_depth;
  str = str.substr(1);

  while (!str.empty()) {
    if (object_depth == 0) break;

    if (str.front() == '"') {
      static LazyRE2 string_pattern = {R"("([^"\\]|\\.)*")"};
      if (!RE2::Consume(&str, *string_pattern)) {
        return TokKind::kError;
      }
      continue;
    }

    if (str.front() == '{') ++object_depth;
    if (str.front() == '}') --object_depth;
    str = str.substr(1);
  }
  if (object_depth != 0) {
    return TokKind::kError;
  }
  current_ptr_ = str.data();
  token_state_.current_kind = TokKind::kString;
  token_state_.str_val =
      std::string(orig.substr(0, orig.length() - str.length()));
  return TokKind::kString;
}

std::string TokKindToString(TokKind kind) {
  switch (kind) {
    case TokKind::kEof:
      return "kEof";
    case TokKind::kError:
      return "kError";
    case TokKind::kEqual:
      return "kEqual";
    case TokKind::kComma:
      return "kComma";
    case TokKind::kColon:
      return "kColon";
    case TokKind::kAsterisk:
      return "kAsterisk";
    case TokKind::kQuestionMark:
      return "kQuestionMark";
    case TokKind::kOctothorp:
      return "kOctothorp";
    case TokKind::kPlus:
      return "kPlus";
    case TokKind::kTilde:
      return "kTilde";
    case TokKind::kLsquare:
      return "kLsquare";
    case TokKind::kRsquare:
      return "kRsquare";
    case TokKind::kLbrace:
      return "kLbrace";
    case TokKind::kRbrace:
      return "kRbrace";
    case TokKind::kLparen:
      return "kLparen";
    case TokKind::kRparen:
      return "kRparen";
    case TokKind::kArrow:
      return "kArrow";
    case TokKind::kLeq:
      return "kLeq";
    case TokKind::kw_HloModule:
      return "kw_HloModule";
    case TokKind::kw_ENTRY:
      return "kw_ENTRY";
    case TokKind::kw_ROOT:
      return "kw_ROOT";
    case TokKind::kw_true:
      return "kw_true";
    case TokKind::kw_false:
      return "kw_false";
    case TokKind::kw_maximal:
      return "kw_maximal";
    case TokKind::kw_replicated:
      return "kw_replicated";
    case TokKind::kw_manual:
      return "kw_manual";
    case TokKind::kw_last_tile_dim_replicate:
      return "kw_last_tile_dim_replicate";
    case TokKind::kw_shard_as:
      return "kw_shard_as";
    case TokKind::kw_shard_like:
      return "kw_shard_like";
    case TokKind::kw_unknown:
      return "kw_unknown";
    case TokKind::kw_inf:
      return "kw_inf";
    case TokKind::kNegInf:
      return "kNegInf";
    case TokKind::kPrimitiveType:
      return "kPrimitiveType";
    case TokKind::kName:
      return "kName";
    case TokKind::kAttributeName:
      return "kAttributeName";
    case TokKind::kDimLabels:
      return "kDimLabels";
    case TokKind::kDxD:
      return "kDxD";
    case TokKind::kPad:
      return "kPad";
    case TokKind::kSparsityDesc:
      return "kSparsityDesc";
    case TokKind::kIdent:
      return "kIdent";
    case TokKind::kString:
      return "kString";
    case TokKind::kInt:
      return "kInt";
    case TokKind::kDecimal:
      return "kDecimal";
    case TokKind::kDots:
      return "kDots";
  }
}

bool LexesAsJsonDict(absl::string_view str) {
  HloLexer lexer(str);
  return lexer.Lex() == TokKind::kLbrace &&
         lexer.LexJsonDict() == TokKind::kString &&
         lexer.Lex() == TokKind::kEof;
}

}  // namespace xla
