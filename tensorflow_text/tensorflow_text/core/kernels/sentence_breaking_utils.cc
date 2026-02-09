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

#include "tensorflow_text/core/kernels/sentence_breaking_utils.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/utypes.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

using ::tensorflow::Status;

namespace tensorflow {
namespace text {

absl::Status UnicodeUtil::GetOneUChar(const absl::string_view& input,
                                      bool* has_more_than_one_char,
                                      UChar32* result) const {
  UErrorCode status = U_ZERO_ERROR;
  const char* source = input.data();
  const char* limit = input.data() + input.length();
  if (!converter_) {
    return tensorflow::errors::Internal(
        absl::StrCat("Converter has not been initialized!"));
  }
  *result = ucnv_getNextUChar(converter_, &source, limit, &status);

  if (U_FAILURE(status)) {
    return tensorflow::errors::Internal(
        absl::StrCat("Failed to decode string, error status=", status));
  }

  if (source != limit) {
    *has_more_than_one_char = true;
  } else {
    *has_more_than_one_char = false;
  }

  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsTerminalPunc(const absl::string_view& input,
                                         bool* result) const {
  *result = false;
  const auto& ellipsis_status = IsEllipsis(input, result);
  // If there was a error decoding, or if we found an ellipsis, then return.
  if (!ellipsis_status.ok()) return ellipsis_status;
  if (*result) return absl::OkStatus();

  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case 0x055C:  // Armenian exclamation mark
    case 0x055E:  // Armenian question mark
    case 0x17d4:  // Khmer sign khan
    case 0x037E:  // Greek question mark
    case 0x2026:  // ellipsis
      *result = true;
      return absl::OkStatus();
  }

  USentenceBreak sb_property = static_cast<USentenceBreak>(
      u_getIntPropertyValue(char_value, UCHAR_SENTENCE_BREAK));
  *result = sb_property == U_SB_ATERM || sb_property == U_SB_STERM;
  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsClosePunc(const absl::string_view& input,
                                      bool* result) const {
  *result = false;
  if (input == "''") {
    *result = true;
    return absl::OkStatus();
  }

  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '>':
    case ']':
    case '`':
    case 64831:  // Ornate right parenthesis
    case 65282:  // fullwidth quotation mark
    case 65287:  // fullwidth apostrophe
      *result = true;
      return absl::OkStatus();
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));

  *result = lb_property == U_LB_CLOSE_PUNCTUATION ||
            lb_property == U_LB_CLOSE_PARENTHESIS ||
            lb_property == U_LB_QUOTATION;
  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsOpenParen(const absl::string_view& input,
                                      bool* result) const {
  *result = false;
  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '<':
    case 64830:  // Ornate left parenthesis
      *result = true;
      return absl::OkStatus();
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));
  *result = lb_property == U_LB_OPEN_PUNCTUATION;
  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsCloseParen(const absl::string_view& input,
                                       bool* result) const {
  *result = false;
  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '>':
    case 64831:  // Ornate right parenthesis
      *result = true;
      return absl::OkStatus();
  }

  ULineBreak lb_property = static_cast<ULineBreak>(
      u_getIntPropertyValue(char_value, UCHAR_LINE_BREAK));
  *result = lb_property == U_LB_CLOSE_PUNCTUATION ||
            lb_property == U_LB_CLOSE_PARENTHESIS;
  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsPunctuationWord(const absl::string_view& input,
                                            bool* result) const {
  *result = false;
  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  // These are unicode characters that should be considered in this category but
  // are not covered by any of the ICU properties.
  switch (char_value) {
    case '`':
    case '<':
    case '>':
    case '~':
    case 5741:
      *result = true;
      return absl::OkStatus();
  }

  *result = u_ispunct(char_value) ||
            u_hasBinaryProperty(char_value, UCHAR_DASH) ||
            u_hasBinaryProperty(char_value, UCHAR_HYPHEN);
  return absl::OkStatus();
}

absl::Status UnicodeUtil::IsEllipsis(const absl::string_view& input,
                                     bool* result) const {
  *result = false;
  if (input == "...") {
    *result = true;
    return absl::OkStatus();
  }

  bool has_more_than_one_char = false;
  UChar32 char_value;
  const auto& status = GetOneUChar(input, &has_more_than_one_char, &char_value);
  if (!status.ok()) return status;
  if (has_more_than_one_char) {
    *result = false;
    return absl::OkStatus();
  }

  *result = char_value == 0x2026;
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
