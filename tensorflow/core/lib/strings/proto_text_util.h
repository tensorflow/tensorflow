/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
#define TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_

#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/scanner.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"

namespace tensorflow {
namespace strings {

static constexpr char kColonSeparator[] = ": ";

// Helper functions for writing proto-text output.
// Used by the code generated from tools/proto_text/gen_proto_text_lib.cc.
class ProtoTextOutput {
 public:
  // Construct a ProtoTextOutput that writes to <output> If short_debug is true,
  // outputs text to match proto.ShortDebugString(); else matches
  // proto.DebugString().
  ProtoTextOutput(string* output, bool short_debug)
      : output_(output),
        short_debug_(short_debug),
        field_separator_(short_debug ? " " : "\n") {}

  // Writes opening of nested message and increases indent level.
  void OpenNestedMessage(const char field_name[]) {
    StrAppend(output_, level_empty_ ? "" : field_separator_, indent_,
              field_name, " {", field_separator_);
    if (!short_debug_) StrAppend(&indent_, "  ");
    level_empty_ = true;
  }

  // Writes close of nested message and decreases indent level.
  void CloseNestedMessage() {
    if (!short_debug_) indent_.resize(indent_.size() - 2);
    StrAppend(output_, level_empty_ ? "" : field_separator_, indent_, "}");
    level_empty_ = false;
  }

  // Print the close of the top-level message that was printed.
  void CloseTopMessage() {
    if (!short_debug_ && !level_empty_) StrAppend(output_, "\n");
  }

  // Appends a numeric value, like my_field: 123
  template <typename T>
  void AppendNumeric(const char field_name[], T value) {
    AppendFieldAndValue(field_name, StrCat(value));
  }

  // Appends a numeric value, like my_field: 123, but only if value != 0.
  template <typename T>
  void AppendNumericIfNotZero(const char field_name[], T value) {
    if (value != 0) AppendNumeric(field_name, value);
  }

  // Appends a bool value, either my_field: true or my_field: false.
  void AppendBool(const char field_name[], bool value) {
    AppendFieldAndValue(field_name, value ? "true" : "false");
  }

  // Appends a bool value, as my_field: true, only if value is true.
  void AppendBoolIfTrue(const char field_name[], bool value) {
    if (value) AppendBool(field_name, value);
  }

  // Appends a string value, like my_field: "abc123".
  void AppendString(const char field_name[], const string& value) {
    AppendFieldAndValue(
        field_name, StrCat("\"", ::tensorflow::str_util::CEscape(value), "\""));
  }

  // Appends a string value, like my_field: "abc123", but only if value is not
  // empty.
  void AppendStringIfNotEmpty(const char field_name[], const string& value) {
    if (!value.empty()) AppendString(field_name, value);
  }

  // Appends the string name of an enum, like my_field: FIRST_ENUM.
  void AppendEnumName(const char field_name[], const string& name) {
    AppendFieldAndValue(field_name, name);
  }

 private:
  void AppendFieldAndValue(const char field_name[], StringPiece value_text) {
    absl::StrAppend(output_, level_empty_ ? "" : field_separator_, indent_,
                    field_name, kColonSeparator, value_text);
    level_empty_ = false;
  }

  string* const output_;
  const bool short_debug_;
  const string field_separator_;
  string indent_;

  // False when at least one field has been output for the message at the
  // current deepest level of nesting.
  bool level_empty_ = true;

  TF_DISALLOW_COPY_AND_ASSIGN(ProtoTextOutput);
};

inline void ProtoSpaceAndComments(Scanner* scanner) {
  for (;;) {
    scanner->AnySpace();
    if (scanner->Peek() != '#') return;
    // Skip until newline.
    while (scanner->Peek('\n') != '\n') scanner->One(Scanner::ALL);
  }
}

// Parse the next numeric value from <scanner>, returning false if parsing
// failed.
template <typename T>
bool ProtoParseNumericFromScanner(Scanner* scanner, T* value) {
  StringPiece numeric_str;
  scanner->RestartCapture();
  if (!scanner->Many(Scanner::LETTER_DIGIT_DOT_PLUS_MINUS)
           .GetResult(nullptr, &numeric_str)) {
    return false;
  }

  // Special case to disallow multiple leading zeroes, to match proto parsing.
  int leading_zero = 0;
  for (size_t i = 0; i < numeric_str.size(); ++i) {
    const char ch = numeric_str[i];
    if (ch == '0') {
      if (++leading_zero > 1) return false;
    } else if (ch != '-') {
      break;
    }
  }

  ProtoSpaceAndComments(scanner);
  return SafeStringToNumeric<T>(numeric_str, value);
}

// Parse the next boolean value from <scanner>, returning false if parsing
// failed.
bool ProtoParseBoolFromScanner(Scanner* scanner, bool* value);

// Parse the next string literal from <scanner>, returning false if parsing
// failed.
bool ProtoParseStringLiteralFromScanner(Scanner* scanner, string* value);

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_STRINGS_PROTO_TEXT_UTIL_H_
