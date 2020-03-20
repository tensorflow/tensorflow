/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_UTILS_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_UTILS_H_

#include <map>
#include <sstream>
#include <string>

namespace tflite {
namespace support {
namespace codegen {

/// Collects runtime error logs which could be showed later.
// TODO(b/150538286): Consider a better mechanism to simplify callsite code.
class ErrorReporter {
 public:
  int Warning(const char* format, ...);
  int Error(const char* format, ...);
  std::string GetMessage();

 private:
  int Report(const char* prefix, const char* format, va_list args);
  std::stringstream buffer_;
};

/// Implements basic code generating with text templates.
///
/// It could accept code templates and concatenate them into complete codes. A
/// template could contain named values.
///
/// Example code:
///   CodeWriter code;
///   code.SetValue("NAME", "Foo");
///   code.Append("void {{NAME}}() { printf("%s", "{{NAME}}"); }");
///   code.SetValue("NAME", "Bar");
///   code.Append("void {{NAME}}() { printf("%s", "{{NAME}}"); }");
///
/// Output:
///  void Foo() { printf("%s", "Foo"); }
///  void Bar() { printf("%s", "Bar"); }
class CodeWriter {
 public:
  explicit CodeWriter(ErrorReporter* err);
  /// Sets value to a token. When generating code with template, a string in a
  /// pair of {{ and }} will be regarded as a token and replaced with the
  /// corresponding value in code generation.
  /// It rewrites if the token already has a value.
  void SetTokenValue(const std::string& token, const std::string& value);

  /// Gets the current value set on the given token.
  const std::string GetTokenValue(const std::string& token) const;

  /// Sets the unit indent string. For example, in Java it should be "  ".
  void SetIndentString(const std::string& indent);

  /// Increases the indent by a unit (the string set in SetIndentString).
  void Indent();

  /// Decreases the indent by a unit (the string set in SetIndentString).
  void Outdent();

  /// Generates the indentation string.
  std::string GenerateIndent() const;

  /// Appends a piece of template codes to the stream. Every named value will be
  /// replaced via the real value. A new line will always be appended at the
  /// end.
  void Append(const std::string& text);

  /// Appends a piece of template codes to the stream. Same with `Append`, but a
  /// new line will not be appended at the end.
  void AppendNoNewLine(const std::string& text);

  /// Appends a new line to the stream.
  void NewLine();

  /// Deletes the last N charaters in the stream. If the stream has less than N
  /// characters, deletes all.
  void Backspace(int n);

  std::string ToString() const;

  /// Checks if the internal string stream is empty. Note: This method has
  // overhead.
  bool IsStreamEmpty() const;

  /// Clears all the internal string stream and value map.
  void Clear();

 private:
  void AppendInternal(const std::string& text, bool newline);

  std::string indent_str_;
  int indent_;

  std::map<std::string, std::string> value_map_;
  std::string buffer_;

  ErrorReporter* err_;
};

/// Converts foo_bar_name to fooBarName. It's callers duty to make sure given
/// string "s" is already in snake case; or unexpected behavior may occur.
std::string SnakeCaseToCamelCase(const std::string& s);

/// Joins 2 parts of file path into one, connected by unix path seperator '/'.
/// It's callers duty to ensure the two parts are valid.
std::string JoinPath(const std::string& a, const std::string& b);

}  // namespace codegen
}  // namespace support
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SUPPORT_CODEGEN_UTILS_H_
