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
#include "tensorflow/lite/experimental/support/codegen/utils.h"

#include <cstdarg>

namespace tflite {
namespace support {
namespace codegen {

int ErrorReporter::Warning(const char* format, ...) {
  va_list args;
  va_start(args, format);
  return Report("[WARN] ", format, args);
}

int ErrorReporter::Error(const char* format, ...) {
  va_list args;
  va_start(args, format);
  return Report("[ERROR] ", format, args);
}

int ErrorReporter::Report(const char* prefix, const char* format,
                          va_list args) {
  char buf[1024];
  int formatted = vsnprintf(buf, sizeof(buf), format, args);
  buffer_ << prefix << buf << std::endl;
  return formatted;
}

std::string ErrorReporter::GetMessage() {
  std::string value = buffer_.str();
  buffer_.str("");
  return value;
}

CodeWriter::CodeWriter(ErrorReporter* err) : indent_(0), err_(err) {}

void CodeWriter::SetTokenValue(const std::string& token,
                               const std::string& value) {
  value_map_[token] = value;
}

const std::string CodeWriter::GetTokenValue(const std::string& token) const {
  auto iter = value_map_.find(token);
  if (iter == value_map_.end()) {
    // Typically only Code Generator's call this function (or `Append`). It's
    // their duty to make sure the token is valid, and requesting for an invalid
    // token implicits flaws in the code generation logic.
    err_->Error("Internal: Cannot find value with token '%s'", token.c_str());
    return "";
  }
  return iter->second;
}

void CodeWriter::SetIndentString(const std::string& indent_str) {
  indent_str_ = indent_str;
}

void CodeWriter::Indent() { indent_++; }

void CodeWriter::Outdent() { indent_--; }

std::string CodeWriter::GenerateIndent() const {
  std::string res;
  res.reserve(indent_str_.size() * indent_);
  for (int i = 0; i < indent_; i++) {
    res.append(indent_str_);
  }
  return res;
}

void CodeWriter::Append(const std::string& text) { AppendInternal(text, true); }

void CodeWriter::AppendNoNewLine(const std::string& text) {
  AppendInternal(text, false);
}

void CodeWriter::AppendInternal(const std::string& text, bool newline) {
  // Prefix indent
  if ((buffer_.empty()             // nothing in the buffer
       || buffer_.back() == '\n')  // is on new line
      // is writing on current line
      && (!text.empty() && text[0] != '\n' && text[0] != '\r')) {
    buffer_.append(GenerateIndent());
  }
  // State machine variables
  bool in_token = false;
  int i = 0;
  // Rough memory reserve
  buffer_.reserve(buffer_.size() + text.size());
  std::string token_buffer;
  // A simple LL1 analysis
  while (i < text.size()) {
    char cur = text[i];
    char cur_next = i == text.size() - 1 ? '\0' : text[i + 1];  // Set guardian
    if (in_token == false) {
      if (cur == '{' && cur_next == '{') {  // Enter token
        in_token = true;
        i += 2;
      } else if (cur == '\n') {  // We need to apply global indent here
        buffer_.push_back(cur);
        if (cur_next != '\0' && cur_next != '\n' && cur_next != '\r') {
          buffer_.append(GenerateIndent());
        }
        i += 1;
      } else {
        buffer_.push_back(cur);
        i += 1;
      }
    } else {
      if (cur == '}' && cur_next == '}') {  // Close token
        in_token = false;
        const auto value = GetTokenValue(token_buffer);
        buffer_.append(value);
        token_buffer.clear();
        i += 2;
      } else {
        token_buffer.push_back(cur);
        i += 1;
      }
    }
  }
  if (!token_buffer.empty()) {
    // Typically only Code Generator's call this function. It's
    // their duty to make sure the code (or template) has valid syntax, and
    // unclosed "{{...}}" implicits severe error in the template.
    err_->Error("Internal: Invalid template: {{token}} is not closed.");
  }
  if (newline) {
    buffer_.push_back('\n');
  }
}

void CodeWriter::NewLine() { Append(""); }

void CodeWriter::Backspace(int n) {
  buffer_.resize(buffer_.size() > n ? buffer_.size() - n : 0);
}

std::string CodeWriter::ToString() const { return buffer_; }

bool CodeWriter::IsStreamEmpty() const { return buffer_.empty(); }

void CodeWriter::Clear() {
  buffer_.clear();
  value_map_.clear();
  indent_ = 0;
}

std::string SnakeCaseToCamelCase(const std::string& s) {
  std::string t;
  t.reserve(s.length());
  size_t i = 0;
  // Note: Use simple string += for simplicity.
  bool cap = false;
  while (i < s.size()) {
    const char c = s[i++];
    if (c == '_') {
      cap = true;
    } else if (cap) {
      t += toupper(c);
      cap = false;
    } else {
      t += c;
    }
  }
  return t;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  std::string a_fixed = a;
  if (!a_fixed.empty() && a_fixed.back() == '/') a_fixed.pop_back();
  std::string b_fixed = b;
  if (!b_fixed.empty() && b_fixed.front() == '/') b_fixed.erase(0, 1);
  return a_fixed + "/" + b_fixed;
}

}  // namespace codegen
}  // namespace support
}  // namespace tflite
