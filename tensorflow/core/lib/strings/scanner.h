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

#ifndef TENSORFLOW_LIB_STRINGS_SCANNER_H_
#define TENSORFLOW_LIB_STRINGS_SCANNER_H_

#include <string>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace strings {

// Scanner provides simplified string parsing, in which a string is parsed as a
// series of scanning calls (e.g. One, Any, Many, OneLiteral, Eos), and then
// finally GetResult is called. If GetResult returns true, then it also returns
// the remaining characters and any captured substring.
//
// The range to capture can be controlled with RestartCapture and StopCapture;
// by default, all processed characters are captured.
class Scanner {
 public:
  // Classes of characters. Each enum name is to be read as the union of the
  // parts - e.g., class LETTER_DIGIT means the class includes all letters and
  // all digits.
  //
  // LETTER means ascii letter a-zA-Z.
  // DIGIT means ascii digit: 0-9.
  enum CharClass {
    // NOTE: When adding a new CharClass, update the AllCharClasses ScannerTest
    // in scanner_test.cc
    ALL,
    DIGIT,
    LETTER,
    LETTER_DIGIT,
    LETTER_DIGIT_DASH_UNDERSCORE,
    LETTER_DIGIT_DASH_DOT_SLASH,             // SLASH is / only, not backslash
    LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE,  // SLASH is / only, not backslash
    LETTER_DIGIT_DOT,
    LETTER_DIGIT_DOT_PLUS_MINUS,
    LETTER_DIGIT_DOT_UNDERSCORE,
    LETTER_DIGIT_UNDERSCORE,
    LOWERLETTER,
    LOWERLETTER_DIGIT,
    LOWERLETTER_DIGIT_UNDERSCORE,
    NON_ZERO_DIGIT,
    SPACE,
    UPPERLETTER,
  };

  explicit Scanner(StringPiece source) : cur_(source) { RestartCapture(); }

  // Consume the next character of the given class from input. If the next
  // character is not in the class, then GetResult will ultimately return false.
  Scanner& One(CharClass clz) {
    if (cur_.empty() || !Matches(clz, cur_[0])) {
      return Error();
    }
    cur_.remove_prefix(1);
    return *this;
  }

  // Consume the next s.size() characters of the input, if they match <s>. If
  // they don't match <s>, this is a no-op.
  Scanner& ZeroOrOneLiteral(StringPiece s) {
    cur_.Consume(s);
    return *this;
  }

  // Consume the next s.size() characters of the input, if they match <s>. If
  // they don't match <s>, then GetResult will ultimately return false.
  Scanner& OneLiteral(StringPiece s) {
    if (!cur_.Consume(s)) {
      error_ = true;
    }
    return *this;
  }

  // Consume characters from the input as long as they match <clz>.
  Scanner& Any(CharClass clz) {
    while (!cur_.empty() && Matches(clz, cur_[0])) {
      cur_.remove_prefix(1);
    }
    return *this;
  }

  // Shorthand for One(clz).Any(clz).
  Scanner& Many(CharClass clz) { return One(clz).Any(clz); }

  // Reset the capture start point.
  //
  // Later, when GetResult is called and if it returns true, the capture
  // returned will start at the position at the time this was called.
  Scanner& RestartCapture() {
    capture_start_ = cur_.data();
    capture_end_ = nullptr;
    return *this;
  }

  // Stop capturing input.
  //
  // Later, when GetResult is called and if it returns true, the capture
  // returned will end at the position at the time this was called.
  Scanner& StopCapture() {
    capture_end_ = cur_.data();
    return *this;
  }

  // If not at the input of input, then GetResult will ultimately return false.
  Scanner& Eos() {
    if (!cur_.empty()) error_ = true;
    return *this;
  }

  // Shorthand for Any(SPACE).
  Scanner& AnySpace() { return Any(SPACE); }

  // This scans input until <end_ch> is reached. <end_ch> is NOT consumed.
  Scanner& ScanUntil(char end_ch) {
    ScanUntilImpl(end_ch, false);
    return *this;
  }

  // This scans input until <end_ch> is reached. <end_ch> is NOT consumed.
  // Backslash escape sequences are skipped.
  // Used for implementing quoted string scanning.
  Scanner& ScanEscapedUntil(char end_ch) {
    ScanUntilImpl(end_ch, true);
    return *this;
  }

  // Return the next character that will be scanned, or <default_value> if there
  // are no more characters to scan.
  // Note that if a scan operation has failed (so GetResult() returns false),
  // then the value of Peek may or may not have advanced since the scan
  // operation that failed.
  char Peek(char default_value = '\0') const {
    return cur_.empty() ? default_value : cur_[0];
  }

  // Returns false if there are no remaining characters to consume.
  int empty() const { return cur_.empty(); }

  // Returns true if the input string successfully matched. When true is
  // returned, the remaining string is returned in <remaining> and the captured
  // string returned in <capture>, if non-NULL.
  bool GetResult(StringPiece* remaining = nullptr,
                 StringPiece* capture = nullptr);

 private:
  void ScanUntilImpl(char end_ch, bool escaped);

  Scanner& Error() {
    error_ = true;
    return *this;
  }

  static bool IsLetter(char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
  }

  static bool IsLowerLetter(char ch) { return ch >= 'a' && ch <= 'z'; }

  static bool IsDigit(char ch) { return ch >= '0' && ch <= '9'; }

  static bool IsSpace(char ch) {
    return (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' ||
            ch == '\r');
  }

  static bool Matches(CharClass clz, char ch) {
    switch (clz) {
      case ALL:
        return true;
      case DIGIT:
        return IsDigit(ch);
      case LETTER:
        return IsLetter(ch);
      case LETTER_DIGIT:
        return IsLetter(ch) || IsDigit(ch);
      case LETTER_DIGIT_DASH_UNDERSCORE:
        return (IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '_');
      case LETTER_DIGIT_DASH_DOT_SLASH:
        return IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '.' ||
               ch == '/';
      case LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE:
        return (IsLetter(ch) || IsDigit(ch) || ch == '-' || ch == '.' ||
                ch == '/' || ch == '_');
      case LETTER_DIGIT_DOT:
        return IsLetter(ch) || IsDigit(ch) || ch == '.';
      case LETTER_DIGIT_DOT_PLUS_MINUS:
        return IsLetter(ch) || IsDigit(ch) || ch == '+' || ch == '-' ||
               ch == '.';
      case LETTER_DIGIT_DOT_UNDERSCORE:
        return IsLetter(ch) || IsDigit(ch) || ch == '.' || ch == '_';
      case LETTER_DIGIT_UNDERSCORE:
        return IsLetter(ch) || IsDigit(ch) || ch == '_';
      case LOWERLETTER:
        return ch >= 'a' && ch <= 'z';
      case LOWERLETTER_DIGIT:
        return IsLowerLetter(ch) || IsDigit(ch);
      case LOWERLETTER_DIGIT_UNDERSCORE:
        return IsLowerLetter(ch) || IsDigit(ch) || ch == '_';
      case NON_ZERO_DIGIT:
        return IsDigit(ch) && ch != '0';
      case SPACE:
        return IsSpace(ch);
      case UPPERLETTER:
        return ch >= 'A' && ch <= 'Z';
    }
    return false;
  }

  StringPiece cur_;
  const char* capture_start_ = nullptr;
  const char* capture_end_ = nullptr;
  bool error_ = false;

  friend class ScannerTest;
  TF_DISALLOW_COPY_AND_ASSIGN(Scanner);
};

}  // namespace strings
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_STRINGS_SCANNER_H_
