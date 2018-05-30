/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/scanner.h"

#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace {

// Returns true if c can be the first character in an identifier.
bool IsIdentifierFirst(int c) { return std::isalpha(c) || c == '_'; }

// Returns true if c can be the non-first character in an identifier.
bool IsIdentifierLater(int c) { return std::isalnum(c) || c == '_'; }

// Returns true if str is an identifier.
bool IsIdentifier(tensorflow::StringPiece str) {
  if (str.empty() || !IsIdentifierFirst(str[0])) {
    return false;
  }
  for (int64 i = 1; i < str.size(); ++i) {
    if (!IsIdentifierLater(str[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

Scanner::Scanner(tensorflow::StringPiece input) : input_(input), position_(0) {}

bool Scanner::ok() const { return status().ok(); }

const Status& Scanner::status() const { return status_; }

bool Scanner::Match(tensorflow::StringPiece match) {
  SkipWhitespace();
  if (ok() && position_ + match.size() <= input_.size() &&
      std::equal(match.begin(), match.end(), input_.begin() + position_)) {
    SkipChars(match.size());

    VLOG(10) << "Matched \"" << match << "\"";
    return true;
  } else {
    return false;
  }
}

void Scanner::Expect(tensorflow::StringPiece expect) {
  if (!Match(expect)) {
    SetError(tensorflow::strings::StrCat("Expected \"", expect, "\"."));
  }
}

bool Scanner::MatchReadIdentifier(string* identifier) {
  SkipWhitespace();
  if (!IsIdentifierFirst(PeekChar())) {
    return false;
  }
  identifier->clear();
  do {
    *identifier += ReadChar();
  } while (IsIdentifierLater(PeekChar()));

  VLOG(10) << "Read identifier " << identifier;
  CHECK(IsIdentifier(*identifier));
  return true;
}

string Scanner::ReadIdentifier() {
  string identifier;
  if (!MatchReadIdentifier(&identifier)) {
    SetError("Expected identifier.");
  }
  return identifier;
}

void Scanner::ExpectIdentifier(tensorflow::StringPiece expect) {
  CHECK(IsIdentifier(expect));

  string identifier;
  if (!MatchReadIdentifier(&identifier)) {
    SetError(tensorflow::strings::StrCat("Expected identifier ", expect, "."));
  }
  if (identifier != expect) {
    SetError(tensorflow::strings::StrCat("Expected identifier ", expect,
                                         ", but got ", identifier, "."));
  }
}

// Matches the end of the input, also known as End Of File (EOF).
bool Scanner::MatchEof() {
  SkipWhitespace();
  return PeekChar() == EOF;
}

void Scanner::ExpectEof() {
  if (!MatchEof()) {
    SetError("Expected end of input.");
  }
}

// Reads a vector of the format "(1, 2, 3)".
std::vector<int64> Scanner::ReadIntVector() {
  std::vector<int64> ints;
  Expect("(");
  if (!Match(")") && ok()) {
    ints.push_back(ReadInt());
    while (Match(",")) {
      ints.push_back(ReadInt());
    }
    Expect(")");
  }

  VLOG(10) << "Read int vector with " << ints.size() << " elements.";
  return ints;
}

int64 Scanner::ReadInt() {
  bool negative = Match("-");
  if (!PeekDigit()) {
    SetError("Expected integer.");
    return 0;
  }

  int64 integer = 0;
  do {
    integer = (ReadChar() - '0') + integer * 10;
  } while (PeekDigit());
  integer = negative ? -integer : integer;

  VLOG(10) << "Read integer " << integer;
  return integer;
}

void Scanner::SkipWhitespace() {
  while (PeekWhitespace()) {
    SkipChars(1);
  }
}

int Scanner::ReadChar() {
  int c = PeekChar();
  SkipChars(1);

  VLOG(20) << "Read char " << c;
  return c;
}

int Scanner::PeekChar() const {
  return ok() && position_ < input_.size() ? input_[position_] : EOF;
}

bool Scanner::PeekDigit() const {
  // Do not use std::isdigit since it depends on the locale and we do not
  // handle any digits beyond 0-9.
  const char c = PeekChar();
  return '0' <= c && c <= '9';
}

bool Scanner::PeekAlnum() const { return std::isalnum(PeekChar()); }

bool Scanner::PeekWhitespace() const { return std::isspace(PeekChar()); }

void Scanner::SkipChars(int64 count) {
  CHECK_GE(count, 0);
  position_ += count;
}

void Scanner::SetError(string error_message) {
  // Only the first error is recorded since any later errors will likely be a
  // consequence of the first error.
  if (ok()) {
    status_ = InvalidArgumentStrCat(std::move(error_message));
    position_ = input_.size();
    VLOG(10) << "Failed scanner with error " << status_.ToString();
  } else {
    VLOG(10) << "Error on already failed scanner is " << error_message;
  }
}

}  // namespace xla
