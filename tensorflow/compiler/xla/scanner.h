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

#ifndef TENSORFLOW_COMPILER_XLA_SCANNER_H_
#define TENSORFLOW_COMPILER_XLA_SCANNER_H_

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace xla {

// Simple class for parsing data. The concepts for the interface are:
//
//   Match(x): Returns true if x is next in the input and in that case skips
//     past it. Otherwise returns false.
//
//   Expect(x): As Match(x), but requires x to be next in the input.
//
//   MatchReadX(x): Returns true if an X is next in the input and in that case
//     skips past it and assigns it to x. Otherwise returns false.
//
//   ReadX(): As ReadMatchX(), but requires an X to be next in the input and
//     returns it.
//
//   PeekX(): Returns true if an X is next in the input and does not skip
//     past it either way.
//
// All of these, except those that work on individual characters, skip
// whitespace.
//
// If a requirement is not met, the error is available in status(). A Scanner
// with a failed status() will behave as though the rest of the input is EOF and
// will not record further errors after that point.
class Scanner {
 public:
  Scanner(tensorflow::StringPiece input);

  bool ok() const;
  const Status& status() const;

  bool Match(tensorflow::StringPiece match);
  void Expect(tensorflow::StringPiece expect);

  // Match-reads an identifier. An identifier starts with an alphabetic
  // character or an underscore followed by any number of characters that are
  // each alphanumeric or underscore.
  bool MatchReadIdentifier(string* identifier);

  string ReadIdentifier();

  void ExpectIdentifier(tensorflow::StringPiece expect);

  // Matches the end of the input, also known as End Of File (EOF).
  bool MatchEof();
  void ExpectEof();

  // Reads a vector of the format "(1, 4, 5)".
  std::vector<int64> ReadIntVector();

  // Reads an integer. Can start with a minus but not a plus.
  int64 ReadInt();

  // Keeps skipping until encountering a non-whitespace character.
  void SkipWhitespace();

  // *** Below here are character-level methods that do not skip whitespace.

  int ReadChar();
  int PeekChar() const;
  bool PeekDigit() const;
  bool PeekAlnum() const;
  bool PeekWhitespace() const;

  // Skip past the next count characters.
  void SkipChars(int64 count);

 private:
  // Sets a failed status. The input is in effect replaced with EOF after
  // this. Only the first error is recorded.
  void SetError(string error_message);

  const tensorflow::StringPiece input_;
  int64 position_;
  Status status_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SCANNER_H_
