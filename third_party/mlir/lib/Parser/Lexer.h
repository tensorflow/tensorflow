//===- Lexer.h - MLIR Lexer Interface ---------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file declares the MLIR Lexer class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_PARSER_LEXER_H
#define MLIR_LIB_PARSER_LEXER_H

#include "Token.h"
#include "mlir/Parser.h"

namespace mlir {
class Location;

/// This class breaks up the current file into a token stream.
class Lexer {
public:
  explicit Lexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() { return sourceMgr; }

  Token lexToken();

  /// Encode the specified source location information into a Location object
  /// for attachment to the IR or error reporting.
  Location getEncodedSourceLocation(llvm::SMLoc loc);

  /// Change the position of the lexer cursor.  The next token we lex will start
  /// at the designated point in the input.
  void resetPointer(const char *newPointer) { curPtr = newPointer; }

  /// Returns the start of the buffer.
  const char *getBufferBegin() { return curBuffer.data(); }

private:
  // Helpers.
  Token formToken(Token::Kind kind, const char *tokStart) {
    return Token(kind, StringRef(tokStart, curPtr - tokStart));
  }

  Token emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  Token lexAtIdentifier(const char *tokStart);
  Token lexBareIdentifierOrKeyword(const char *tokStart);
  Token lexEllipsis(const char *tokStart);
  Token lexNumber(const char *tokStart);
  Token lexPrefixedIdentifier(const char *tokStart);
  Token lexString(const char *tokStart);

  /// Skip a comment line, starting with a '//'.
  void skipComment();

  const llvm::SourceMgr &sourceMgr;
  MLIRContext *context;

  StringRef curBuffer;
  const char *curPtr;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
};

} // end namespace mlir

#endif // MLIR_LIB_PARSER_LEXER_H
