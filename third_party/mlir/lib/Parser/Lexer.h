//===- Lexer.h - MLIR Lexer Interface ---------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
