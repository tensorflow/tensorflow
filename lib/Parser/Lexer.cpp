//===- Lexer.cpp - MLIR Lexer Implementation ------------------------------===//
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
// This file implements the lexer for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;
using llvm::SMLoc;
using llvm::SourceMgr;

Lexer::Lexer(llvm::SourceMgr &sourceMgr) : sourceMgr(sourceMgr) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// emitError - Emit an error message and return an Token::error token.
Token Lexer::emitError(const char *loc, const Twine &message) {
  // TODO(clattner): If/when we want to implement a -verify mode, this will need
  // to package up errors into SMDiagnostic and report them.
  sourceMgr.PrintMessage(SMLoc::getFromPointer(loc), SourceMgr::DK_Error,
                         message);
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  const char *tokStart = curPtr;

  switch (*curPtr++) {
  default:
    // Handle bare identifiers.
    if (isalpha(curPtr[-1]))
      return lexBareIdentifierOrKeyword(tokStart);

    // Unknown character, emit an error.
    return emitError(tokStart, "unexpected character");

  case 0:
    // This may either be a nul character in the source file or may be the EOF
    // marker that llvm::MemoryBuffer guarantees will be there.
    if (curPtr-1 == curBuffer.end())
      return formToken(Token::eof, tokStart);

    LLVM_FALLTHROUGH;
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return lexToken();

  case ',': return formToken(Token::comma, tokStart);
  case '(': return formToken(Token::l_paren, tokStart);
  case ')': return formToken(Token::r_paren, tokStart);
  case '<': return formToken(Token::less, tokStart);
  case '>': return formToken(Token::greater, tokStart);

  case '-':
    if (*curPtr == '>') {
      ++curPtr;
      return formToken(Token::arrow, tokStart);
    }
    return emitError(tokStart, "unexpected character");

  case '?':
    if (*curPtr == '?') {
      ++curPtr;
      return formToken(Token::questionquestion, tokStart);
    }

    return formToken(Token::question, tokStart);

  case ';': return lexComment();
  case '@': return lexAtIdentifier(tokStart);

  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
    return lexNumber(tokStart);
  }
}

/// Lex a comment line, starting with a semicolon.
///
///   TODO: add a regex for comments here and to the spec.
///
Token Lexer::lexComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return lexToken();
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr-1 == curBuffer.end()) {
        --curPtr;
        return lexToken();
      }
      LLVM_FALLTHROUGH;
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a bare identifier or keyword that starts with a letter.
///
///   bare-id ::= letter (letter|digit)*
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z]*
  while (isalpha(*curPtr) || isdigit(*curPtr))
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr-tokStart);

  Token::TokenKind kind = llvm::StringSwitch<Token::TokenKind>(spelling)
    .Case("bf16",    Token::kw_bf16)
    .Case("cfgfunc", Token::kw_cfgfunc)
    .Case("extfunc", Token::kw_extfunc)
    .Case("f16", Token::kw_f16)
    .Case("f32", Token::kw_f32)
    .Case("f64", Token::kw_f64)
    .Case("i1", Token::kw_i1)
    .Case("i16", Token::kw_i16)
    .Case("i32", Token::kw_i32)
    .Case("i64", Token::kw_i64)
    .Case("i8", Token::kw_i8)
    .Case("int", Token::kw_int)
    .Case("memref", Token::kw_memref)
    .Case("mlfunc", Token::kw_mlfunc)
    .Case("tensor", Token::kw_tensor)
    .Case("vector", Token::kw_vector)
    .Default(Token::bare_identifier);

  return Token(kind, spelling);
}

/// Lex an '@foo' identifier.
///
///   function-id ::= `@` bare-id
///
Token Lexer::lexAtIdentifier(const char *tokStart) {
  // These always start with a letter.
  if (!isalpha(*curPtr++))
    return emitError(curPtr-1, "expected letter in @ identifier");

  while (isalpha(*curPtr) || isdigit(*curPtr))
    ++curPtr;
  return formToken(Token::at_identifier, tokStart);
}

/// Lex an integer literal.
///
///   integer-literal ::= digit+ | `0x` hex_digit+
///
Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  // Handle the hexadecimal case.
  if (curPtr[-1] == '0' && *curPtr == 'x') {
    ++curPtr;

    if (!isxdigit(*curPtr))
      return emitError(curPtr, "expected hexadecimal digit");

    while (isxdigit(*curPtr))
      ++curPtr;

    return formToken(Token::integer, tokStart);
  }

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(Token::integer, tokStart);
}
