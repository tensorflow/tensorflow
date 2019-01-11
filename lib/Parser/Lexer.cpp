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
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/SourceMgr.h"
using namespace mlir;

using llvm::SMLoc;
using llvm::SourceMgr;

// Returns true if 'c' is an allowable puncuation character: [$._-]
// Returns false otherwise.
static bool isPunct(char c) {
  return c == '$' || c == '.' || c == '_' || c == '-';
}

Lexer::Lexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// Encode the specified source location information into an attribute for
/// attachment to the IR.
Location Lexer::getEncodedSourceLocation(llvm::SMLoc loc) {
  auto &sourceMgr = getSourceMgr();
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  auto *buffer = sourceMgr.getMemoryBuffer(mainFileID);
  auto filename = UniquedFilename::get(buffer->getBufferIdentifier(), context);

  return FileLineColLoc::get(filename, lineAndColumn.first,
                             lineAndColumn.second, context);
}

/// emitError - Emit an error message and return an Token::error token.
Token Lexer::emitError(const char *loc, const Twine &message) {
  context->emitError(getEncodedSourceLocation(SMLoc::getFromPointer(loc)),
                     message);
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  // Ignore whitespace.
  while (curPtr) {
    switch (*curPtr) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
      ++curPtr;
      continue;
    default:
      break;
    }
    break;
  }

  const char *tokStart = curPtr;
  switch (*curPtr++) {
  default:
    // Handle bare identifiers.
    if (isalpha(curPtr[-1]))
      return lexBareIdentifierOrKeyword(tokStart);

    // Unknown character, emit an error.
    return emitError(tokStart, "unexpected character");

  case '_':
    // Handle bare identifiers.
    return lexBareIdentifierOrKeyword(tokStart);

  case 0:
    // This may either be a nul character in the source file or may be the EOF
    // marker that llvm::MemoryBuffer guarantees will be there.
    if (curPtr-1 == curBuffer.end())
      return formToken(Token::eof, tokStart);

    LLVM_FALLTHROUGH;
  case ':': return formToken(Token::colon, tokStart);
  case ',': return formToken(Token::comma, tokStart);
  case '(': return formToken(Token::l_paren, tokStart);
  case ')': return formToken(Token::r_paren, tokStart);
  case '{': return formToken(Token::l_brace, tokStart);
  case '}': return formToken(Token::r_brace, tokStart);
  case '[':
    return formToken(Token::l_square, tokStart);
  case ']':
    return formToken(Token::r_square, tokStart);
  case '<': return formToken(Token::less, tokStart);
  case '>': return formToken(Token::greater, tokStart);
  case '=': return formToken(Token::equal, tokStart);

  case '+': return formToken(Token::plus, tokStart);
  case '*': return formToken(Token::star, tokStart);
  case '-':
    if (*curPtr == '>') {
      ++curPtr;
      return formToken(Token::arrow, tokStart);
    }
    return formToken(Token::minus, tokStart);

  case '?':
    return formToken(Token::question, tokStart);

  case '/':
    if (*curPtr == '/')
      return lexComment();
    return emitError(tokStart, "unexpected character");

  case '@':
    return lexAtIdentifier(tokStart);

  case '!':
    LLVM_FALLTHROUGH;
  case '^':
    LLVM_FALLTHROUGH;
  case '#':
    LLVM_FALLTHROUGH;
  case '%':
    return lexPrefixedIdentifier(tokStart);
  case '"': return lexString(tokStart);

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
  // Advance over the second '/' in a '//' comment.
  assert(*curPtr == '/');
  ++curPtr;

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
///   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
///   integer-type ::= `i[1-9][0-9]*`
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_.$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '.')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr-tokStart);

  // Check for i123.
  if (tokStart[0] == 'i') {
    bool allDigits = true;
    for (auto c : spelling.drop_front())
      allDigits &= isdigit(c) != 0;
    if (allDigits && spelling.size() != 1)
      return Token(Token::inttype, spelling);
  }

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) \
    .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
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

  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_')
    ++curPtr;
  return formToken(Token::at_identifier, tokStart);
}

/// Lex an identifier that starts with a prefix followed by suffix-id.
///
///   affine-map-id ::= `#` suffix-id
///   ssa-id        ::= '%' suffix-id
///   block-id      ::= '^' suffix-id
///   type-id       ::= '!' suffix-id
///   suffix-id     ::= digit+ | (letter|id-punct) (letter|id-punct|digit)*
///
Token Lexer::lexPrefixedIdentifier(const char *tokStart) {
  Token::Kind kind;
  StringRef errorKind;
  switch (*tokStart) {
  case '#':
    kind = Token::hash_identifier;
    errorKind = "invalid affine map name";
    break;
  case '%':
    kind = Token::percent_identifier;
    errorKind = "invalid SSA name";
    break;
  case '^':
    kind = Token::caret_identifier;
    errorKind = "invalid block name";
    break;
  case '!':
    kind = Token::exclamation_identifier;
    errorKind = "invalid type identifier";
    break;
  default:
    llvm_unreachable("invalid caller");
  }

  // Parse suffix-id.
  if (isdigit(*curPtr)) {
    // If suffix-id starts with a digit, the rest must be digits.
    while (isdigit(*curPtr)) {
      ++curPtr;
    }
  } else if (isalpha(*curPtr) || isPunct(*curPtr)) {
    do  {
      ++curPtr;
    } while (isalpha(*curPtr) || isdigit(*curPtr) || isPunct(*curPtr));
  } else {
    return emitError(curPtr - 1, errorKind);
  }

  return formToken(kind, tokStart);
}

/// Lex a number literal.
///
///   integer-literal ::= digit+ | `0x` hex_digit+
///   float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
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

  if (*curPtr != '.')
    return formToken(Token::integer, tokStart);
  ++curPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(*curPtr)) ++curPtr;

  if (*curPtr == 'e' || *curPtr == 'E') {
    if (isdigit(static_cast<unsigned char>(curPtr[1])) ||
        ((curPtr[1] == '-' || curPtr[1] == '+') &&
         isdigit(static_cast<unsigned char>(curPtr[2])))) {
      curPtr += 2;
      while (isdigit(*curPtr)) ++curPtr;
    }
  }
  return formToken(Token::floatliteral, tokStart);
}

/// Lex a string literal.
///
///   string-literal ::= '"' [^"\n\f\v\r]* '"'
///
/// TODO: define escaping rules.
Token Lexer::lexString(const char *tokStart) {
  assert(curPtr[-1] == '"');

  while (1) {
    switch (*curPtr++) {
    case '"':
      return formToken(Token::string, tokStart);
    case 0:
      // If this is a random nul character in the middle of a string, just
      // include it.  If it is the end of file, then it is an error.
      if (curPtr-1 != curBuffer.end())
        continue;
      LLVM_FALLTHROUGH;
    case '\n':
    case '\v':
    case '\f':
      return emitError(curPtr-1, "expected '\"' in string literal");
    case '\\':
      // Handle explicitly a few escapes.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1]))
        // Support \xx for two hex digits.
        curPtr += 2;
      else
        return emitError(curPtr - 1, "unknown escape in string literal");
      continue;

    default:
      continue;
    }
  }
}
