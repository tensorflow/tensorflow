//===- Token.cpp - MLIR Token Implementation ------------------------------===//
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
// This file implements the Token class for the MLIR textual form.
//
//===----------------------------------------------------------------------===//

#include "Token.h"
using namespace mlir;
using llvm::SMLoc;
using llvm::SMRange;

SMLoc Token::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const {
  return SMRange(getLoc(), getEndLoc());
}

/// For an integer token, return its value as an unsigned.  If it doesn't fit,
/// return None.
Optional<unsigned> Token::getUnsignedIntegerValue() const {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  unsigned result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return None;
  return result;
}

/// For an integer token, return its value as a uint64_t.  If it doesn't fit,
/// return None.
Optional<uint64_t> Token::getUInt64IntegerValue() const {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  uint64_t result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return None;
  return result;
}

/// For a floatliteral, return its value as a double. Return None if the value
/// underflows or overflows.
Optional<double> Token::getFloatingPointValue() const {
  double result = 0;
  if (spelling.getAsDouble(result))
    return None;
  return result;
}

/// For an inttype token, return its bitwidth.
Optional<unsigned> Token::getIntTypeBitwidth() const {
  unsigned result = 0;
  if (spelling[1] == '0' ||
      spelling.drop_front().getAsInteger(10, result) ||
      // Arbitrary but large limit on bitwidth.
      result > 4096 || result == 0)
    return None;
  return result;
}

/// Given a 'string' token, return its value, including removing the quote
/// characters and unescaping the contents of the string.
std::string Token::getStringValue() const {
  // Start by dropping the quotes.
  // TODO: Un-escape the string here instead of passing through the raw content.
  return getSpelling().drop_front().drop_back().str();
}

/// Given a hash_identifier token like #123, try to parse the number out of
/// the identifier, returning None if it is a named identifier like #x or
/// if the integer doesn't fit.
Optional<unsigned> Token::getHashIdentifierNumber() const {
  assert(getKind() == hash_identifier);
  unsigned result = 0;
  if (spelling.drop_front().getAsInteger(10, result))
    return None;
  return result;
}

/// Given a punctuation or keyword token kind, return the spelling of the
/// token as a string.  Warning: This will abort on markers, identifiers and
/// literal tokens since they have no fixed spelling.
StringRef Token::getTokenSpelling(Kind kind) {
  switch (kind) {
  default: llvm_unreachable("This token kind has no fixed spelling");
#define TOK_PUNCTUATION(NAME, SPELLING) case NAME: return SPELLING;
#define TOK_OPERATOR(NAME, SPELLING) case NAME: return SPELLING;
#define TOK_KEYWORD(SPELLING) case kw_##SPELLING: return #SPELLING;
#include "TokenKinds.def"
  }
}

/// Return true if this is one of the keyword token kinds (e.g. kw_if).
bool Token::isKeyword() const {
  switch (kind) {
  default: return false;
#define TOK_KEYWORD(SPELLING) case kw_##SPELLING: return true;
#include "TokenKinds.def"
  }
}
