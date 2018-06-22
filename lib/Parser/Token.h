//===- Token.h - MLIR Token Interface ---------------------------*- C++ -*-===//
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

#ifndef MLIR_LIB_PARSER_TOKEN_H
#define MLIR_LIB_PARSER_TOKEN_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace mlir {

/// This represents a token in the MLIR syntax.
class Token {
public:
  enum TokenKind {
    // Markers
    eof, error,

    // Identifiers.
    bare_identifier,    // foo
    at_identifier,      // @foo
    // TODO: @@foo, etc.

    // Punctuation.
    l_paren, r_paren,   // ( )
    less, greater,      // < >
    // TODO: More punctuation.

    // Keywords.
    kw_cfgfunc,
    kw_extfunc,
    kw_mlfunc,
    // TODO: More keywords.
  };

  Token(TokenKind kind, StringRef spelling)
    : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  TokenKind getKind() const { return kind; }
  bool is(TokenKind K) const { return kind == K; }

  bool isAny(TokenKind k1, TokenKind k2) const {
    return is(k1) || is(k2);
  }

  /// Return true if this token is one of the specified kinds.
  template <typename ...T>
  bool isAny(TokenKind k1, TokenKind k2, TokenKind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(TokenKind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename ...T>
  bool isNot(TokenKind k1, TokenKind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }


  /// Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  TokenKind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

} // end namespace mlir

#endif  // MLIR_LIB_PARSER_TOKEN_H
