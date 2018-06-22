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
#include "llvm/Support/raw_ostream.h"

/// For an integer token, return its value as an unsigned.  If it doesn't fit,
/// return None.
Optional<unsigned> Token::getUnsignedIntegerValue() {
  bool isHex = spelling.size() > 1 && spelling[1] == 'x';

  unsigned result = 0;
  if (spelling.getAsInteger(isHex ? 0 : 10, result))
    return None;
  return result;
}
