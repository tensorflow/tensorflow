//===- Identifier.h - MLIR Identifier Class ---------------------*- C++ -*-===//
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

#ifndef MLIR_IR_IDENTIFIER_H
#define MLIR_IR_IDENTIFIER_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
  class MLIRContext;

/// This class represents a uniqued string owned by an MLIRContext.  Strings
/// represented by this type cannot contain nul characters, and may not have a
/// zero length.
///
/// This is a POD type with pointer size, so it should be passed around by
/// value.  The underlying data is owned by MLIRContext and is thus immortal for
/// almost all clients.
class Identifier {
public:
  /// Return an identifier for the specified string.
  static Identifier get(StringRef str, const MLIRContext *context);

  /// Return a StringRef for the string.
  StringRef str() const {
    return StringRef(pointer, size());
  }

  /// Return a pointer to the start of the string data.
  const char *data() const {
    return pointer;
  }

  /// Return the number of bytes in this string.
  unsigned size() const {
    return ::strlen(pointer);
  }

  /// Return true if this identifier is the specified string.
  bool is(StringRef string) const {
    return str().equals(string);
  }

  Identifier(const Identifier&) = default;
private:
  /// These are the bytes of the string, which is a nul terminated string.
  const char *pointer;

  Identifier(const char *pointer) : pointer(pointer) {}
  void operator=(Identifier&) = delete;
};

} // end namespace mlir

#endif
