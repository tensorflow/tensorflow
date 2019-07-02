//===- SymbolTable.cpp - MLIR Symbol Table Class --------------------------===//
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

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Module.h"

using namespace mlir;

/// Build a symbol table with the symbols within the given module.
SymbolTable::SymbolTable(Module module) : context(module.getContext()) {
  for (auto func : module) {
    auto inserted = symbolTable.insert({func.getName(), func});
    (void)inserted;
    assert(inserted.second &&
           "expected module to contain uniquely named functions");
  }
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Function SymbolTable::lookup(StringRef name) const {
  return lookup(Identifier::get(name, context));
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Function SymbolTable::lookup(Identifier name) const {
  return symbolTable.lookup(name);
}

/// Erase the given symbol from the table.
void SymbolTable::erase(Function symbol) {
  auto it = symbolTable.find(symbol.getName());
  if (it != symbolTable.end() && it->second == symbol)
    symbolTable.erase(it);
}

/// Insert a new symbol into the table, and rename it as necessary to avoid
/// collisions.
void SymbolTable::insert(Function symbol) {
  // Add this symbol to the symbol table, uniquing the name if a conflict is
  // detected.
  if (symbolTable.insert({symbol.getName(), symbol}).second)
    return;

  // If a conflict was detected, then the function will not have been added to
  // the symbol table.  Try suffixes until we get to a unique name that works.
  SmallString<128> nameBuffer(symbol.getName());
  unsigned originalLength = nameBuffer.size();

  // Iteratively try suffixes until we find one that isn't used.  We use a
  // module level uniquing counter to avoid N^2 behavior.
  do {
    nameBuffer.resize(originalLength);
    nameBuffer += '_';
    nameBuffer += std::to_string(uniquingCounter++);
    symbol.setName(Identifier::get(nameBuffer, context));
  } while (!symbolTable.insert({symbol.getName(), symbol}).second);
}
