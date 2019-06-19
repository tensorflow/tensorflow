//===- SymbolTable.h - MLIR Symbol Table Class ------------------*- C++ -*-===//
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

#ifndef MLIR_IR_SYMBOLTABLE_H
#define MLIR_IR_SYMBOLTABLE_H

#include "mlir/IR/Identifier.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class Function;
class MLIRContext;

/// This class represents the symbol table used by a module for function
/// symbols.
class SymbolTable {
public:
  SymbolTable(MLIRContext *ctx) : context(ctx) {}

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Function *lookup(StringRef name) const;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Function *lookup(Identifier name) const;

  /// Erase the given symbol from the table.
  void erase(Function *symbol);

  /// Insert a new symbol into the table, and rename it as necessary to avoid
  /// collisions.
  void insert(Function *symbol);

  /// Returns the context held by this symbol table.
  MLIRContext *getContext() const { return context; }

private:
  MLIRContext *context;

  /// This is a mapping from a name to the function with that name.
  llvm::DenseMap<Identifier, Function *> symbolTable;

  /// This is used when name conflicts are detected.
  unsigned uniquingCounter = 0;
};

} // end namespace mlir

#endif // MLIR_IR_SYMBOLTABLE_H
