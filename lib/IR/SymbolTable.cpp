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
#include "llvm/ADT/SmallString.h"

using namespace mlir;

/// Build a symbol table with the symbols within the given operation.
SymbolTable::SymbolTable(Operation *op) : context(op->getContext()) {
  assert(op->hasTrait<OpTrait::SymbolTable>() &&
         "expected operation to have SymbolTable trait");
  assert(op->getNumRegions() == 1 &&
         "expected operation to have a single region");

  for (auto &block : op->getRegion(0)) {
    for (auto &op : block) {
      auto nameAttr = op.getAttrOfType<StringAttr>(getSymbolAttrName());
      if (!nameAttr)
        continue;

      auto inserted = symbolTable.insert({nameAttr.getValue(), &op});
      (void)inserted;
      assert(inserted.second &&
             "expected region to contain uniquely named symbol operations");
    }
  }
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Operation *SymbolTable::lookup(StringRef name) const {
  return symbolTable.lookup(name);
}

/// Erase the given symbol from the table.
void SymbolTable::erase(Operation *symbol) {
  auto nameAttr = symbol->getAttrOfType<StringAttr>(getSymbolAttrName());
  assert(nameAttr && "expected valid 'name' attribute");

  auto it = symbolTable.find(nameAttr.getValue());
  if (it != symbolTable.end() && it->second == symbol)
    symbolTable.erase(it);
}

/// Insert a new symbol into the table, and rename it as necessary to avoid
/// collisions.
void SymbolTable::insert(Operation *symbol) {
  auto nameAttr = symbol->getAttrOfType<StringAttr>(getSymbolAttrName());
  assert(nameAttr && "expected valid 'name' attribute");

  // Add this symbol to the symbol table, uniquing the name if a conflict is
  // detected.
  if (symbolTable.insert({nameAttr.getValue(), symbol}).second)
    return;

  // If a conflict was detected, then the symbol will not have been added to
  // the symbol table. Try suffixes until we get to a unique name that works.
  SmallString<128> nameBuffer(nameAttr.getValue());
  unsigned originalLength = nameBuffer.size();

  // Iteratively try suffixes until we find one that isn't used.
  do {
    nameBuffer.resize(originalLength);
    nameBuffer += '_';
    nameBuffer += std::to_string(uniquingCounter++);
  } while (!symbolTable.insert({nameBuffer, symbol}).second);
  symbol->setAttr(getSymbolAttrName(), StringAttr::get(nameBuffer, context));
}

/// Returns the operation registered with the given symbol name with the
/// regions of 'symbolTableOp'. 'symbolTableOp' is required to be an operation
/// with the 'OpTrait::SymbolTable' trait. Returns nullptr if no valid symbol
/// was found.
Operation *SymbolTable::lookupSymbolIn(Operation *symbolTableOp,
                                       StringRef symbol) {
  assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>());

  // Look for a symbol with the given name.
  for (auto &block : symbolTableOp->getRegion(0)) {
    for (auto &op : block) {
      auto nameAttr = op.template getAttrOfType<StringAttr>(
          mlir::SymbolTable::getSymbolAttrName());
      if (nameAttr && nameAttr.getValue() == symbol)
        return &op;
    }
  }
  return nullptr;
}

/// Returns the operation registered with the given symbol name within the
/// closes parent operation with the 'OpTrait::SymbolTable' trait. Returns
/// nullptr if no valid symbol was found.
Operation *SymbolTable::lookupNearestSymbolFrom(Operation *from,
                                                StringRef symbol) {
  while (from && !from->hasTrait<OpTrait::SymbolTable>())
    from = from->getParentOp();
  return from ? lookupSymbolIn(from, symbol) : nullptr;
}

//===----------------------------------------------------------------------===//
// SymbolTable Trait Types
//===----------------------------------------------------------------------===//

LogicalResult OpTrait::impl::verifySymbolTable(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitOpError()
           << "Operations with a 'SymbolTable' must have exactly one region";

  // Check that all symboles are uniquely named within child regions.
  llvm::StringMap<Location> nameToOrigLoc;
  for (auto &block : op->getRegion(0)) {
    for (auto &op : block) {
      // Check for a symbol name attribute.
      auto nameAttr =
          op.getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
      if (!nameAttr)
        continue;

      // Try to insert this symbol into the table.
      auto it = nameToOrigLoc.try_emplace(nameAttr.getValue(), op.getLoc());
      if (!it.second)
        return op.emitError()
            .append("redefinition of symbol named '", nameAttr.getValue(), "'")
            .attachNote(it.first->second)
            .append("see existing symbol definition here");
    }
  }
  return success();
}
