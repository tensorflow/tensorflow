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

/// Return true if the given operation is unknown and may potentially define a
/// symbol table.
static bool isPotentiallyUnknownSymbolTable(Operation *op) {
  return !op->getDialect() && op->getNumRegions() == 1;
}

//===----------------------------------------------------------------------===//
// SymbolTable
//===----------------------------------------------------------------------===//

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
  assert(from && "expected valid operation");
  while (!from->hasTrait<OpTrait::SymbolTable>()) {
    from = from->getParentOp();

    // Check that this is a valid op and isn't an unknown symbol table.
    if (!from || isPotentiallyUnknownSymbolTable(from))
      return nullptr;
  }
  return lookupSymbolIn(from, symbol);
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

//===----------------------------------------------------------------------===//
// SymbolTable Trait Types
//===----------------------------------------------------------------------===//

/// A utility result for walking a nested attribute for symbol uses.
enum HandlerResult {
  /// The walk of the containter can continue.
  Continue = 0,
  /// The walk should recurse into the given attribute, as it is a container.
  RecurseNestedAttribute,
  /// The walk should end immediately, as an interrupt has been signaled.
  Interrupt
};

/// Utility function used to handle a nested attribute during a walk of symbol
/// uses. It returns the above HandlerResult signaling the next action for the
/// walk.
HandlerResult handleAttrDuringSymbolWalk(
    Operation *op, Attribute attr,
    SmallVectorImpl<std::pair<Attribute, unsigned>> &worklist,
    function_ref<WalkResult(SymbolTable::SymbolUse)> callback) {
  switch (attr.getKind()) {
  /// Check for a nested container attribute, these will also need to be
  /// walked.
  case StandardAttributes::Array:
  case StandardAttributes::Dictionary: {
    worklist.push_back({attr, /*index*/ 0});
    return HandlerResult::RecurseNestedAttribute;
  }

  // Invoke the provided callback if we find a symbol use and check for a
  // requested interrupt.
  case StandardAttributes::SymbolRef: {
    SymbolTable::SymbolUse use(op, attr.cast<SymbolRefAttr>());
    return callback(use).wasInterrupted() ? HandlerResult::Interrupt
                                          : HandlerResult::Continue;
  }
  default:
    return HandlerResult::Continue;
  }
}

/// Walk all of the symbol references within the given operation, invoking the
/// provided callback for each found use.
static WalkResult
walkSymbolRefs(Operation *op,
               function_ref<WalkResult(SymbolTable::SymbolUse)> callback) {
  // Check to see if the operation has any attributes.
  DictionaryAttr attrDict = op->getAttrList().getDictionary();
  if (!attrDict)
    return WalkResult::advance();

  // A worklist of a container attribute and the current index into the held
  // attribute list.
  SmallVector<std::pair<Attribute, unsigned>, 1> worklist;
  worklist.push_back({attrDict, /*index*/ 0});
  while (!worklist.empty()) {
    Attribute attr = worklist.back().first;
    unsigned &index = worklist.back().second;

    // Iterate over the given attribute, which is guaranteed to be a container.
    HandlerResult handlerResult = HandlerResult::Continue;
    if (auto arrayAttr = attr.dyn_cast<ArrayAttr>()) {
      ArrayRef<Attribute> attrs = arrayAttr.getValue();
      unsigned attrSize = attrs.size();
      while (index != attrSize)
        if ((handlerResult = handleAttrDuringSymbolWalk(op, attrs[index++],
                                                        worklist, callback)))
          break;
    } else {
      auto dictAttr = attr.cast<DictionaryAttr>();
      ArrayRef<NamedAttribute> attrs = dictAttr.getValue();
      unsigned attrSize = attrs.size();
      while (index != attrSize)
        if ((handlerResult = handleAttrDuringSymbolWalk(
                 op, attrs[index++].second, worklist, callback)))
          break;
    }
    if (handlerResult == HandlerResult::Interrupt)
      return WalkResult::interrupt();

    // If we didn't encounter a nested attribute, pop the last item from the
    // worklist.
    if (handlerResult != HandlerResult::RecurseNestedAttribute)
      worklist.pop_back();
  }
  return WalkResult::advance();
}

/// Walk all of the uses, for any symbol, that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables, and will also only return uses on
/// 'from' if it does not also define a symbol table.
static Optional<WalkResult>
walkSymbolUses(Operation *from,
               function_ref<WalkResult(SymbolTable::SymbolUse)> callback) {
  // If from is not a symbol table, check for uses. A symbol table defines a new
  // scope, so we can't walk the attributes from the symbol table op.
  if (!from->hasTrait<OpTrait::SymbolTable>()) {
    if (walkSymbolRefs(from, callback).wasInterrupted())
      return WalkResult::interrupt();
  }

  SmallVector<Region *, 1> worklist;
  worklist.reserve(from->getNumRegions());
  for (Region &region : from->getRegions())
    worklist.push_back(&region);

  while (!worklist.empty()) {
    Region *region = worklist.pop_back_val();
    for (Block &block : *region) {
      for (Operation &op : block) {
        if (walkSymbolRefs(&op, callback).wasInterrupted())
          return WalkResult::interrupt();

        // If this operation has regions, and it as well as its dialect arent't
        // registered then conservatively fail. The operation may define a
        // symbol table, so we can't opaquely know if we should traverse to find
        // nested uses.
        if (isPotentiallyUnknownSymbolTable(&op))
          return llvm::None;

        // If this op defines a new symbol table scope, we can't traverse. Any
        // symbol references nested within 'op' are different semantically.
        if (!op.hasTrait<OpTrait::SymbolTable>()) {
          for (Region &region : op.getRegions())
            worklist.push_back(&region);
        }
      }
    }
  }
  return WalkResult::advance();
}

/// Get an iterator range for all of the uses, for any symbol, that are nested
/// within the given operation 'from'. This does not traverse into any nested
/// symbol tables, and will also only return uses on 'from' if it does not
/// also define a symbol table. This function returns None if there are any
/// unknown operations that may potentially be symbol tables.
auto SymbolTable::getSymbolUses(Operation *from) -> Optional<UseRange> {
  std::vector<SymbolUse> uses;
  Optional<WalkResult> result = walkSymbolUses(from, [&](SymbolUse symbolUse) {
    uses.push_back(symbolUse);
    return WalkResult::advance();
  });
  return result ? Optional<UseRange>(std::move(uses)) : Optional<UseRange>();
}

/// Get all of the uses of the given symbol that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables, and will also only return uses on
/// 'from' if it does not also define a symbol table. This function returns
/// None if there are any unknown operations that may potentially be symbol
/// tables.
auto SymbolTable::getSymbolUses(StringRef symbol, Operation *from)
    -> Optional<UseRange> {
  SymbolRefAttr symbolRefAttr = SymbolRefAttr::get(symbol, from->getContext());

  std::vector<SymbolUse> uses;
  Optional<WalkResult> result = walkSymbolUses(from, [&](SymbolUse symbolUse) {
    if (symbolRefAttr == symbolUse.getSymbolRef())
      uses.push_back(symbolUse);
    return WalkResult::advance();
  });
  return result ? Optional<UseRange>(std::move(uses)) : Optional<UseRange>();
}

/// Return if the given symbol is known to have no uses that are nested within
/// the given operation 'from'. This does not traverse into any nested symbol
/// tables, and will also only count uses on 'from' if it does not also define
/// a symbol table. This function will also return false if there are any
/// unknown operations that may potentially be symbol tables.
bool SymbolTable::symbolKnownUseEmpty(StringRef symbol, Operation *from) {
  SymbolRefAttr symbolRefAttr = SymbolRefAttr::get(symbol, from->getContext());

  // Walk all of the symbol uses looking for a reference to 'symbol'.
  Optional<WalkResult> walkResult =
      walkSymbolUses(from, [&](SymbolUse symbolUse) {
        return symbolUse.getSymbolRef() == symbolRefAttr
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      });
  return !walkResult || !walkResult->wasInterrupted();
}
