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
SymbolTable::SymbolTable(Operation *symbolTableOp)
    : symbolTableOp(symbolTableOp) {
  assert(symbolTableOp->hasTrait<OpTrait::SymbolTable>() &&
         "expected operation to have SymbolTable trait");
  assert(symbolTableOp->getNumRegions() == 1 &&
         "expected operation to have a single region");
  assert(has_single_element(symbolTableOp->getRegion(0)) &&
         "expected operation to have a single block");

  for (auto &op : symbolTableOp->getRegion(0).front()) {
    auto nameAttr = op.getAttrOfType<StringAttr>(getSymbolAttrName());
    if (!nameAttr)
      continue;

    auto inserted = symbolTable.insert({nameAttr.getValue(), &op});
    (void)inserted;
    assert(inserted.second &&
           "expected region to contain uniquely named symbol operations");
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
  assert(symbol->getParentOp() == symbolTableOp &&
         "expected this operation to be inside of the operation with this "
         "SymbolTable");

  auto it = symbolTable.find(nameAttr.getValue());
  if (it != symbolTable.end() && it->second == symbol) {
    symbolTable.erase(it);
    symbol->erase();
  }
}

/// Insert a new symbol into the table and associated operation, and rename it
/// as necessary to avoid collisions.
void SymbolTable::insert(Operation *symbol, Block::iterator insertPt) {
  auto nameAttr = symbol->getAttrOfType<StringAttr>(getSymbolAttrName());
  assert(nameAttr && "expected valid 'name' attribute");

  auto &body = symbolTableOp->getRegion(0).front();
  if (insertPt == Block::iterator() || insertPt == body.end())
    insertPt = Block::iterator(body.getTerminator());

  assert(insertPt->getParentOp() == symbolTableOp &&
         "expected insertPt to be in the associated module operation");

  body.getOperations().insert(insertPt, symbol);

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
  symbol->setAttr(getSymbolAttrName(),
                  StringAttr::get(nameBuffer, symbolTableOp->getContext()));
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
  if (!has_single_element(op->getRegion(0)))
    return op->emitOpError()
           << "Operations with a 'SymbolTable' must have exactly one block";

  // Check that all symbols are uniquely named within child regions.
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

LogicalResult OpTrait::impl::verifySymbol(Operation *op) {
  if (!op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName()))
    return op->emitOpError() << "requires string attribute '"
                             << mlir::SymbolTable::getSymbolAttrName() << "'";
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol Use Lists
//===----------------------------------------------------------------------===//

/// Walk all of the symbol references within the given operation, invoking the
/// provided callback for each found use. The callbacks takes as arguments: the
/// use of the symbol, and the nested access chain to the attribute within the
/// operation dictionary. An access chain is a set of indices into nested
/// container attributes. For example, a symbol use in an attribute dictionary
/// that looks like the following:
///
///    {use = [{other_attr, @symbol}]}
///
/// May have the following access chain:
///
///     [0, 0, 1]
///
static WalkResult walkSymbolRefs(
    Operation *op,
    function_ref<WalkResult(SymbolTable::SymbolUse, ArrayRef<int>)> callback) {
  // Check to see if the operation has any attributes.
  DictionaryAttr attrDict = op->getAttrList().getDictionary();
  if (!attrDict)
    return WalkResult::advance();

  // A worklist of a container attribute and the current index into the held
  // attribute list.
  SmallVector<Attribute, 1> attrWorklist(1, attrDict);
  SmallVector<int, 1> curAccessChain(1, /*Value=*/-1);

  // Process the symbol references within the given nested attribute range.
  auto processAttrs = [&](int &index, auto attrRange) -> WalkResult {
    for (Attribute attr : llvm::drop_begin(attrRange, index)) {
      /// Check for a nested container attribute, these will also need to be
      /// walked.
      if (attr.isa<ArrayAttr>() || attr.isa<DictionaryAttr>()) {
        attrWorklist.push_back(attr);
        curAccessChain.push_back(-1);
        return WalkResult::advance();
      }

      // Invoke the provided callback if we find a symbol use and check for a
      // requested interrupt.
      if (auto symbolRef = attr.dyn_cast<SymbolRefAttr>())
        if (callback({op, symbolRef}, curAccessChain).wasInterrupted())
          return WalkResult::interrupt();

      // Make sure to keep the index counter in sync.
      ++index;
    }

    // Pop this container attribute from the worklist.
    attrWorklist.pop_back();
    curAccessChain.pop_back();
    return WalkResult::advance();
  };

  WalkResult result = WalkResult::advance();
  do {
    Attribute attr = attrWorklist.back();
    int &index = curAccessChain.back();
    ++index;

    // Process the given attribute, which is guaranteed to be a container.
    if (auto dict = attr.dyn_cast<DictionaryAttr>())
      result = processAttrs(index, make_second_range(dict.getValue()));
    else
      result = processAttrs(index, attr.cast<ArrayAttr>().getValue());
  } while (!attrWorklist.empty() && !result.wasInterrupted());
  return result;
}

/// Walk all of the uses, for any symbol, that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables, and will also only return uses on
/// 'from' if it does not also define a symbol table.
static Optional<WalkResult> walkSymbolUses(
    Operation *from,
    function_ref<WalkResult(SymbolTable::SymbolUse, ArrayRef<int>)> callback) {
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

        // If this operation has regions, and it as well as its dialect aren't
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
/// also define a symbol table. This is because we treat the region as the
/// boundary of the symbol table, and not the op itself. This function returns
/// None if there are any unknown operations that may potentially be symbol
/// tables.
auto SymbolTable::getSymbolUses(Operation *from) -> Optional<UseRange> {
  std::vector<SymbolUse> uses;
  Optional<WalkResult> result =
      walkSymbolUses(from, [&](SymbolUse symbolUse, ArrayRef<int>) {
        uses.push_back(symbolUse);
        return WalkResult::advance();
      });
  return result ? Optional<UseRange>(std::move(uses)) : Optional<UseRange>();
}

/// Get all of the uses of the given symbol that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables, and will also only return uses on
/// 'from' if it does not also define a symbol table. This is because we treat
/// the region as the boundary of the symbol table, and not the op itself. This
/// function returns None if there are any unknown operations that may
/// potentially be symbol tables.
auto SymbolTable::getSymbolUses(StringRef symbol, Operation *from)
    -> Optional<UseRange> {
  SymbolRefAttr symbolRefAttr = SymbolRefAttr::get(symbol, from->getContext());

  std::vector<SymbolUse> uses;
  Optional<WalkResult> result =
      walkSymbolUses(from, [&](SymbolUse symbolUse, ArrayRef<int>) {
        if (symbolRefAttr == symbolUse.getSymbolRef())
          uses.push_back(symbolUse);
        return WalkResult::advance();
      });
  return result ? Optional<UseRange>(std::move(uses)) : Optional<UseRange>();
}

/// Return if the given symbol is known to have no uses that are nested within
/// the given operation 'from'. This does not traverse into any nested symbol
/// tables, and will also only count uses on 'from' if it does not also define
/// a symbol table. This is because we treat the region as the boundary of the
/// symbol table, and not the op itself. This function will also return false if
/// there are any unknown operations that may potentially be symbol tables.
bool SymbolTable::symbolKnownUseEmpty(StringRef symbol, Operation *from) {
  SymbolRefAttr symbolRefAttr = SymbolRefAttr::get(symbol, from->getContext());

  // Walk all of the symbol uses looking for a reference to 'symbol'.
  Optional<WalkResult> walkResult =
      walkSymbolUses(from, [&](SymbolUse symbolUse, ArrayRef<int>) {
        return symbolUse.getSymbolRef() == symbolRefAttr
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      });
  return walkResult && !walkResult->wasInterrupted();
}

/// Rebuild the given attribute container after replacing all references to a
/// symbol with `newSymAttr`.
static Attribute rebuildAttrAfterRAUW(Attribute container,
                                      ArrayRef<SmallVector<int, 1>> accesses,
                                      SymbolRefAttr newSymAttr,
                                      unsigned depth) {
  // Given a range of Attributes, update the ones referred to by the given
  // access chains to point to the new symbol attribute.
  auto updateAttrs = [&](auto &&attrRange) {
    auto attrBegin = std::begin(attrRange);
    for (unsigned i = 0, e = accesses.size(); i != e;) {
      ArrayRef<int> access = accesses[i];
      Attribute &attr = *std::next(attrBegin, access[depth]);

      // Check to see if this is a leaf access, i.e. a SymbolRef.
      if (access.size() == depth + 1) {
        attr = newSymAttr;
        ++i;
        continue;
      }

      // Otherwise, this is a container. Collect all of the accesses for this
      // index and recurse. The recursion here is bounded by the size of the
      // largest access array.
      auto nestedAccesses =
          accesses.drop_front(i).take_while([&](ArrayRef<int> nextAccess) {
            return nextAccess.size() > depth + 1 &&
                   nextAccess[depth] == access[depth];
          });
      attr = rebuildAttrAfterRAUW(attr, nestedAccesses, newSymAttr, depth + 1);

      // Skip over all of the accesses that refer to the nested container.
      i += nestedAccesses.size();
    }
  };

  if (auto dictAttr = container.dyn_cast<DictionaryAttr>()) {
    auto newAttrs = llvm::to_vector<4>(dictAttr.getValue());
    updateAttrs(make_second_range(newAttrs));
    return DictionaryAttr::get(newAttrs, dictAttr.getContext());
  }
  auto newAttrs = llvm::to_vector<4>(container.cast<ArrayAttr>().getValue());
  updateAttrs(newAttrs);
  return ArrayAttr::get(newAttrs, container.getContext());
}

/// Attempt to replace all uses of the given symbol 'oldSymbol' with the
/// provided symbol 'newSymbol' that are nested within the given operation
/// 'from'. This does not traverse into any nested symbol tables, and will
/// also only replace uses on 'from' if it does not also define a symbol
/// table. This is because we treat the region as the boundary of the symbol
/// table, and not the op itself. If there are any unknown operations that may
/// potentially be symbol tables, no uses are replaced and failure is returned.
LogicalResult SymbolTable::replaceAllSymbolUses(StringRef oldSymbol,
                                                StringRef newSymbol,
                                                Operation *from) {
  SymbolRefAttr oldAttr = SymbolRefAttr::get(oldSymbol, from->getContext());
  SymbolRefAttr newSymAttr = SymbolRefAttr::get(newSymbol, from->getContext());

  // A collection of operations along with their new attribute dictionary.
  std::vector<std::pair<Operation *, DictionaryAttr>> updatedAttrDicts;

  // The current operation, and its old symbol access chains, being processed.
  Operation *curOp = nullptr;
  SmallVector<SmallVector<int, 1>, 1> accessChains;

  // Generate a new attribute dictionary for the current operation by replacing
  // references to the old symbol.
  auto generateNewAttrDict = [&] {
    auto newAttrDict =
        rebuildAttrAfterRAUW(curOp->getAttrList().getDictionary(), accessChains,
                             newSymAttr, /*depth=*/0);
    return newAttrDict.cast<DictionaryAttr>();
  };

  // Walk the symbol uses collecting uses of the old symbol.
  auto walkFn = [&](SymbolTable::SymbolUse symbolUse,
                    ArrayRef<int> accessChain) {
    if (symbolUse.getSymbolRef() != oldAttr)
      return WalkResult::advance();

    // If there was a previous operation, generate a new attribute dict for it.
    // This means that we've finished processing the current operation, so
    // generate a new dictionary for it.
    if (curOp && symbolUse.getUser() != curOp) {
      updatedAttrDicts.push_back({curOp, generateNewAttrDict()});
      accessChains.clear();
    }

    // Record this access.
    curOp = symbolUse.getUser();
    accessChains.push_back(llvm::to_vector<1>(accessChain));
    return WalkResult::advance();
  };
  if (!walkSymbolUses(from, walkFn))
    return failure();

  // Update the attribute dictionaries as necessary.
  for (auto &it : updatedAttrDicts)
    it.first->setAttrs(it.second);

  // Check to see if we have a dangling op that needs to be processed.
  if (curOp)
    curOp->setAttrs(generateNewAttrDict());

  return success();
}
