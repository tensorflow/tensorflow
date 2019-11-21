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

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
class Identifier;
class MLIRContext;
class Operation;

/// This class allows for representing and managing the symbol table used by
/// operations with the 'SymbolTable' trait.
class SymbolTable {
public:
  /// Build a symbol table with the symbols within the given operation.
  SymbolTable(Operation *op);

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringRef name) const;
  template <typename T> T lookup(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

  /// Erase the given symbol from the table.
  void erase(Operation *symbol);

  /// Insert a new symbol into the table, and rename it as necessary to avoid
  /// collisions.
  void insert(Operation *symbol);

  /// Returns the context held by this symbol table.
  MLIRContext *getContext() const { return context; }

  /// Return the name of the attribute used for symbol names.
  static StringRef getSymbolAttrName() { return "sym_name"; }

  //===--------------------------------------------------------------------===//
  // Symbol Utilities
  //===--------------------------------------------------------------------===//

  /// Returns the operation registered with the given symbol name with the
  /// regions of 'symbolTableOp'. 'symbolTableOp' is required to be an operation
  /// with the 'OpTrait::SymbolTable' trait.
  static Operation *lookupSymbolIn(Operation *symbolTableOp, StringRef symbol);

  /// Returns the operation registered with the given symbol name within the
  /// closest parent operation of, or including, 'from' with the
  /// 'OpTrait::SymbolTable' trait. Returns nullptr if no valid symbol was
  /// found.
  static Operation *lookupNearestSymbolFrom(Operation *from, StringRef symbol);

  /// This class represents a specific symbol use.
  class SymbolUse {
  public:
    SymbolUse(Operation *op, SymbolRefAttr symbolRef)
        : owner(op), symbolRef(symbolRef) {}

    /// Return the operation user of this symbol reference.
    Operation *getUser() const { return owner; }

    /// Return the symbol reference that this use represents.
    SymbolRefAttr getSymbolRef() const { return symbolRef; }

  private:
    /// The operation that this access is held by.
    Operation *owner;

    /// The symbol reference that this use represents.
    SymbolRefAttr symbolRef;
  };

  /// This class implements a range of SymbolRef uses.
  class UseRange {
  public:
    UseRange(std::vector<SymbolUse> &&uses) : uses(std::move(uses)) {}

    using iterator = std::vector<SymbolUse>::const_iterator;
    iterator begin() const { return uses.begin(); }
    iterator end() const { return uses.end(); }

  private:
    std::vector<SymbolUse> uses;
  };

  /// Get an iterator range for all of the uses, for any symbol, that are nested
  /// within the given operation 'from'. This does not traverse into any nested
  /// symbol tables, and will also only return uses on 'from' if it does not
  /// also define a symbol table. This is because we treat the region as the
  /// boundary of the symbol table, and not the op itself. This function returns
  /// None if there are any unknown operations that may potentially be symbol
  /// tables.
  static Optional<UseRange> getSymbolUses(Operation *from);

  /// Get all of the uses of the given symbol that are nested within the given
  /// operation 'from'. This does not traverse into any nested symbol tables,
  /// and will also only return uses on 'from' if it does not also define a
  /// symbol table. This is because we treat the region as the boundary of the
  /// symbol table, and not the op itself. This function returns None if there
  /// are any unknown operations that may potentially be symbol tables.
  static Optional<UseRange> getSymbolUses(StringRef symbol, Operation *from);

  /// Return if the given symbol is known to have no uses that are nested within
  /// the given operation 'from'. This does not traverse into any nested symbol
  /// tables, and will also only count uses on 'from' if it does not also define
  /// a symbol table. This is because we treat the region as the boundary of
  /// the symbol table, and not the op itself. This function will also return
  /// false if there are any unknown operations that may potentially be symbol
  /// tables. This doesn't necessarily mean that there are no uses, we just
  /// can't convervatively prove it.
  static bool symbolKnownUseEmpty(StringRef symbol, Operation *from);

  /// Attempt to replace all uses of the given symbol 'oldSymbol' with the
  /// provided symbol 'newSymbol' that are nested within the given operation
  /// 'from'. This does not traverse into any nested symbol tables, and will
  /// also only replace uses on 'from' if it does not also define a symbol
  /// table. This is because we treat the region as the boundary of the symbol
  /// table, and not the op itself. If there are any unknown operations that may
  /// potentially be symbol tables, no uses are replaced and failure is
  /// returned.
  LLVM_NODISCARD static LogicalResult replaceAllSymbolUses(StringRef oldSymbol,
                                                           StringRef newSymbol,
                                                           Operation *from);

private:
  MLIRContext *context;

  /// This is a mapping from a name to the symbol with that name.
  llvm::StringMap<Operation *> symbolTable;

  /// This is used when name conflicts are detected.
  unsigned uniquingCounter = 0;
};

//===----------------------------------------------------------------------===//
// SymbolTable Trait Types
//===----------------------------------------------------------------------===//

namespace OpTrait {
namespace impl {
LogicalResult verifySymbolTable(Operation *op);
LogicalResult verifySymbol(Operation *op);
} // namespace impl

/// A trait used to provide symbol table functionalities to a region operation.
/// This operation must hold exactly 1 region. Once attached, all operations
/// that are directly within the region, i.e not including those within child
/// regions, that contain a 'SymbolTable::getSymbolAttrName()' StringAttr will
/// be verified to ensure that the names are uniqued. These operations must also
/// adhere to the constraints defined by the `Symbol` trait, even if they do not
/// inherit from it.
template <typename ConcreteType>
class SymbolTable : public TraitBase<ConcreteType, SymbolTable> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySymbolTable(op);
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Symbol names never include the @ on them. Note: This
  /// performs a linear scan of held symbols.
  Operation *lookupSymbol(StringRef name) {
    return mlir::SymbolTable::lookupSymbolIn(this->getOperation(), name);
  }
  template <typename T> T lookupSymbol(StringRef name) {
    return dyn_cast_or_null<T>(lookupSymbol(name));
  }
};

/// A trait used to define a symbol that can be used on operations within a
/// symbol table. Operations using this trait must adhere to the following:
///   * Have a StringAttr attribute named 'SymbolTable::getSymbolAttrName()'.
template <typename ConcreteType>
class Symbol : public TraitBase<ConcreteType, Symbol> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifySymbol(op);
  }

  /// Returns the name of this symbol.
  StringRef getName() {
    return this->getOperation()
        ->template getAttrOfType<StringAttr>(
            mlir::SymbolTable::getSymbolAttrName())
        .getValue();
  }

  /// Set the name of this symbol.
  void setName(StringRef name) {
    this->getOperation()->setAttr(
        mlir::SymbolTable::getSymbolAttrName(),
        StringAttr::get(name, this->getOperation()->getContext()));
  }

  /// Get all of the uses of the current symbol that are nested within the given
  /// operation 'from'.
  /// Note: See mlir::SymbolTable::getSymbolUses for more details.
  Optional<::mlir::SymbolTable::UseRange> getSymbolUses(Operation *from) {
    return ::mlir::SymbolTable::getSymbolUses(getName(), from);
  }

  /// Return if the current symbol is known to have no uses that are nested
  /// within the given operation 'from'.
  /// Note: See mlir::SymbolTable::symbolKnownUseEmpty for more details.
  bool symbolKnownUseEmpty(Operation *from) {
    return ::mlir::SymbolTable::symbolKnownUseEmpty(getName(), from);
  }

  /// Attempt to replace all uses of the current symbol with the provided symbol
  /// 'newSymbol' that are nested within the given operation 'from'.
  /// Note: See mlir::SymbolTable::replaceAllSymbolUses for more details.
  LLVM_NODISCARD LogicalResult replaceAllSymbolUses(StringRef newSymbol,
                                                    Operation *from) {
    return ::mlir::SymbolTable::replaceAllSymbolUses(getName(), newSymbol,
                                                     from);
  }
};

} // end namespace OpTrait
} // end namespace mlir

#endif // MLIR_IR_SYMBOLTABLE_H
