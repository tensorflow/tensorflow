//===- Module.h - MLIR Module Class -----------------------------*- C++ -*-===//
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
// Module is the top-level container for code in an MLIR program.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MODULE_H
#define MLIR_IR_MODULE_H

#include "mlir/IR/SymbolTable.h"

namespace mlir {
class ModuleTerminatorOp;

//===----------------------------------------------------------------------===//
// Module Operation.
//===----------------------------------------------------------------------===//

/// ModuleOp represents a module, or an operation containing one region with a
/// single block containing opaque operations. The region of a module is not
/// allowed to implicitly capture global values, and all external references
/// must use symbolic references via attributes(e.g. via a string name).
class ModuleOp
    : public Op<
          ModuleOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
          OpTrait::IsIsolatedFromAbove, OpTrait::SymbolTable,
          OpTrait::SingleBlockImplicitTerminator<ModuleTerminatorOp>::Impl> {
public:
  using Op::Op;
  using Op::print;

  static StringRef getOperationName() { return "module"; }

  static void build(Builder *builder, OperationState &result,
                    Optional<StringRef> name = llvm::None);

  /// Construct a module from the given location with an optional name.
  static ModuleOp create(Location loc, Optional<StringRef> name = llvm::None);

  /// Operation hooks.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &p);
  LogicalResult verify();

  /// Return body of this module.
  Region &getBodyRegion();
  Block *getBody();

  /// Return the name of this module if present.
  Optional<StringRef> getName();

  /// Print the this module in the custom top-level form.
  void print(raw_ostream &os, OpPrintingFlags flags = llvm::None);
  void dump();

  //===--------------------------------------------------------------------===//
  // Body Management.
  //===--------------------------------------------------------------------===//

  /// Iteration over the operations in the module.
  using iterator = Block::iterator;

  iterator begin() { return getBody()->begin(); }
  iterator end() { return getBody()->end(); }
  Operation &front() { return *begin(); }

  /// This returns a range of operations of the given type 'T' held within the
  /// module.
  template <typename T> llvm::iterator_range<Block::op_iterator<T>> getOps() {
    return getBody()->getOps<T>();
  }

  /// Insert the operation into the back of the body, before the terminator.
  void push_back(Operation *op) {
    insert(Block::iterator(getBody()->getTerminator()), op);
  }

  /// Insert the operation at the given insertion point. Note: The operation is
  /// never inserted after the terminator, even if the insertion point is end().
  void insert(Operation *insertPt, Operation *op) {
    insert(Block::iterator(insertPt), op);
  }
  void insert(Block::iterator insertPt, Operation *op) {
    auto *body = getBody();
    if (insertPt == body->end())
      insertPt = Block::iterator(body->getTerminator());
    body->getOperations().insert(insertPt, op);
  }
};

/// The ModuleTerminatorOp is a special terminator operation for the body of a
/// ModuleOp, it has no semantic meaning beyond keeping the body of a ModuleOp
/// well-formed.
///
/// This operation does _not_ have a custom syntax. However, ModuleOp will omit
/// the terminator in their custom syntax for brevity.
class ModuleTerminatorOp
    : public Op<ModuleTerminatorOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                OpTrait::HasParent<ModuleOp>::Impl, OpTrait::IsTerminator> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "module_terminator"; }
  static void build(Builder *, OperationState &) {}
};

/// This class acts as an owning reference to a module, and will automatically
/// destroy the held module if valid.
class OwningModuleRef {
public:
  OwningModuleRef(std::nullptr_t = nullptr) {}
  OwningModuleRef(ModuleOp module) : module(module) {}
  OwningModuleRef(OwningModuleRef &&other) : module(other.release()) {}
  ~OwningModuleRef() {
    if (module)
      module.erase();
  }

  // Assign from another module reference.
  OwningModuleRef &operator=(OwningModuleRef &&other) {
    if (module)
      module.erase();
    module = other.release();
    return *this;
  }

  /// Allow accessing the internal module.
  ModuleOp get() const { return module; }
  ModuleOp operator*() const { return module; }
  ModuleOp *operator->() { return &module; }
  explicit operator bool() const { return module; }

  /// Release the referenced module.
  ModuleOp release() {
    ModuleOp released;
    std::swap(released, module);
    return released;
  }

private:
  ModuleOp module;
};

} // end namespace mlir

namespace llvm {

/// Allow stealing the low bits of ModuleOp.
template <> struct PointerLikeTypeTraits<mlir::ModuleOp> {
public:
  static inline void *getAsVoidPointer(mlir::ModuleOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::ModuleOp getFromVoidPointer(void *P) {
    return mlir::ModuleOp::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // end namespace llvm

#endif // MLIR_IR_MODULE_H
