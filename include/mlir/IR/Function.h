//===- Function.h - MLIR Function Class -------------------------*- C++ -*-===//
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
// Functions are the basic unit of composition in MLIR.  There are three
// different kinds of functions: external functions, CFG functions, and ML
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"

namespace mlir {
class AttributeListStorage;
class FunctionType;
class MLIRContext;
class Module;
template <typename ObjectType, typename ElementType> class ArgumentIterator;

/// NamedAttribute is used for function attribute lists, it holds an
/// identifier for the name and a value for the attribute.  The attribute
/// pointer should always be non-null.
using NamedAttribute = std::pair<Identifier, Attribute>;

/// This is the base class for all of the MLIR function types.
class Function : public llvm::ilist_node_with_parent<Function, Module> {
public:
  enum class Kind { ExtFunc, CFGFunc, MLFunc };

  Function(Kind kind, Location location, StringRef name, FunctionType type,
           ArrayRef<NamedAttribute> attrs = {});
  ~Function();

  Kind getKind() const { return (Kind)nameAndKind.getInt(); }

  bool isCFG() const { return getKind() == Kind::CFGFunc; }
  bool isML() const { return getKind() == Kind::MLFunc; }

  /// The source location the operation was defined or derived from.
  Location getLoc() const { return location; }

  /// Return the name of this function, without the @.
  Identifier getName() const { return nameAndKind.getPointer(); }

  /// Return the type of this function.
  FunctionType getType() const { return type; }

  /// Returns all of the attributes on this function.
  ArrayRef<NamedAttribute> getAttrs() const;

  MLIRContext *getContext() const;
  Module *getModule() { return module; }
  const Module *getModule() const { return module; }

  /// Unlink this function from its module and delete it.
  void erase();

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  BlockList &getBlockList() { return blocks; }
  const BlockList &getBlockList() const { return blocks; }

  /// This is the list of blocks in the function.
  using BlockListType = llvm::iplist<Block>;
  BlockListType &getBlocks() { return blocks.getBlocks(); }
  const BlockListType &getBlocks() const { return blocks.getBlocks(); }

  // Iteration over the block in the function.
  using iterator = BlockListType::iterator;
  using const_iterator = BlockListType::const_iterator;
  using reverse_iterator = BlockListType::reverse_iterator;
  using const_reverse_iterator = BlockListType::const_reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  const_iterator begin() const { return blocks.begin(); }
  const_iterator end() const { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }
  const_reverse_iterator rbegin() const { return blocks.rbegin(); }
  const_reverse_iterator rend() const { return blocks.rend(); }

  bool empty() const { return blocks.empty(); }
  void push_back(Block *block) { blocks.push_back(block); }
  void push_front(Block *block) { blocks.push_front(block); }

  Block &back() { return blocks.back(); }
  const Block &back() const { return const_cast<Function *>(this)->back(); }

  Block &front() { return blocks.front(); }
  const Block &front() const { return const_cast<Function *>(this)->front(); }

  /// Return the 'return' instruction of this Function.
  const OperationInst *getReturn() const;
  OperationInst *getReturn();

  // These should only be used on MLFunctions.
  Block *getBody() {
    assert(isML());
    return &blocks.front();
  }
  const Block *getBody() const {
    return const_cast<Function *>(this)->getBody();
  }

  /// Walk the instructions in the function in preorder, calling the callback
  /// for each operation instruction.
  void walk(std::function<void(OperationInst *)> callback);

  /// Walk the instructions in the function in postorder, calling the callback
  /// for each operation instruction.
  void walkPostOrder(std::function<void(OperationInst *)> callback);

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  /// Returns number of arguments.
  unsigned getNumArguments() const { return getType().getInputs().size(); }

  /// Gets argument.
  BlockArgument *getArgument(unsigned idx) {
    return getBlocks().front().getArgument(idx);
  }

  const BlockArgument *getArgument(unsigned idx) const {
    return getBlocks().front().getArgument(idx);
  }

  // Supports non-const operand iteration.
  using args_iterator = ArgumentIterator<Function, BlockArgument>;
  args_iterator args_begin();
  args_iterator args_end();
  llvm::iterator_range<args_iterator> getArguments();

  // Supports const operand iteration.
  using const_args_iterator =
      ArgumentIterator<const Function, const BlockArgument>;
  const_args_iterator args_begin() const;
  const_args_iterator args_end() const;
  llvm::iterator_range<const_args_iterator> getArguments() const;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// Perform (potentially expensive) checks of invariants, used to detect
  /// compiler bugs.  On error, this reports the error through the MLIRContext
  /// and returns true.
  bool verify() const;

  void print(raw_ostream &os) const;
  void dump() const;

  /// Emit an error about fatal conditions with this operation, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns true.  NOTE: This may terminate the containing application, only
  /// use when the IR is in an inconsistent state.
  bool emitError(const Twine &message) const;

  /// Emit a warning about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message) const;

  /// Emit a note about this operation, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitNote(const Twine &message) const;

  /// Displays the CFG in a window. This is for use from the debugger and
  /// depends on Graphviz to generate the graph.
  /// This function is defined in CFGFunctionViewGraph and only works with that
  /// target linked.
  void viewGraph() const;

private:
  /// The name of the function and the kind of function this is.
  llvm::PointerIntPair<Identifier, 2, Kind> nameAndKind;

  /// The module this function is embedded into.
  Module *module = nullptr;

  /// The source location the function was defined or derived from.
  Location location;

  /// The type of the function.
  FunctionType type;

  /// This holds general named attributes for the function.
  AttributeListStorage *attrs;

  /// The contents of the body.
  BlockList blocks;

  void operator=(const Function &) = delete;
  friend struct llvm::ilist_traits<Function>;
};

//===--------------------------------------------------------------------===//
// ArgumentIterator
//===--------------------------------------------------------------------===//

/// This template implements the argument iterator in terms of getArgument(idx).
template <typename ObjectType, typename ElementType>
class ArgumentIterator final
    : public IndexedAccessorIterator<ArgumentIterator<ObjectType, ElementType>,
                                     ObjectType, ElementType> {
public:
  /// Initializes the result iterator to the specified index.
  ArgumentIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<ArgumentIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator ArgumentIterator<const ObjectType, const ElementType>() const {
    return ArgumentIterator<const ObjectType, const ElementType>(this->object,
                                                                 this->index);
  }

  ElementType *operator*() const {
    return this->object->getArgument(this->index);
  }
};

//===--------------------------------------------------------------------===//
// Function iterator methods.
//===--------------------------------------------------------------------===//

inline Function::args_iterator Function::args_begin() {
  return args_iterator(this, 0);
}

inline Function::args_iterator Function::args_end() {
  return args_iterator(this, getNumArguments());
}

inline llvm::iterator_range<Function::args_iterator> Function::getArguments() {
  return {args_begin(), args_end()};
}

inline Function::const_args_iterator Function::args_begin() const {
  return const_args_iterator(this, 0);
}

inline Function::const_args_iterator Function::args_end() const {
  return const_args_iterator(this, getNumArguments());
}

inline llvm::iterator_range<Function::const_args_iterator>
Function::getArguments() const {
  return {args_begin(), args_end()};
}

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Function
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::Function>
    : public ilist_alloc_traits<::mlir::Function> {
  using Function = ::mlir::Function;
  using function_iterator = simple_ilist<Function>::iterator;

  static void deleteNode(Function *function) { delete function; }

  void addNodeToList(Function *function);
  void removeNodeFromList(Function *function);
  void transferNodesFromList(ilist_traits<Function> &otherList,
                             function_iterator first, function_iterator last);

private:
  mlir::Module *getContainingModule();
};
} // end namespace llvm

#endif  // MLIR_IR_FUNCTION_H
