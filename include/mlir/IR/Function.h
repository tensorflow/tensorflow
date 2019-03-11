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
// Functions are the basic unit of composition in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"

namespace mlir {
class BlockAndValueMapping;
class FunctionType;
class MLIRContext;
class Module;
template <typename ObjectType, typename ElementType> class ArgumentIterator;
template <typename T> class OpPointer;

/// This is the base class for all of the MLIR function types.
class Function : public llvm::ilist_node_with_parent<Function, Module> {
public:
  Function(Location location, StringRef name, FunctionType type,
           ArrayRef<NamedAttribute> attrs = {});
  Function(Location location, StringRef name, FunctionType type,
           ArrayRef<NamedAttribute> attrs,
           ArrayRef<NamedAttributeList> argAttrs);

  ~Function();

  /// The source location the function was defined or derived from.
  Location getLoc() const { return location; }

  /// Set the source location this function was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Return the name of this function, without the @.
  Identifier getName() const { return name; }

  /// Return the type of this function.
  FunctionType getType() const { return type; }

  MLIRContext *getContext() const;
  Module *getModule() { return module; }
  const Module *getModule() const { return module; }

  /// Add an entry block to an empty function, and set up the block arguments
  /// to match the signature of the function.
  void addEntryBlock();

  /// Unlink this function from its module and delete it.
  void erase();

  /// Returns true if this function is external, i.e. it has no body.
  bool isExternal() const { return empty(); }

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

  //===--------------------------------------------------------------------===//
  // Instruction Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the instructions in the function in preorder, calling the callback
  /// for each instruction.
  void walk(const std::function<void(Instruction *)> &callback);

  /// Specialization of walk to only visit operations of 'OpTy'.
  template <typename OpTy>
  void walk(std::function<void(OpPointer<OpTy>)> callback) {
    walk([&](Instruction *inst) {
      if (auto op = inst->dyn_cast<OpTy>())
        callback(op);
    });
  }

  /// Walk the instructions in the function in postorder, calling the callback
  /// for each instruction.
  void walkPostOrder(const std::function<void(Instruction *)> &callback);

  /// Specialization of walkPostOrder to only visit operations of 'OpTy'.
  template <typename OpTy>
  void walkPostOrder(std::function<void(OpPointer<OpTy>)> callback) {
    walkPostOrder([&](Instruction *inst) {
      if (auto op = inst->dyn_cast<OpTy>())
        callback(op);
    });
  }

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
  // Attributes
  //===--------------------------------------------------------------------===//

  /// Functions may optionally carry a list of attributes that associate
  /// constants to names.  Attributes may be dynamically added and removed over
  /// the lifetime of an function.

  /// Return all of the attributes on this function.
  ArrayRef<NamedAttribute> getAttrs() const { return attrs.getAttrs(); }

  /// Return all of the attributes for the argument at 'index'.
  ArrayRef<NamedAttribute> getArgAttrs(unsigned index) const {
    assert(index < getNumArguments() && "invalid argument number");
    return argAttrs[index].getAttrs();
  }

  /// Set the attributes held by this function.
  void setAttrs(ArrayRef<NamedAttribute> attributes) {
    attrs.setAttrs(getContext(), attributes);
  }

  /// Set the attributes held by the argument at 'index'.
  void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
    assert(index < getNumArguments() && "invalid argument number");
    argAttrs[index].setAttrs(getContext(), attributes);
  }

  /// Return all argument attributes of this function.
  MutableArrayRef<NamedAttributeList> getAllArgAttrs() { return argAttrs; }
  ArrayRef<NamedAttributeList> getAllArgAttrs() const { return argAttrs; }

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) const { return attrs.get(name); }
  Attribute getAttr(StringRef name) const { return attrs.get(name); }

  /// Return the specified attribute, if present, for the argument at 'index',
  /// null otherwise.
  Attribute getArgAttr(unsigned index, Identifier name) const {
    assert(index < getNumArguments() && "invalid argument number");
    return argAttrs[index].get(name);
  }
  Attribute getArgAttr(unsigned index, StringRef name) const {
    assert(index < getNumArguments() && "invalid argument number");
    return argAttrs[index].get(name);
  }

  template <typename AttrClass> AttrClass getAttrOfType(Identifier name) const {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) const {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, Identifier name) const {
    return getArgAttr(index, name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, StringRef name) const {
    return getArgAttr(index, name).dyn_cast_or_null<AttrClass>();
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value.  Otherwise, add a new attribute with the specified name/value.
  void setAttr(Identifier name, Attribute value) {
    attrs.set(getContext(), name, value);
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(Identifier::get(name, getContext()), value);
  }
  void setArgAttr(unsigned index, Identifier name, Attribute value) {
    assert(index < getNumArguments() && "invalid argument number");
    argAttrs[index].set(getContext(), name, value);
  }
  void setArgAttr(unsigned index, StringRef name, Attribute value) {
    setArgAttr(index, Identifier::get(name, getContext()), value);
  }

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  NamedAttributeList::RemoveResult removeAttr(Identifier name) {
    return attrs.remove(getContext(), name);
  }
  NamedAttributeList::RemoveResult removeArgAttr(unsigned index,
                                                 Identifier name) {
    assert(index < getNumArguments() && "invalid argument number");
    return attrs.remove(getContext(), name);
  }

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

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  Function *clone(BlockAndValueMapping &mapper) const;
  Function *clone() const;

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(Function *dest, BlockAndValueMapping &mapper) const;

private:
  /// The name of the function.
  Identifier name;

  /// The module this function is embedded into.
  Module *module = nullptr;

  /// The source location the function was defined or derived from.
  Location location;

  /// The type of the function.
  FunctionType type;

  /// This holds general named attributes for the function.
  NamedAttributeList attrs;

  /// The attributes lists for each of the function arguments.
  std::vector<NamedAttributeList> argAttrs;

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
