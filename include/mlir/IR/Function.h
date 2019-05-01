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
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ilist.h"

namespace mlir {
class BlockAndValueMapping;
class FunctionType;
class MLIRContext;
class Module;
class ArgumentIterator;

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
  Location getLoc() { return location; }

  /// Set the source location this function was defined or derived from.
  void setLoc(Location loc) { location = loc; }

  /// Return the name of this function, without the @.
  Identifier getName() { return name; }

  /// Return the type of this function.
  FunctionType getType() { return type; }

  /// Change the type of this function in place. This is an extremely dangerous
  /// operation and it is up to the caller to ensure that this is legal for this
  /// function, and to restore invariants:
  ///   - the entry block args must be updated to match the function params.
  ///  - the arguments attributes may need an update: if the new type has less
  ///    parameters we drop the extra attributes, if there are more parameters
  ///    they won't have any attributes.
  void setType(FunctionType newType) {
    type = newType;
    argAttrs.resize(type.getNumInputs());
  }

  MLIRContext *getContext();
  Module *getModule() { return module; }

  /// Add an entry block to an empty function, and set up the block arguments
  /// to match the signature of the function.
  void addEntryBlock();

  /// Unlink this function from its module and delete it.
  void erase();

  /// Returns true if this function is external, i.e. it has no body.
  bool isExternal() { return empty(); }

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  Region &getBody() { return body; }

  /// This is the list of blocks in the function.
  using RegionType = Region::RegionType;
  RegionType &getBlocks() { return body.getBlocks(); }

  // Iteration over the block in the function.
  using iterator = RegionType::iterator;
  using reverse_iterator = RegionType::reverse_iterator;

  iterator begin() { return body.begin(); }
  iterator end() { return body.end(); }
  reverse_iterator rbegin() { return body.rbegin(); }
  reverse_iterator rend() { return body.rend(); }

  bool empty() { return body.empty(); }
  void push_back(Block *block) { body.push_back(block); }
  void push_front(Block *block) { body.push_front(block); }

  Block &back() { return body.back(); }
  Block &front() { return body.front(); }

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operations in the function in postorder, calling the callback for
  /// each operation.
  void walk(const std::function<void(Operation *)> &callback);

  /// Specialization of walk to only visit operations of 'OpTy'.
  template <typename OpTy> void walk(std::function<void(OpTy)> callback) {
    walk([&](Operation *opInst) {
      if (auto op = opInst->dyn_cast<OpTy>())
        callback(op);
    });
  }

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  /// Returns number of arguments.
  unsigned getNumArguments() { return getType().getInputs().size(); }

  /// Gets argument.
  BlockArgument *getArgument(unsigned idx) {
    return getBlocks().front().getArgument(idx);
  }

  // Supports non-const operand iteration.
  using args_iterator = ArgumentIterator;
  args_iterator args_begin();
  args_iterator args_end();
  llvm::iterator_range<args_iterator> getArguments();

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  /// Functions may optionally carry a list of attributes that associate
  /// constants to names.  Attributes may be dynamically added and removed over
  /// the lifetime of an function.

  /// Return all of the attributes on this function.
  ArrayRef<NamedAttribute> getAttrs() { return attrs.getAttrs(); }

  /// Return all of the attributes for the argument at 'index'.
  ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
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

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) { return attrs.get(name); }
  Attribute getAttr(StringRef name) { return attrs.get(name); }

  /// Return the specified attribute, if present, for the argument at 'index',
  /// null otherwise.
  Attribute getArgAttr(unsigned index, Identifier name) {
    assert(index < getNumArguments() && "invalid argument number");
    return argAttrs[index].get(name);
  }
  Attribute getArgAttr(unsigned index, StringRef name) {
    assert(index < getNumArguments() && "invalid argument number");
    return argAttrs[index].get(name);
  }

  template <typename AttrClass> AttrClass getAttrOfType(Identifier name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass> AttrClass getAttrOfType(StringRef name) {
    return getAttr(name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, Identifier name) {
    return getArgAttr(index, name).dyn_cast_or_null<AttrClass>();
  }

  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, StringRef name) {
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
  /// and returns failure.
  LogicalResult verify();

  void print(raw_ostream &os);
  void dump();

  /// Emit an error about fatal conditions with this function, reporting up to
  /// any diagnostic handlers that may be listening.  This function always
  /// returns failure.  NOTE: This may terminate the containing application,
  /// only use when the IR is in an inconsistent state.
  LogicalResult emitError(const Twine &message);

  /// Emit a warning about this function, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitWarning(const Twine &message);

  /// Emit a remark about this function, reporting up to any diagnostic
  /// handlers that may be listening.
  void emitRemark(const Twine &message);

  /// Displays the CFG in a window. This is for use from the debugger and
  /// depends on Graphviz to generate the graph.
  /// This function is defined in CFGFunctionViewGraph and only works with that
  /// target linked.
  void viewGraph();

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  Function *clone(BlockAndValueMapping &mapper);
  Function *clone();

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(Function *dest, BlockAndValueMapping &mapper);

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

  /// The body of the function.
  Region body;

  void operator=(Function &) = delete;
  friend struct llvm::ilist_traits<Function>;
};

//===--------------------------------------------------------------------===//
// ArgumentIterator
//===--------------------------------------------------------------------===//

/// This template implements the argument iterator in terms of getArgument(idx).
class ArgumentIterator final
    : public IndexedAccessorIterator<ArgumentIterator, Function,
                                     BlockArgument> {
public:
  /// Initializes the result iterator to the specified index.
  ArgumentIterator(Function *object, unsigned index)
      : IndexedAccessorIterator<ArgumentIterator, Function, BlockArgument>(
            object, index) {}

  BlockArgument *operator*() const {
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
