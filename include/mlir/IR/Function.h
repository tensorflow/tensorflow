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

#include "mlir/IR/Block.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {
class ModuleOp;

//===--------------------------------------------------------------------===//
// Function Operation.
//===--------------------------------------------------------------------===//

/// FuncOp represents a function, or an operation containing one region that
/// forms a CFG(Control Flow Graph). The region of a function is not allowed to
/// implicitly capture global values, and all external references must use
/// Function arguments or attributes that establish a symbolic connection(e.g.
/// symbols referenced by name via a string attribute).
class FuncOp : public Op<FuncOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                         OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;

  static StringRef getOperationName() { return "func"; }

  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs = {});
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       llvm::iterator_range<dialect_attr_iterator> attrs);
  static FuncOp create(Location location, StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs,
                       ArrayRef<NamedAttributeList> argAttrs);

  static void build(Builder *builder, OperationState *result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs);
  static void build(Builder *builder, OperationState *result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs,
                    ArrayRef<NamedAttributeList> argAttrs);

  /// Get the parent module.
  ModuleOp getModule();

  /// Operation hooks.
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);
  LogicalResult verify();

  /// Returns the name of this function.
  StringRef getName() { return getAttrOfType<StringAttr>("name").getValue(); }

  /// Set the name of this function.
  void setName(StringRef name) {
    return setAttr("name", StringAttr::get(name, getContext()));
  }

  /// Returns the type of this function.
  FunctionType getType() {
    return getAttrOfType<TypeAttr>("type").getValue().cast<FunctionType>();
  }

  /// Change the type of this function in place. This is an extremely dangerous
  /// operation and it is up to the caller to ensure that this is legal for this
  /// function, and to restore invariants:
  ///  - the entry block args must be updated to match the function params.
  ///  - the arguments attributes may need an update: if the new type has less
  ///    parameters we drop the extra attributes, if there are more parameters
  ///    they won't have any attributes.
  void setType(FunctionType newType) {
    setAttr("type", TypeAttr::get(newType));
  }

  /// Returns true if this function is external, i.e. it has no body.
  bool isExternal() { return empty(); }

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  FuncOp clone(BlockAndValueMapping &mapper);
  FuncOp clone();

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  Region &getBody() { return getOperation()->getRegion(0); }

  void eraseBody() { getBody().getBlocks().clear(); }

  /// This is the list of blocks in the function.
  using RegionType = Region::RegionType;
  RegionType &getBlocks() { return getBody().getBlocks(); }

  // Iteration over the block in the function.
  using iterator = RegionType::iterator;
  using reverse_iterator = RegionType::reverse_iterator;

  iterator begin() { return getBody().begin(); }
  iterator end() { return getBody().end(); }
  reverse_iterator rbegin() { return getBody().rbegin(); }
  reverse_iterator rend() { return getBody().rend(); }

  bool empty() { return getBody().empty(); }
  void push_back(Block *block) { getBody().push_back(block); }
  void push_front(Block *block) { getBody().push_front(block); }

  Block &back() { return getBody().back(); }
  Block &front() { return getBody().front(); }

  /// Add an entry block to an empty function, and set up the block arguments
  /// to match the signature of the function.
  void addEntryBlock();

  //===--------------------------------------------------------------------===//
  // Argument Handling
  //===--------------------------------------------------------------------===//

  /// Returns number of arguments.
  unsigned getNumArguments() { return getType().getInputs().size(); }

  /// Gets argument.
  BlockArgument *getArgument(unsigned idx) {
    return getBlocks().front().getArgument(idx);
  }

  // Supports non-const operand iteration.
  using args_iterator = Block::args_iterator;
  args_iterator args_begin() { return front().args_begin(); }
  args_iterator args_end() { return front().args_end(); }
  llvm::iterator_range<args_iterator> getArguments() {
    return {args_begin(), args_end()};
  }

  //===--------------------------------------------------------------------===//
  // Argument Attributes
  //===--------------------------------------------------------------------===//

  /// FuncOp allows for attaching attributes to each of the respective function
  /// arguments. These argument attributes are stored as DictionaryAttrs in the
  /// main operation attribute dictionary. The name of these entries is `arg`
  /// followed by the index of the argument. These argument attribute
  /// dictionaries are optional, and will generally only exist if they are
  /// non-empty.

  /// Return all of the attributes for the argument at 'index'.
  ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
    auto argDict = getArgAttrDict(index);
    return argDict ? argDict.getValue() : llvm::None;
  }

  /// Return all argument attributes of this function.
  void getAllArgAttrs(SmallVectorImpl<NamedAttributeList> &result) {
    for (unsigned i = 0, e = getNumArguments(); i != e; ++i)
      result.emplace_back(getArgAttrDict(i));
  }

  /// Return the specified attribute, if present, for the argument at 'index',
  /// null otherwise.
  Attribute getArgAttr(unsigned index, Identifier name) {
    auto argDict = getArgAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }
  Attribute getArgAttr(unsigned index, StringRef name) {
    auto argDict = getArgAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }

  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, Identifier name) {
    return getArgAttr(index, name).dyn_cast_or_null<AttrClass>();
  }
  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, StringRef name) {
    return getArgAttr(index, name).dyn_cast_or_null<AttrClass>();
  }

  /// Set the attributes held by the argument at 'index'.
  void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes);
  void setArgAttrs(unsigned index, NamedAttributeList attributes);
  void setAllArgAttrs(ArrayRef<NamedAttributeList> attributes) {
    assert(attributes.size() == getNumArguments());
    for (unsigned i = 0, e = attributes.size(); i != e; ++i)
      setArgAttrs(i, attributes[i]);
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setArgAttr(unsigned index, Identifier name, Attribute value);
  void setArgAttr(unsigned index, StringRef name, Attribute value) {
    setArgAttr(index, Identifier::get(name, getContext()), value);
  }

  /// Remove the attribute 'name' from the argument at 'index'.
  NamedAttributeList::RemoveResult removeArgAttr(unsigned index,
                                                 Identifier name);

private:
  /// Returns the attribute entry name for the set of argument attributes at
  /// index 'arg'.
  static StringRef getArgAttrName(unsigned arg, SmallVectorImpl<char> &out);

  /// Returns the dictionary attribute corresponding to the argument at 'index'.
  /// If there are no argument attributes at 'index', a null attribute is
  /// returned.
  DictionaryAttr getArgAttrDict(unsigned index) {
    assert(index < getNumArguments() && "invalid argument number");
    SmallString<8> nameOut;
    return getAttrOfType<DictionaryAttr>(getArgAttrName(index, nameOut));
  }
};

/// Temporary forward declaration of FuncOp to the legacy Function.
using Function = FuncOp;
} // end namespace mlir

namespace llvm {

// Functions hash just like pointers.
template <> struct DenseMapInfo<mlir::FuncOp> {
  static mlir::FuncOp getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static mlir::FuncOp getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::FuncOp::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::FuncOp val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::FuncOp LHS, mlir::FuncOp RHS) { return LHS == RHS; }
};

/// Allow stealing the low bits of FuncOp.
template <> struct PointerLikeTypeTraits<mlir::FuncOp> {
public:
  static inline void *getAsVoidPointer(mlir::FuncOp I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::FuncOp getFromVoidPointer(void *P) {
    return mlir::FuncOp::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif // MLIR_IR_FUNCTION_H
