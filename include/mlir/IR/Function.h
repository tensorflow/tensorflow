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
class BlockAndValueMapping;
class FunctionType;
class Function;
class MLIRContext;
class Module;

namespace detail {
class ModuleStorage;

/// This class represents all of the internal state of a Function. This allows
/// for the Function class to be value typed.
class FunctionStorage
    : public llvm::ilist_node_with_parent<FunctionStorage, ModuleStorage> {
  FunctionStorage(Location location, StringRef name, FunctionType type,
                  ArrayRef<NamedAttribute> attrs = {});
  FunctionStorage(Location location, StringRef name, FunctionType type,
                  ArrayRef<NamedAttribute> attrs,
                  ArrayRef<NamedAttributeList> argAttrs);
  /// The name of the function.
  Identifier name;

  /// The module this function is embedded into.
  ModuleStorage *module = nullptr;

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

  friend struct llvm::ilist_traits<FunctionStorage>;
  friend Function;
};
} // namespace detail

/// This class represents an MLIR function, or the common unit of computation.
/// The region of a function is not allowed to implicitly capture global values,
/// and all external references must use Function arguments or attributes.
class Function {
public:
  Function(detail::FunctionStorage *impl = nullptr) : impl(impl) {}

  static Function create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {}) {
    return new detail::FunctionStorage(location, name, type, attrs);
  }
  static Function create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<NamedAttributeList> argAttrs) {
    return new detail::FunctionStorage(location, name, type, attrs, argAttrs);
  }

  /// Allow converting a Function to bool for null checks.
  operator bool() const { return impl; }
  bool operator==(Function other) const { return impl == other.impl; }
  bool operator!=(Function other) const { return !(*this == other); }

  /// The source location the function was defined or derived from.
  Location getLoc() { return impl->location; }

  /// Set the source location this function was defined or derived from.
  void setLoc(Location loc) { impl->location = loc; }

  /// Return the name of this function, without the @.
  Identifier getName() { return impl->name; }

  /// Return the type of this function.
  FunctionType getType() { return impl->type; }

  /// Change the type of this function in place. This is an extremely dangerous
  /// operation and it is up to the caller to ensure that this is legal for this
  /// function, and to restore invariants:
  ///   - the entry block args must be updated to match the function params.
  ///  - the arguments attributes may need an update: if the new type has less
  ///    parameters we drop the extra attributes, if there are more parameters
  ///    they won't have any attributes.
  void setType(FunctionType newType) {
    impl->type = newType;
    impl->argAttrs.resize(newType.getNumInputs());
  }

  MLIRContext *getContext();
  Module getModule();

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

  Region &getBody() { return impl->body; }
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

  //===--------------------------------------------------------------------===//
  // Operation Walkers
  //===--------------------------------------------------------------------===//

  /// Walk the operations in the function in postorder, calling the callback for
  /// each operation.
  void walk(llvm::function_ref<void(Operation *)> callback);

  /// Specialization of walk to only visit operations of 'OpTy'.
  template <typename OpTy> void walk(llvm::function_ref<void(OpTy)> callback) {
    walk([&](Operation *opInst) {
      if (auto op = dyn_cast<OpTy>(opInst))
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

  // Supports argument iteration.
  using args_iterator = Block::args_iterator;
  args_iterator args_begin() { return front().args_begin(); }
  args_iterator args_end() { return front().args_end(); }
  llvm::iterator_range<args_iterator> getArguments() {
    return {args_begin(), args_end()};
  }

  //===--------------------------------------------------------------------===//
  // Attributes
  //===--------------------------------------------------------------------===//

  /// Functions may optionally carry a list of attributes that associate
  /// constants to names.  Attributes may be dynamically added and removed over
  /// the lifetime of an function.

  /// Return all of the attributes on this function.
  ArrayRef<NamedAttribute> getAttrs() { return impl->attrs.getAttrs(); }

  /// Return the internal attribute list on this function.
  NamedAttributeList &getAttrList() { return impl->attrs; }

  /// Return all of the attributes for the argument at 'index'.
  ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
    assert(index < getNumArguments() && "invalid argument number");
    return impl->argAttrs[index].getAttrs();
  }

  /// Set the attributes held by this function.
  void setAttrs(ArrayRef<NamedAttribute> attributes) {
    impl->attrs.setAttrs(attributes);
  }

  /// Set the attributes held by the argument at 'index'.
  void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
    assert(index < getNumArguments() && "invalid argument number");
    impl->argAttrs[index].setAttrs(attributes);
  }
  void setArgAttrs(unsigned index, NamedAttributeList attributes) {
    assert(index < getNumArguments() && "invalid argument number");
    impl->argAttrs[index] = attributes;
  }
  void setAllArgAttrs(ArrayRef<NamedAttributeList> attributes) {
    assert(attributes.size() == getNumArguments());
    for (unsigned i = 0, e = attributes.size(); i != e; ++i)
      impl->argAttrs[i] = attributes[i];
  }

  /// Return all argument attributes of this function.
  void getAllArgAttrs(SmallVectorImpl<NamedAttributeList> &result) {
    result.assign(impl->argAttrs.begin(), impl->argAttrs.end());
  }

  /// Return the specified attribute if present, null otherwise.
  Attribute getAttr(Identifier name) { return impl->attrs.get(name); }
  Attribute getAttr(StringRef name) { return impl->attrs.get(name); }

  /// Return the specified attribute, if present, for the argument at 'index',
  /// null otherwise.
  Attribute getArgAttr(unsigned index, Identifier name) {
    assert(index < getNumArguments() && "invalid argument number");
    return impl->argAttrs[index].get(name);
  }
  Attribute getArgAttr(unsigned index, StringRef name) {
    assert(index < getNumArguments() && "invalid argument number");
    return impl->argAttrs[index].get(name);
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
    impl->attrs.set(name, value);
  }
  void setAttr(StringRef name, Attribute value) {
    setAttr(Identifier::get(name, getContext()), value);
  }
  void setArgAttr(unsigned index, Identifier name, Attribute value) {
    assert(index < getNumArguments() && "invalid argument number");
    impl->argAttrs[index].set(name, value);
  }
  void setArgAttr(unsigned index, StringRef name, Attribute value) {
    setArgAttr(index, Identifier::get(name, getContext()), value);
  }

  /// Remove the attribute with the specified name if it exists.  The return
  /// value indicates whether the attribute was present or not.
  NamedAttributeList::RemoveResult removeAttr(Identifier name) {
    return impl->attrs.remove(name);
  }
  NamedAttributeList::RemoveResult removeArgAttr(unsigned index,
                                                 Identifier name) {
    assert(index < getNumArguments() && "invalid argument number");
    return impl->attrs.remove(name);
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  void print(raw_ostream &os);
  void dump();

  /// Emit an error about fatal conditions with this function, reporting up to
  /// any diagnostic handlers that may be listening.
  InFlightDiagnostic emitError();
  InFlightDiagnostic emitError(const Twine &message);

  /// Emit a warning about this function, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitWarning();
  InFlightDiagnostic emitWarning(const Twine &message);

  /// Emit a remark about this function, reporting up to any diagnostic
  /// handlers that may be listening.
  InFlightDiagnostic emitRemark();
  InFlightDiagnostic emitRemark(const Twine &message);

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). If the mapper
  /// contains entries for function arguments, these arguments are not included
  /// in the new function. Replaces references to cloned sub-values with the
  /// corresponding value that is copied, and adds those mappings to the mapper.
  Function clone(BlockAndValueMapping &mapper);
  Function clone();

  /// Clone the internal blocks and attributes from this function into dest. Any
  /// cloned blocks are appended to the back of dest. This function asserts that
  /// the attributes of the current function and dest are compatible.
  void cloneInto(Function dest, BlockAndValueMapping &mapper);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const {
    return static_cast<const void *>(impl);
  }
  static Function getFromOpaquePointer(const void *pointer) {
    return reinterpret_cast<detail::FunctionStorage *>(
        const_cast<void *>(pointer));
  }

private:
  /// Set the name of this function.
  void setName(Identifier newName) { impl->name = newName; }

  /// A pointer to the impl storage instance for this function. This allows for
  /// 'Function' to be treated as a value type.
  detail::FunctionStorage *impl = nullptr;

  // Allow access to 'setName'.
  friend class SymbolTable;

  // Allow access to 'impl'.
  friend class Module;
  friend class Region;
};

//===--------------------------------------------------------------------===//
// Function Operation.
//===--------------------------------------------------------------------===//

/// FuncOp represents a function, or a named operation containing one region
/// that forms a CFG(Control Flow Graph). The region of a function is not
/// allowed to implicitly capture global values, and all external references
/// must use Function arguments or attributes.
class FuncOp : public Op<FuncOp, OpTrait::ZeroOperands, OpTrait::ZeroResult,
                         OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "func"; }

  static void build(Builder *builder, OperationState *result, StringRef name,
                    FunctionType type, ArrayRef<NamedAttribute> attrs);

  /// Operation hooks.
  static ParseResult parse(OpAsmParser *parser, OperationState *result);
  void print(OpAsmPrinter *p);
  LogicalResult verify();

  /// Returns the name of this function.
  StringRef getName() { return getAttrOfType<StringAttr>("name").getValue(); }

  /// Returns the type of this function.
  FunctionType getType() {
    return getAttrOfType<TypeAttr>("type").getValue().cast<FunctionType>();
  }

  /// Returns true if this function is external, i.e. it has no body.
  bool isExternal() { return empty(); }

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  Region &getBody() { return getOperation()->getRegion(0); }

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

} // end namespace mlir

//===----------------------------------------------------------------------===//
// ilist_traits for Function
//===----------------------------------------------------------------------===//

namespace llvm {

template <>
struct ilist_traits<::mlir::detail::FunctionStorage>
    : public ilist_alloc_traits<::mlir::detail::FunctionStorage> {
  using FunctionStorage = ::mlir::detail::FunctionStorage;
  using function_iterator = simple_ilist<FunctionStorage>::iterator;

  static void deleteNode(FunctionStorage *function) { delete function; }

  void addNodeToList(FunctionStorage *function);
  void removeNodeFromList(FunctionStorage *function);
  void transferNodesFromList(ilist_traits<FunctionStorage> &otherList,
                             function_iterator first, function_iterator last);

private:
  mlir::detail::ModuleStorage *getContainingModule();
};

// Functions hash just like pointers.
template <> struct DenseMapInfo<mlir::Function> {
  static mlir::Function getEmptyKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::Function::getFromOpaquePointer(pointer);
  }
  static mlir::Function getTombstoneKey() {
    auto pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::Function::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::Function val) {
    return hash_value(val.getAsOpaquePointer());
  }
  static bool isEqual(mlir::Function LHS, mlir::Function RHS) {
    return LHS == RHS;
  }
};

/// Allow stealing the low bits of FunctionStorage.
template <> struct PointerLikeTypeTraits<mlir::Function> {
public:
  static inline void *getAsVoidPointer(mlir::Function I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::Function getFromVoidPointer(void *P) {
    return mlir::Function::getFromOpaquePointer(P);
  }
  enum { NumLowBitsAvailable = 3 };
};

} // namespace llvm

#endif // MLIR_IR_FUNCTION_H
