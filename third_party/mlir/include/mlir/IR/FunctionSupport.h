//===- FunctionSupport.h - Utility types for function-like ops --*- C++ -*-===//
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
// This file defines support types for Operations that represent function-like
// constructs to use.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTIONSUPPORT_H
#define MLIR_IR_FUNCTIONSUPPORT_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallString.h"

namespace mlir {

namespace impl {

/// Return the name of the attribute used for function types.
inline StringRef getTypeAttrName() { return "type"; }

/// Return the name of the attribute used for function arguments.
inline StringRef getArgAttrName(unsigned arg, SmallVectorImpl<char> &out) {
  out.clear();
  return ("arg" + Twine(arg)).toStringRef(out);
}

/// Return the name of the attribute used for function results.
inline StringRef getResultAttrName(unsigned arg, SmallVectorImpl<char> &out) {
  out.clear();
  return ("result" + Twine(arg)).toStringRef(out);
}

/// Returns the dictionary attribute corresponding to the argument at 'index'.
/// If there are no argument attributes at 'index', a null attribute is
/// returned.
inline DictionaryAttr getArgAttrDict(Operation *op, unsigned index) {
  SmallString<8> nameOut;
  return op->getAttrOfType<DictionaryAttr>(getArgAttrName(index, nameOut));
}

/// Returns the dictionary attribute corresponding to the result at 'index'.
/// If there are no result attributes at 'index', a null attribute is
/// returned.
inline DictionaryAttr getResultAttrDict(Operation *op, unsigned index) {
  SmallString<8> nameOut;
  return op->getAttrOfType<DictionaryAttr>(getResultAttrName(index, nameOut));
}

/// Return all of the attributes for the argument at 'index'.
inline ArrayRef<NamedAttribute> getArgAttrs(Operation *op, unsigned index) {
  auto argDict = getArgAttrDict(op, index);
  return argDict ? argDict.getValue() : llvm::None;
}

/// Return all of the attributes for the result at 'index'.
inline ArrayRef<NamedAttribute> getResultAttrs(Operation *op, unsigned index) {
  auto resultDict = getResultAttrDict(op, index);
  return resultDict ? resultDict.getValue() : llvm::None;
}

} // namespace impl

namespace OpTrait {

/// This trait provides APIs for Ops that behave like functions.  In particular:
/// - Ops must be symbols, i.e. also have the `Symbol` trait;
/// - Ops have a single region with multiple blocks that corresponds to the body
///   of the function;
/// - the absence of a region corresponds to an external function;
/// - leading arguments of the first block of the region are treated as function
///   arguments;
/// - they can have argument attributes that are stored in a dictionary
///   attribute on the Op itself.
/// This trait does *NOT* provide type support for the functions, meaning that
/// concrete Ops must handle the type of the declared or defined function.
/// `getTypeAttrName()` is a convenience function that returns the name of the
/// attribute that can be used to store the function type, but the trait makes
/// no assumption based on it.
///
/// - Concrete ops *must* define a member function `getNumFuncArguments()` that
///   returns the number of function arguments based exclusively on type (so
///   that it can be called on function declarations).
/// - Concrete ops *must* define a member function `getNumFuncResults()` that
///   returns the number of function results based exclusively on type (so that
///   it can be called on function declarations).
/// - To verify that the type respects op-specific invariants, concrete ops may
///   redefine the `verifyType()` hook that will be called after verifying the
///   presence of the `type` attribute and before any call to
///   `getNumFuncArguments`/`getNumFuncResults` from the verifier.
/// - To verify that the body respects op-specific invariants, concrete ops may
///   redefine the `verifyBody()` hook that will be called after verifying the
///   function type and the presence of the (potentially empty) body region.
template <typename ConcreteType>
class FunctionLike : public OpTrait::TraitBase<ConcreteType, FunctionLike> {
public:
  /// Verify that all of the argument attributes are dialect attributes.
  static LogicalResult verifyTrait(Operation *op);

  //===--------------------------------------------------------------------===//
  // Body Handling
  //===--------------------------------------------------------------------===//

  /// Returns true if this function is external, i.e. it has no body.
  bool isExternal() { return empty(); }

  Region &getBody() { return this->getOperation()->getRegion(0); }

  /// Delete all blocks from this function.
  void eraseBody() {
    getBody().dropAllReferences();
    getBody().getBlocks().clear();
  }

  /// This is the list of blocks in the function.
  using BlockListType = Region::BlockListType;
  BlockListType &getBlocks() { return getBody().getBlocks(); }

  // Iteration over the block in the function.
  using iterator = BlockListType::iterator;
  using reverse_iterator = BlockListType::reverse_iterator;

  iterator begin() { return getBody().begin(); }
  iterator end() { return getBody().end(); }
  reverse_iterator rbegin() { return getBody().rbegin(); }
  reverse_iterator rend() { return getBody().rend(); }

  bool empty() { return getBody().empty(); }
  void push_back(Block *block) { getBody().push_back(block); }
  void push_front(Block *block) { getBody().push_front(block); }

  Block &back() { return getBody().back(); }
  Block &front() { return getBody().front(); }

  /// Hook for concrete ops to verify the contents of the body. Called as a
  /// part of trait verification, after type verification and ensuring that a
  /// region exists.
  LogicalResult verifyBody();

  //===--------------------------------------------------------------------===//
  // Type Attribute Handling
  //===--------------------------------------------------------------------===//

  /// Return the name of the attribute used for function types.
  static StringRef getTypeAttrName() { return ::mlir::impl::getTypeAttrName(); }

  TypeAttr getTypeAttr() {
    return this->getOperation()->template getAttrOfType<TypeAttr>(
        getTypeAttrName());
  }

  bool isTypeAttrValid() {
    auto typeAttr = getTypeAttr();
    if (!typeAttr)
      return false;
    return typeAttr.getValue() != Type{};
  }

  //===--------------------------------------------------------------------===//
  // Argument Handling
  //===--------------------------------------------------------------------===//

  unsigned getNumArguments() {
    return static_cast<ConcreteType *>(this)->getNumFuncArguments();
  }

  unsigned getNumResults() {
    return static_cast<ConcreteType *>(this)->getNumFuncResults();
  }

  /// Gets argument.
  BlockArgument *getArgument(unsigned idx) {
    return getBlocks().front().getArgument(idx);
  }

  // Supports non-const operand iteration.
  using args_iterator = Block::args_iterator;
  args_iterator args_begin() { return front().args_begin(); }
  args_iterator args_end() { return front().args_end(); }
  iterator_range<args_iterator> getArguments() {
    return {args_begin(), args_end()};
  }

  //===--------------------------------------------------------------------===//
  // Argument Attributes
  //===--------------------------------------------------------------------===//

  /// FunctionLike operations allow for attaching attributes to each of the
  /// respective function arguments. These argument attributes are stored as
  /// DictionaryAttrs in the main operation attribute dictionary. The name of
  /// these entries is `arg` followed by the index of the argument. These
  /// argument attribute dictionaries are optional, and will generally only
  /// exist if they are non-empty.

  /// Return all of the attributes for the argument at 'index'.
  ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
    return ::mlir::impl::getArgAttrs(this->getOperation(), index);
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
    return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
  }
  template <typename AttrClass>
  AttrClass getArgAttrOfType(unsigned index, StringRef name) {
    return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
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
    setArgAttr(index, Identifier::get(name, this->getOperation()->getContext()),
               value);
  }

  /// Remove the attribute 'name' from the argument at 'index'.
  NamedAttributeList::RemoveResult removeArgAttr(unsigned index,
                                                 Identifier name);

  //===--------------------------------------------------------------------===//
  // Result Attributes
  //===--------------------------------------------------------------------===//

  /// FunctionLike operations allow for attaching attributes to each of the
  /// respective function results. These result attributes are stored as
  /// DictionaryAttrs in the main operation attribute dictionary. The name of
  /// these entries is `result` followed by the index of the result. These
  /// result attribute dictionaries are optional, and will generally only
  /// exist if they are non-empty.

  /// Return all of the attributes for the result at 'index'.
  ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
    return ::mlir::impl::getResultAttrs(this->getOperation(), index);
  }

  /// Return all result attributes of this function.
  void getAllResultAttrs(SmallVectorImpl<NamedAttributeList> &result) {
    for (unsigned i = 0, e = getNumResults(); i != e; ++i)
      result.emplace_back(getResultAttrDict(i));
  }

  /// Return the specified attribute, if present, for the result at 'index',
  /// null otherwise.
  Attribute getResultAttr(unsigned index, Identifier name) {
    auto argDict = getResultAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }
  Attribute getResultAttr(unsigned index, StringRef name) {
    auto argDict = getResultAttrDict(index);
    return argDict ? argDict.get(name) : nullptr;
  }

  template <typename AttrClass>
  AttrClass getResultAttrOfType(unsigned index, Identifier name) {
    return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
  }
  template <typename AttrClass>
  AttrClass getResultAttrOfType(unsigned index, StringRef name) {
    return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
  }

  /// Set the attributes held by the result at 'index'.
  void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes);
  void setResultAttrs(unsigned index, NamedAttributeList attributes);
  void setAllResultAttrs(ArrayRef<NamedAttributeList> attributes) {
    assert(attributes.size() == getNumResults());
    for (unsigned i = 0, e = attributes.size(); i != e; ++i)
      setResultAttrs(i, attributes[i]);
  }

  /// If the an attribute exists with the specified name, change it to the new
  /// value. Otherwise, add a new attribute with the specified name/value.
  void setResultAttr(unsigned index, Identifier name, Attribute value);
  void setResultAttr(unsigned index, StringRef name, Attribute value) {
    setResultAttr(index,
                  Identifier::get(name, this->getOperation()->getContext()),
                  value);
  }

  /// Remove the attribute 'name' from the result at 'index'.
  NamedAttributeList::RemoveResult removeResultAttr(unsigned index,
                                                    Identifier name);

protected:
  /// Returns the attribute entry name for the set of argument attributes at
  /// 'index'.
  static StringRef getArgAttrName(unsigned index, SmallVectorImpl<char> &out) {
    return ::mlir::impl::getArgAttrName(index, out);
  }

  /// Returns the dictionary attribute corresponding to the argument at 'index'.
  /// If there are no argument attributes at 'index', a null attribute is
  /// returned.
  DictionaryAttr getArgAttrDict(unsigned index) {
    assert(index < getNumArguments() && "invalid argument number");
    return ::mlir::impl::getArgAttrDict(this->getOperation(), index);
  }

  /// Returns the attribute entry name for the set of result attributes at
  /// 'index'.
  static StringRef getResultAttrName(unsigned index,
                                     SmallVectorImpl<char> &out) {
    return ::mlir::impl::getResultAttrName(index, out);
  }

  /// Returns the dictionary attribute corresponding to the result at 'index'.
  /// If there are no result attributes at 'index', a null attribute is
  /// returned.
  DictionaryAttr getResultAttrDict(unsigned index) {
    assert(index < getNumResults() && "invalid result number");
    return ::mlir::impl::getResultAttrDict(this->getOperation(), index);
  }

  /// Hook for concrete classes to verify that the type attribute respects
  /// op-specific invariants.  Default implementation always succeeds.
  LogicalResult verifyType() { return success(); }
};

/// Default verifier checks that if the entry block exists, it has the same
/// number of arguments as the function-like operation.
template <typename ConcreteType>
LogicalResult FunctionLike<ConcreteType>::verifyBody() {
  auto funcOp = cast<ConcreteType>(this->getOperation());

  if (funcOp.isExternal())
    return success();

  unsigned numArguments = funcOp.getNumArguments();
  if (funcOp.front().getNumArguments() != numArguments)
    return funcOp.emitOpError("entry block must have ")
           << numArguments << " arguments to match function signature";

  return success();
}

template <typename ConcreteType>
LogicalResult FunctionLike<ConcreteType>::verifyTrait(Operation *op) {
  MLIRContext *ctx = op->getContext();
  auto funcOp = cast<ConcreteType>(op);

  if (!funcOp.isTypeAttrValid())
    return funcOp.emitOpError("requires a type attribute '")
           << getTypeAttrName() << '\'';

  if (failed(funcOp.verifyType()))
    return failure();

  for (unsigned i = 0, e = funcOp.getNumArguments(); i != e; ++i) {
    // Verify that all of the argument attributes are dialect attributes, i.e.
    // that they contain a dialect prefix in their name.  Call the dialect, if
    // registered, to verify the attributes themselves.
    for (auto attr : funcOp.getArgAttrs(i)) {
      if (!attr.first.strref().contains('.'))
        return funcOp.emitOpError("arguments may only have dialect attributes");
      auto dialectNamePair = attr.first.strref().split('.');
      if (auto *dialect = ctx->getRegisteredDialect(dialectNamePair.first)) {
        if (failed(dialect->verifyRegionArgAttribute(op, /*regionIndex=*/0,
                                                     /*argIndex=*/i, attr)))
          return failure();
      }
    }
  }

  for (unsigned i = 0, e = funcOp.getNumResults(); i != e; ++i) {
    // Verify that all of the result attributes are dialect attributes, i.e.
    // that they contain a dialect prefix in their name.  Call the dialect, if
    // registered, to verify the attributes themselves.
    for (auto attr : funcOp.getResultAttrs(i)) {
      if (!attr.first.strref().contains('.'))
        return funcOp.emitOpError("results may only have dialect attributes");
      auto dialectNamePair = attr.first.strref().split('.');
      if (auto *dialect = ctx->getRegisteredDialect(dialectNamePair.first)) {
        if (failed(dialect->verifyRegionResultAttribute(op, /*regionIndex=*/0,
                                                        /*resultIndex=*/i,
                                                        attr)))
          return failure();
      }
    }
  }

  // Check that the op has exactly one region for the body.
  if (op->getNumRegions() != 1)
    return funcOp.emitOpError("expects one region");

  return funcOp.verifyBody();
}

//===----------------------------------------------------------------------===//
// Function Argument Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the argument at 'index'.
template <typename ConcreteType>
void FunctionLike<ConcreteType>::setArgAttrs(
    unsigned index, ArrayRef<NamedAttribute> attributes) {
  assert(index < getNumArguments() && "invalid argument number");
  SmallString<8> nameOut;
  getArgAttrName(index, nameOut);

  if (attributes.empty())
    return (void)static_cast<ConcreteType *>(this)->removeAttr(nameOut);
  Operation *op = this->getOperation();
  op->setAttr(nameOut, DictionaryAttr::get(attributes, op->getContext()));
}

template <typename ConcreteType>
void FunctionLike<ConcreteType>::setArgAttrs(unsigned index,
                                             NamedAttributeList attributes) {
  assert(index < getNumArguments() && "invalid argument number");
  SmallString<8> nameOut;
  if (auto newAttr = attributes.getDictionary())
    return this->getOperation()->setAttr(getArgAttrName(index, nameOut),
                                         newAttr);
  static_cast<ConcreteType *>(this)->removeAttr(getArgAttrName(index, nameOut));
}

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void FunctionLike<ConcreteType>::setArgAttr(unsigned index, Identifier name,
                                            Attribute value) {
  auto curAttr = getArgAttrDict(index);
  NamedAttributeList attrList(curAttr);
  attrList.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (curAttr != attrList.getDictionary())
    setArgAttrs(index, attrList);
}

/// Remove the attribute 'name' from the argument at 'index'.
template <typename ConcreteType>
NamedAttributeList::RemoveResult
FunctionLike<ConcreteType>::removeArgAttr(unsigned index, Identifier name) {
  // Build an attribute list and remove the attribute at 'name'.
  NamedAttributeList attrList(getArgAttrDict(index));
  auto result = attrList.remove(name);

  // If the attribute was removed, then update the argument dictionary.
  if (result == NamedAttributeList::RemoveResult::Removed)
    setArgAttrs(index, attrList);
  return result;
}

//===----------------------------------------------------------------------===//
// Function Result Attribute.
//===----------------------------------------------------------------------===//

/// Set the attributes held by the result at 'index'.
template <typename ConcreteType>
void FunctionLike<ConcreteType>::setResultAttrs(
    unsigned index, ArrayRef<NamedAttribute> attributes) {
  assert(index < getNumResults() && "invalid result number");
  SmallString<8> nameOut;
  getResultAttrName(index, nameOut);

  if (attributes.empty())
    return (void)static_cast<ConcreteType *>(this)->removeAttr(nameOut);
  Operation *op = this->getOperation();
  op->setAttr(nameOut, DictionaryAttr::get(attributes, op->getContext()));
}

template <typename ConcreteType>
void FunctionLike<ConcreteType>::setResultAttrs(unsigned index,
                                                NamedAttributeList attributes) {
  assert(index < getNumResults() && "invalid result number");
  SmallString<8> nameOut;
  if (auto newAttr = attributes.getDictionary())
    return this->getOperation()->setAttr(getResultAttrName(index, nameOut),
                                         newAttr);
  static_cast<ConcreteType *>(this)->removeAttr(
      getResultAttrName(index, nameOut));
}

/// If the an attribute exists with the specified name, change it to the new
/// value. Otherwise, add a new attribute with the specified name/value.
template <typename ConcreteType>
void FunctionLike<ConcreteType>::setResultAttr(unsigned index, Identifier name,
                                               Attribute value) {
  auto curAttr = getResultAttrDict(index);
  NamedAttributeList attrList(curAttr);
  attrList.set(name, value);

  // If the attribute changed, then set the new arg attribute list.
  if (curAttr != attrList.getDictionary())
    setResultAttrs(index, attrList);
}

/// Remove the attribute 'name' from the result at 'index'.
template <typename ConcreteType>
NamedAttributeList::RemoveResult
FunctionLike<ConcreteType>::removeResultAttr(unsigned index, Identifier name) {
  // Build an attribute list and remove the attribute at 'name'.
  NamedAttributeList attrList(getResultAttrDict(index));
  auto result = attrList.remove(name);

  // If the attribute was removed, then update the result dictionary.
  if (result == NamedAttributeList::RemoveResult::Removed)
    setResultAttrs(index, attrList);
  return result;
}

} // end namespace OpTrait

} // end namespace mlir

#endif // MLIR_IR_FUNCTIONSUPPORT_H
