//===- Attributes.cpp - MLIR Affine Expr Classes --------------------------===//
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

#include "mlir/IR/Attributes.h"
#include "AttributeDetail.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

AttributeStorage::AttributeStorage(Type type)
    : type(type.getAsOpaquePointer()) {}
AttributeStorage::AttributeStorage() : type(nullptr) {}

Type AttributeStorage::getType() const {
  return Type::getFromOpaquePointer(type);
}
void AttributeStorage::setType(Type newType) {
  type = newType.getAsOpaquePointer();
}

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the type of this attribute.
Type Attribute::getType() const { return impl->getType(); }

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getType().getContext(); }

/// Get the dialect this attribute is registered to.
Dialect &Attribute::getDialect() const { return impl->getDialect(); }

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

AffineMapAttr AffineMapAttr::get(AffineMap value) {
  return Base::get(value.getContext(), StandardAttributes::AffineMap, value);
}

AffineMap AffineMapAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

ArrayAttr ArrayAttr::get(ArrayRef<Attribute> value, MLIRContext *context) {
  return Base::get(context, StandardAttributes::Array, value);
}

ArrayRef<Attribute> ArrayAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

bool BoolAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// DictionaryAttr
//===----------------------------------------------------------------------===//

/// Perform a three-way comparison between the names of the specified
/// NamedAttributes.
static int compareNamedAttributes(const NamedAttribute *lhs,
                                  const NamedAttribute *rhs) {
  return lhs->first.strref().compare(rhs->first.strref());
}

DictionaryAttr DictionaryAttr::get(ArrayRef<NamedAttribute> value,
                                   MLIRContext *context) {
  assert(llvm::all_of(value,
                      [](const NamedAttribute &attr) { return attr.second; }) &&
         "value cannot have null entries");

  // We need to sort the element list to canonicalize it, but we also don't want
  // to do a ton of work in the super common case where the element list is
  // already sorted.
  SmallVector<NamedAttribute, 8> storage;
  switch (value.size()) {
  case 0:
    break;
  case 1:
    // A single element is already sorted.
    break;
  case 2:
    assert(value[0].first != value[1].first &&
           "DictionaryAttr element names must be unique");

    // Don't invoke a general sort for two element case.
    if (value[0].first.strref() > value[1].first.strref()) {
      storage.push_back(value[1]);
      storage.push_back(value[0]);
      value = storage;
    }
    break;
  default:
    // Check to see they are sorted already.
    bool isSorted = true;
    for (unsigned i = 0, e = value.size() - 1; i != e; ++i) {
      if (value[i].first.strref() > value[i + 1].first.strref()) {
        isSorted = false;
        break;
      }
    }
    // If not, do a general sort.
    if (!isSorted) {
      storage.append(value.begin(), value.end());
      llvm::array_pod_sort(storage.begin(), storage.end(),
                           compareNamedAttributes);
      value = storage;
    }

    // Ensure that the attribute elements are unique.
    assert(std::adjacent_find(value.begin(), value.end(),
                              [](NamedAttribute l, NamedAttribute r) {
                                return l.first == r.first;
                              }) == value.end() &&
           "DictionaryAttr element names must be unique");
  }

  return Base::get(context, StandardAttributes::Dictionary, value);
}

ArrayRef<NamedAttribute> DictionaryAttr::getValue() const {
  return getImpl()->getElements();
}

/// Return the specified attribute if present, null otherwise.
Attribute DictionaryAttr::get(StringRef name) const {
  ArrayRef<NamedAttribute> values = getValue();
  auto compare = [](NamedAttribute attr, StringRef name) {
    return attr.first.strref() < name;
  };
  auto it = llvm::lower_bound(values, name, compare);
  return it != values.end() && it->first.is(name) ? it->second : Attribute();
}
Attribute DictionaryAttr::get(Identifier name) const {
  for (auto elt : getValue())
    if (elt.first == name)
      return elt.second;
  return nullptr;
}

DictionaryAttr::iterator DictionaryAttr::begin() const {
  return getValue().begin();
}
DictionaryAttr::iterator DictionaryAttr::end() const {
  return getValue().end();
}
size_t DictionaryAttr::size() const { return getValue().size(); }

//===----------------------------------------------------------------------===//
// FloatAttr
//===----------------------------------------------------------------------===//

FloatAttr FloatAttr::get(Type type, double value) {
  return Base::get(type.getContext(), StandardAttributes::Float, type, value);
}

FloatAttr FloatAttr::getChecked(Type type, double value, Location loc) {
  return Base::getChecked(loc, type.getContext(), StandardAttributes::Float,
                          type, value);
}

FloatAttr FloatAttr::get(Type type, const APFloat &value) {
  return Base::get(type.getContext(), StandardAttributes::Float, type, value);
}

FloatAttr FloatAttr::getChecked(Type type, const APFloat &value, Location loc) {
  return Base::getChecked(loc, type.getContext(), StandardAttributes::Float,
                          type, value);
}

APFloat FloatAttr::getValue() const { return getImpl()->getValue(); }

double FloatAttr::getValueAsDouble() const {
  return getValueAsDouble(getValue());
}
double FloatAttr::getValueAsDouble(APFloat value) {
  if (&value.getSemantics() != &APFloat::IEEEdouble()) {
    bool losesInfo = false;
    value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                  &losesInfo);
  }
  return value.convertToDouble();
}

/// Verify construction invariants.
static LogicalResult verifyFloatTypeInvariants(Optional<Location> loc,
                                               Type type) {
  if (!type.isa<FloatType>())
    return emitOptionalError(loc, "expected floating point type");
  return success();
}

LogicalResult FloatAttr::verifyConstructionInvariants(Optional<Location> loc,
                                                      MLIRContext *ctx,
                                                      Type type, double value) {
  return verifyFloatTypeInvariants(loc, type);
}

LogicalResult FloatAttr::verifyConstructionInvariants(Optional<Location> loc,
                                                      MLIRContext *ctx,
                                                      Type type,
                                                      const APFloat &value) {
  // Verify that the type is correct.
  if (failed(verifyFloatTypeInvariants(loc, type)))
    return failure();

  // Verify that the type semantics match that of the value.
  if (&type.cast<FloatType>().getFloatSemantics() != &value.getSemantics()) {
    return emitOptionalError(
        loc, "FloatAttr type doesn't match the type implied by its value");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolRefAttr
//===----------------------------------------------------------------------===//

FlatSymbolRefAttr SymbolRefAttr::get(StringRef value, MLIRContext *ctx) {
  return Base::get(ctx, StandardAttributes::SymbolRef, value, llvm::None)
      .cast<FlatSymbolRefAttr>();
}

SymbolRefAttr SymbolRefAttr::get(StringRef value,
                                 ArrayRef<FlatSymbolRefAttr> nestedReferences,
                                 MLIRContext *ctx) {
  return Base::get(ctx, StandardAttributes::SymbolRef, value, nestedReferences);
}

StringRef SymbolRefAttr::getRootReference() const { return getImpl()->value; }

StringRef SymbolRefAttr::getLeafReference() const {
  ArrayRef<FlatSymbolRefAttr> nestedRefs = getNestedReferences();
  return nestedRefs.empty() ? getRootReference() : nestedRefs.back().getValue();
}

ArrayRef<FlatSymbolRefAttr> SymbolRefAttr::getNestedReferences() const {
  return getImpl()->getNestedRefs();
}

//===----------------------------------------------------------------------===//
// IntegerAttr
//===----------------------------------------------------------------------===//

IntegerAttr IntegerAttr::get(Type type, const APInt &value) {
  return Base::get(type.getContext(), StandardAttributes::Integer, type, value);
}

IntegerAttr IntegerAttr::get(Type type, int64_t value) {
  // This uses 64 bit APInts by default for index type.
  if (type.isIndex())
    return get(type, APInt(64, value));

  auto intType = type.cast<IntegerType>();
  return get(type, APInt(intType.getWidth(), value));
}

APInt IntegerAttr::getValue() const { return getImpl()->getValue(); }

int64_t IntegerAttr::getInt() const { return getValue().getSExtValue(); }

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  return Base::get(value.getConstraint(0).getContext(),
                   StandardAttributes::IntegerSet, value);
}

IntegerSet IntegerSetAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

OpaqueAttr OpaqueAttr::get(Identifier dialect, StringRef attrData, Type type,
                           MLIRContext *context) {
  return Base::get(context, StandardAttributes::Opaque, dialect, attrData,
                   type);
}

OpaqueAttr OpaqueAttr::getChecked(Identifier dialect, StringRef attrData,
                                  Type type, Location location) {
  return Base::getChecked(location, type.getContext(),
                          StandardAttributes::Opaque, dialect, attrData, type);
}

/// Returns the dialect namespace of the opaque attribute.
Identifier OpaqueAttr::getDialectNamespace() const {
  return getImpl()->dialectNamespace;
}

/// Returns the raw attribute data of the opaque attribute.
StringRef OpaqueAttr::getAttrData() const { return getImpl()->attrData; }

/// Verify the construction of an opaque attribute.
LogicalResult OpaqueAttr::verifyConstructionInvariants(Optional<Location> loc,
                                                       MLIRContext *context,
                                                       Identifier dialect,
                                                       StringRef attrData,
                                                       Type type) {
  if (!Dialect::isValidNamespace(dialect.strref()))
    return emitOptionalError(loc, "invalid dialect namespace '", dialect, "'");
  return success();
}

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

StringAttr StringAttr::get(StringRef bytes, MLIRContext *context) {
  return get(bytes, NoneType::get(context));
}

/// Get an instance of a StringAttr with the given string and Type.
StringAttr StringAttr::get(StringRef bytes, Type type) {
  return Base::get(type.getContext(), StandardAttributes::String, bytes, type);
}

StringRef StringAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

TypeAttr TypeAttr::get(Type value) {
  return Base::get(value.getContext(), StandardAttributes::Type, value);
}

Type TypeAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

ShapedType ElementsAttr::getType() const {
  return Attribute::getType().cast<ShapedType>();
}

/// Returns the number of elements held by this attribute.
int64_t ElementsAttr::getNumElements() const {
  return getType().getNumElements();
}

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute ElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  switch (getKind()) {
  case StandardAttributes::DenseElements:
    return cast<DenseElementsAttr>().getValue(index);
  case StandardAttributes::OpaqueElements:
    return cast<OpaqueElementsAttr>().getValue(index);
  case StandardAttributes::SparseElements:
    return cast<SparseElementsAttr>().getValue(index);
  default:
    llvm_unreachable("unknown ElementsAttr kind");
  }
}

/// Return if the given 'index' refers to a valid element in this attribute.
bool ElementsAttr::isValidIndex(ArrayRef<uint64_t> index) const {
  auto type = getType();

  // Verify that the rank of the indices matches the held type.
  auto rank = type.getRank();
  if (rank != static_cast<int64_t>(index.size()))
    return false;

  // Verify that all of the indices are within the shape dimensions.
  auto shape = type.getShape();
  return llvm::all_of(llvm::seq<int>(0, rank), [&](int i) {
    return static_cast<int64_t>(index[i]) < shape[i];
  });
}

ElementsAttr
ElementsAttr::mapValues(Type newElementType,
                        function_ref<APInt(const APInt &)> mapping) const {
  switch (getKind()) {
  case StandardAttributes::DenseElements:
    return cast<DenseElementsAttr>().mapValues(newElementType, mapping);
  default:
    llvm_unreachable("unsupported ElementsAttr subtype");
  }
}

ElementsAttr
ElementsAttr::mapValues(Type newElementType,
                        function_ref<APInt(const APFloat &)> mapping) const {
  switch (getKind()) {
  case StandardAttributes::DenseElements:
    return cast<DenseElementsAttr>().mapValues(newElementType, mapping);
  default:
    llvm_unreachable("unsupported ElementsAttr subtype");
  }
}

/// Returns the 1 dimensional flattened row-major index from the given
/// multi-dimensional index.
uint64_t ElementsAttr::getFlattenedIndex(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  auto type = getType();

  // Reduce the provided multidimensional index into a flattended 1D row-major
  // index.
  auto rank = type.getRank();
  auto shape = type.getShape();
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }
  return valueIndex;
}

//===----------------------------------------------------------------------===//
// DenseElementAttr Utilities
//===----------------------------------------------------------------------===//

static size_t getDenseElementBitwidth(Type eltType) {
  // FIXME(b/121118307): using 64 bits for BF16 because it is currently stored
  // with double semantics.
  return eltType.isBF16() ? 64 : eltType.getIntOrFloatBitWidth();
}

/// Get the bitwidth of a dense element type within the buffer.
/// DenseElementsAttr requires bitwidths greater than 1 to be aligned by 8.
static size_t getDenseElementStorageWidth(size_t origWidth) {
  return origWidth == 1 ? origWidth : llvm::alignTo<8>(origWidth);
}

/// Set a bit to a specific value.
static void setBit(char *rawData, size_t bitPos, bool value) {
  if (value)
    rawData[bitPos / CHAR_BIT] |= (1 << (bitPos % CHAR_BIT));
  else
    rawData[bitPos / CHAR_BIT] &= ~(1 << (bitPos % CHAR_BIT));
}

/// Return the value of the specified bit.
static bool getBit(const char *rawData, size_t bitPos) {
  return (rawData[bitPos / CHAR_BIT] & (1 << (bitPos % CHAR_BIT))) != 0;
}

/// Writes value to the bit position `bitPos` in array `rawData`.
static void writeBits(char *rawData, size_t bitPos, APInt value) {
  size_t bitWidth = value.getBitWidth();

  // If the bitwidth is 1 we just toggle the specific bit.
  if (bitWidth == 1)
    return setBit(rawData, bitPos, value.isOneValue());

  // Otherwise, the bit position is guaranteed to be byte aligned.
  assert((bitPos % CHAR_BIT) == 0 && "expected bitPos to be 8-bit aligned");
  std::copy_n(reinterpret_cast<const char *>(value.getRawData()),
              llvm::divideCeil(bitWidth, CHAR_BIT),
              rawData + (bitPos / CHAR_BIT));
}

/// Reads the next `bitWidth` bits from the bit position `bitPos` in array
/// `rawData`.
static APInt readBits(const char *rawData, size_t bitPos, size_t bitWidth) {
  // Handle a boolean bit position.
  if (bitWidth == 1)
    return APInt(1, getBit(rawData, bitPos) ? 1 : 0);

  // Otherwise, the bit position must be 8-bit aligned.
  assert((bitPos % CHAR_BIT) == 0 && "expected bitPos to be 8-bit aligned");
  APInt result(bitWidth, 0);
  std::copy_n(
      rawData + (bitPos / CHAR_BIT), llvm::divideCeil(bitWidth, CHAR_BIT),
      const_cast<char *>(reinterpret_cast<const char *>(result.getRawData())));
  return result;
}

/// Returns if 'values' corresponds to a splat, i.e. one element, or has the
/// same element count as 'type'.
template <typename Values>
static bool hasSameElementsOrSplat(ShapedType type, const Values &values) {
  return (values.size() == 1) ||
         (type.getNumElements() == static_cast<int64_t>(values.size()));
}

//===----------------------------------------------------------------------===//
// DenseElementAttr Iterators
//===----------------------------------------------------------------------===//

/// Constructs a new iterator.
DenseElementsAttr::AttributeElementIterator::AttributeElementIterator(
    DenseElementsAttr attr, size_t index)
    : indexed_accessor_iterator<AttributeElementIterator, const void *,
                                Attribute, Attribute, Attribute>(
          attr.getAsOpaquePointer(), index) {}

/// Accesses the Attribute value at this iterator position.
Attribute DenseElementsAttr::AttributeElementIterator::operator*() const {
  auto owner = getFromOpaquePointer(base).cast<DenseElementsAttr>();
  Type eltTy = owner.getType().getElementType();
  if (auto intEltTy = eltTy.dyn_cast<IntegerType>()) {
    if (intEltTy.getWidth() == 1)
      return BoolAttr::get((*IntElementIterator(owner, index)).isOneValue(),
                           owner.getContext());
    return IntegerAttr::get(eltTy, *IntElementIterator(owner, index));
  }
  if (auto floatEltTy = eltTy.dyn_cast<FloatType>()) {
    IntElementIterator intIt(owner, index);
    FloatElementIterator floatIt(floatEltTy.getFloatSemantics(), intIt);
    return FloatAttr::get(eltTy, *floatIt);
  }
  llvm_unreachable("unexpected element type");
}

/// Constructs a new iterator.
DenseElementsAttr::BoolElementIterator::BoolElementIterator(
    DenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<BoolElementIterator, bool, bool, bool>(
          attr.getRawData().data(), attr.isSplat(), dataIndex) {}

/// Accesses the bool value at this iterator position.
bool DenseElementsAttr::BoolElementIterator::operator*() const {
  return getBit(getData(), getDataIndex());
}

/// Constructs a new iterator.
DenseElementsAttr::IntElementIterator::IntElementIterator(
    DenseElementsAttr attr, size_t dataIndex)
    : DenseElementIndexedIteratorImpl<IntElementIterator, APInt, APInt, APInt>(
          attr.getRawData().data(), attr.isSplat(), dataIndex),
      bitWidth(getDenseElementBitwidth(attr.getType().getElementType())) {}

/// Accesses the raw APInt value at this iterator position.
APInt DenseElementsAttr::IntElementIterator::operator*() const {
  return readBits(getData(),
                  getDataIndex() * getDenseElementStorageWidth(bitWidth),
                  bitWidth);
}

DenseElementsAttr::FloatElementIterator::FloatElementIterator(
    const llvm::fltSemantics &smt, IntElementIterator it)
    : llvm::mapped_iterator<IntElementIterator,
                            std::function<APFloat(const APInt &)>>(
          it, [&](const APInt &val) { return APFloat(smt, val); }) {}

//===----------------------------------------------------------------------===//
// DenseElementsAttr
//===----------------------------------------------------------------------===//

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<Attribute> values) {
  assert(type.getElementType().isIntOrFloat() &&
         "expected int or float element type");
  assert(hasSameElementsOrSplat(type, values));

  auto eltType = type.getElementType();
  size_t bitWidth = getDenseElementBitwidth(eltType);
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

  // Compress the attribute values into a character buffer.
  SmallVector<char, 8> data(llvm::divideCeil(storageBitWidth, CHAR_BIT) *
                            values.size());
  APInt intVal;
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    assert(eltType == values[i].getType() &&
           "expected attribute value to have element type");

    switch (eltType.getKind()) {
    case StandardTypes::BF16:
    case StandardTypes::F16:
    case StandardTypes::F32:
    case StandardTypes::F64:
      intVal = values[i].cast<FloatAttr>().getValue().bitcastToAPInt();
      break;
    case StandardTypes::Integer:
      intVal = values[i].isa<BoolAttr>()
                   ? APInt(1, values[i].cast<BoolAttr>().getValue() ? 1 : 0)
                   : values[i].cast<IntegerAttr>().getValue();
      break;
    default:
      llvm_unreachable("unexpected element type");
    }
    assert(intVal.getBitWidth() == bitWidth &&
           "expected value to have same bitwidth as element type");
    writeBits(data.data(), i * storageBitWidth, intVal);
  }
  return getRaw(type, data, /*isSplat=*/(values.size() == 1));
}

DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<bool> values) {
  assert(hasSameElementsOrSplat(type, values));
  assert(type.getElementType().isInteger(1));

  std::vector<char> buff(llvm::divideCeil(values.size(), CHAR_BIT));
  for (int i = 0, e = values.size(); i != e; ++i)
    setBit(buff.data(), i, values[i]);
  return getRaw(type, buff, /*isSplat=*/(values.size() == 1));
}

/// Constructs a dense integer elements attribute from an array of APInt
/// values. Each APInt value is expected to have the same bitwidth as the
/// element type of 'type'.
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<APInt> values) {
  assert(type.getElementType().isa<IntegerType>());
  return getRaw(type, values);
}

// Constructs a dense float elements attribute from an array of APFloat
// values. Each APFloat value is expected to have the same bitwidth as the
// element type of 'type'.
DenseElementsAttr DenseElementsAttr::get(ShapedType type,
                                         ArrayRef<APFloat> values) {
  assert(type.getElementType().isa<FloatType>());

  // Convert the APFloat values to APInt and create a dense elements attribute.
  std::vector<APInt> intValues(values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i)
    intValues[i] = values[i].bitcastToAPInt();
  return getRaw(type, intValues);
}

// Constructs a dense elements attribute from an array of raw APInt values.
// Each APInt value is expected to have the same bitwidth as the element type
// of 'type'.
DenseElementsAttr DenseElementsAttr::getRaw(ShapedType type,
                                            ArrayRef<APInt> values) {
  assert(hasSameElementsOrSplat(type, values));

  size_t bitWidth = getDenseElementBitwidth(type.getElementType());
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);
  std::vector<char> elementData(llvm::divideCeil(storageBitWidth, CHAR_BIT) *
                                values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i) {
    assert(values[i].getBitWidth() == bitWidth);
    writeBits(elementData.data(), i * storageBitWidth, values[i]);
  }
  return getRaw(type, elementData, /*isSplat=*/(values.size() == 1));
}

DenseElementsAttr DenseElementsAttr::getRaw(ShapedType type,
                                            ArrayRef<char> data, bool isSplat) {
  assert((type.isa<RankedTensorType>() || type.isa<VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), StandardAttributes::DenseElements, type,
                   data, isSplat);
}

/// Check the information for a c++ data type, check if this type is valid for
/// the current attribute. This method is used to verify specific type
/// invariants that the templatized 'getValues' method cannot.
static bool isValidIntOrFloat(ShapedType type, int64_t dataEltSize,
                              bool isInt) {
  // Make sure that the data element size is the same as the type element width.
  if ((dataEltSize * CHAR_BIT) != type.getElementTypeBitWidth())
    return false;

  // Check that the element type is valid.
  return isInt ? type.getElementType().isa<IntegerType>()
               : type.getElementType().isa<FloatType>();
}

/// Overload of the 'getRaw' method that asserts that the given type is of
/// integer type. This method is used to verify type invariants that the
/// templatized 'get' method cannot.
DenseElementsAttr DenseElementsAttr::getRawIntOrFloat(ShapedType type,
                                                      ArrayRef<char> data,
                                                      int64_t dataEltSize,
                                                      bool isInt) {
  assert(::isValidIntOrFloat(type, dataEltSize, isInt));

  int64_t numElements = data.size() / dataEltSize;
  assert(numElements == 1 || numElements == type.getNumElements());
  return getRaw(type, data, /*isSplat=*/numElements == 1);
}

/// A method used to verify specific type invariants that the templatized 'get'
/// method cannot.
bool DenseElementsAttr::isValidIntOrFloat(int64_t dataEltSize,
                                          bool isInt) const {
  return ::isValidIntOrFloat(getType(), dataEltSize, isInt);
}

/// Return the raw storage data held by this attribute.
ArrayRef<char> DenseElementsAttr::getRawData() const {
  return static_cast<ImplType *>(impl)->data;
}

/// Returns if this attribute corresponds to a splat, i.e. if all element
/// values are the same.
bool DenseElementsAttr::isSplat() const { return getImpl()->isSplat; }

/// Return the held element values as a range of Attributes.
auto DenseElementsAttr::getAttributeValues() const
    -> llvm::iterator_range<AttributeElementIterator> {
  return {attr_value_begin(), attr_value_end()};
}
auto DenseElementsAttr::attr_value_begin() const -> AttributeElementIterator {
  return AttributeElementIterator(*this, 0);
}
auto DenseElementsAttr::attr_value_end() const -> AttributeElementIterator {
  return AttributeElementIterator(*this, getNumElements());
}

/// Return the held element values as a range of bool. The element type of
/// this attribute must be of integer type of bitwidth 1.
auto DenseElementsAttr::getBoolValues() const
    -> llvm::iterator_range<BoolElementIterator> {
  auto eltType = getType().getElementType().dyn_cast<IntegerType>();
  assert(eltType && eltType.getWidth() == 1 && "expected i1 integer type");
  (void)eltType;
  return {BoolElementIterator(*this, 0),
          BoolElementIterator(*this, getNumElements())};
}

/// Return the held element values as a range of APInts. The element type of
/// this attribute must be of integer type.
auto DenseElementsAttr::getIntValues() const
    -> llvm::iterator_range<IntElementIterator> {
  assert(getType().getElementType().isa<IntegerType>() &&
         "expected integer type");
  return {raw_int_begin(), raw_int_end()};
}
auto DenseElementsAttr::int_value_begin() const -> IntElementIterator {
  assert(getType().getElementType().isa<IntegerType>() &&
         "expected integer type");
  return raw_int_begin();
}
auto DenseElementsAttr::int_value_end() const -> IntElementIterator {
  assert(getType().getElementType().isa<IntegerType>() &&
         "expected integer type");
  return raw_int_end();
}

/// Return the held element values as a range of APFloat. The element type of
/// this attribute must be of float type.
auto DenseElementsAttr::getFloatValues() const
    -> llvm::iterator_range<FloatElementIterator> {
  auto elementType = getType().getElementType().cast<FloatType>();
  assert(elementType.isa<FloatType>() && "expected float type");
  const auto &elementSemantics = elementType.getFloatSemantics();
  return {FloatElementIterator(elementSemantics, raw_int_begin()),
          FloatElementIterator(elementSemantics, raw_int_end())};
}
auto DenseElementsAttr::float_value_begin() const -> FloatElementIterator {
  return getFloatValues().begin();
}
auto DenseElementsAttr::float_value_end() const -> FloatElementIterator {
  return getFloatValues().end();
}

/// Return a new DenseElementsAttr that has the same data as the current
/// attribute, but has been reshaped to 'newType'. The new type must have the
/// same total number of elements as well as element type.
DenseElementsAttr DenseElementsAttr::reshape(ShapedType newType) {
  ShapedType curType = getType();
  if (curType == newType)
    return *this;

  (void)curType;
  assert(newType.getElementType() == curType.getElementType() &&
         "expected the same element type");
  assert(newType.getNumElements() == curType.getNumElements() &&
         "expected the same number of elements");
  return getRaw(newType, getRawData(), isSplat());
}

DenseElementsAttr
DenseElementsAttr::mapValues(Type newElementType,
                             function_ref<APInt(const APInt &)> mapping) const {
  return cast<DenseIntElementsAttr>().mapValues(newElementType, mapping);
}

DenseElementsAttr DenseElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APFloat &)> mapping) const {
  return cast<DenseFPElementsAttr>().mapValues(newElementType, mapping);
}

//===----------------------------------------------------------------------===//
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

template <typename Fn, typename Attr>
static ShapedType mappingHelper(Fn mapping, Attr &attr, ShapedType inType,
                                Type newElementType,
                                llvm::SmallVectorImpl<char> &data) {
  size_t bitWidth = getDenseElementBitwidth(newElementType);
  size_t storageBitWidth = getDenseElementStorageWidth(bitWidth);

  ShapedType newArrayType;
  if (inType.isa<RankedTensorType>())
    newArrayType = RankedTensorType::get(inType.getShape(), newElementType);
  else if (inType.isa<UnrankedTensorType>())
    newArrayType = RankedTensorType::get(inType.getShape(), newElementType);
  else if (inType.isa<VectorType>())
    newArrayType = VectorType::get(inType.getShape(), newElementType);
  else
    assert(newArrayType && "Unhandled tensor type");

  size_t numRawElements = attr.isSplat() ? 1 : newArrayType.getNumElements();
  data.resize(llvm::divideCeil(storageBitWidth, CHAR_BIT) * numRawElements);

  // Functor used to process a single element value of the attribute.
  auto processElt = [&](decltype(*attr.begin()) value, size_t index) {
    auto newInt = mapping(value);
    assert(newInt.getBitWidth() == bitWidth);
    writeBits(data.data(), index * storageBitWidth, newInt);
  };

  // Check for the splat case.
  if (attr.isSplat()) {
    processElt(*attr.begin(), /*index=*/0);
    return newArrayType;
  }

  // Otherwise, process all of the element values.
  uint64_t elementIdx = 0;
  for (auto value : attr)
    processElt(value, elementIdx++);
  return newArrayType;
}

DenseElementsAttr DenseFPElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APFloat &)> mapping) const {
  llvm::SmallVector<char, 8> elementData;
  auto newArrayType =
      mappingHelper(mapping, *this, getType(), newElementType, elementData);

  return getRaw(newArrayType, elementData, isSplat());
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseFPElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseElementsAttr>() &&
         attr.getType().cast<ShapedType>().getElementType().isa<FloatType>();
}

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

DenseElementsAttr DenseIntElementsAttr::mapValues(
    Type newElementType, function_ref<APInt(const APInt &)> mapping) const {
  llvm::SmallVector<char, 8> elementData;
  auto newArrayType =
      mappingHelper(mapping, *this, getType(), newElementType, elementData);

  return getRaw(newArrayType, elementData, isSplat());
}

/// Method for supporting type inquiry through isa, cast and dyn_cast.
bool DenseIntElementsAttr::classof(Attribute attr) {
  return attr.isa<DenseElementsAttr>() &&
         attr.getType().cast<ShapedType>().getElementType().isa<IntegerType>();
}

//===----------------------------------------------------------------------===//
// OpaqueElementsAttr
//===----------------------------------------------------------------------===//

OpaqueElementsAttr OpaqueElementsAttr::get(Dialect *dialect, ShapedType type,
                                           StringRef bytes) {
  assert(TensorType::isValidElementType(type.getElementType()) &&
         "Input element type should be a valid tensor element type");
  return Base::get(type.getContext(), StandardAttributes::OpaqueElements, type,
                   dialect, bytes);
}

StringRef OpaqueElementsAttr::getValue() const { return getImpl()->bytes; }

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute OpaqueElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  if (Dialect *dialect = getDialect())
    return dialect->extractElementHook(*this, index);
  return Attribute();
}

Dialect *OpaqueElementsAttr::getDialect() const { return getImpl()->dialect; }

bool OpaqueElementsAttr::decode(ElementsAttr &result) {
  if (auto *d = getDialect())
    return d->decodeHook(*this, result);
  return true;
}

//===----------------------------------------------------------------------===//
// SparseElementsAttr
//===----------------------------------------------------------------------===//

SparseElementsAttr SparseElementsAttr::get(ShapedType type,
                                           DenseElementsAttr indices,
                                           DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  assert((type.isa<RankedTensorType>() || type.isa<VectorType>()) &&
         "type must be ranked tensor or vector");
  assert(type.hasStaticShape() && "type must have static shape");
  return Base::get(type.getContext(), StandardAttributes::SparseElements, type,
                   indices.cast<DenseIntElementsAttr>(), values);
}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return getImpl()->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return getImpl()->values;
}

/// Return the value of the element at the given index.
Attribute SparseElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  assert(isValidIndex(index) && "expected valid multi-dimensional index");
  auto type = getType();

  // The sparse indices are 64-bit integers, so we can reinterpret the raw data
  // as a 1-D index array.
  auto sparseIndices = getIndices();
  auto sparseIndexValues = sparseIndices.getValues<uint64_t>();

  // Check to see if the indices are a splat.
  if (sparseIndices.isSplat()) {
    // If the index is also not a splat of the index value, we know that the
    // value is zero.
    auto splatIndex = *sparseIndexValues.begin();
    if (llvm::any_of(index, [=](uint64_t i) { return i != splatIndex; }))
      return getZeroAttr();

    // If the indices are a splat, we also expect the values to be a splat.
    assert(getValues().isSplat() && "expected splat values");
    return getValues().getSplatValue();
  }

  // Build a mapping between known indices and the offset of the stored element.
  llvm::SmallDenseMap<llvm::ArrayRef<uint64_t>, size_t> mappedIndices;
  auto numSparseIndices = sparseIndices.getType().getDimSize(0);
  size_t rank = type.getRank();
  for (size_t i = 0, e = numSparseIndices; i != e; ++i)
    mappedIndices.try_emplace(
        {&*std::next(sparseIndexValues.begin(), i * rank), rank}, i);

  // Look for the provided index key within the mapped indices. If the provided
  // index is not found, then return a zero attribute.
  auto it = mappedIndices.find(index);
  if (it == mappedIndices.end())
    return getZeroAttr();

  // Otherwise, return the held sparse value element.
  return getValues().getValue(it->second);
}

/// Get a zero APFloat for the given sparse attribute.
APFloat SparseElementsAttr::getZeroAPFloat() const {
  auto eltType = getType().getElementType().cast<FloatType>();
  return APFloat(eltType.getFloatSemantics());
}

/// Get a zero APInt for the given sparse attribute.
APInt SparseElementsAttr::getZeroAPInt() const {
  auto eltType = getType().getElementType().cast<IntegerType>();
  return APInt::getNullValue(eltType.getWidth());
}

/// Get a zero attribute for the given attribute type.
Attribute SparseElementsAttr::getZeroAttr() const {
  auto eltType = getType().getElementType();

  // Handle floating point elements.
  if (eltType.isa<FloatType>())
    return FloatAttr::get(eltType, 0);

  // Otherwise, this is an integer.
  auto intEltTy = eltType.cast<IntegerType>();
  if (intEltTy.getWidth() == 1)
    return BoolAttr::get(false, eltType.getContext());
  return IntegerAttr::get(eltType, 0);
}

/// Flatten, and return, all of the sparse indices in this attribute in
/// row-major order.
std::vector<ptrdiff_t> SparseElementsAttr::getFlattenedSparseIndices() const {
  std::vector<ptrdiff_t> flatSparseIndices;

  // The sparse indices are 64-bit integers, so we can reinterpret the raw data
  // as a 1-D index array.
  auto sparseIndices = getIndices();
  auto sparseIndexValues = sparseIndices.getValues<uint64_t>();
  if (sparseIndices.isSplat()) {
    SmallVector<uint64_t, 8> indices(getType().getRank(),
                                     *sparseIndexValues.begin());
    flatSparseIndices.push_back(getFlattenedIndex(indices));
    return flatSparseIndices;
  }

  // Otherwise, reinterpret each index as an ArrayRef when flattening.
  auto numSparseIndices = sparseIndices.getType().getDimSize(0);
  size_t rank = getType().getRank();
  for (size_t i = 0, e = numSparseIndices; i != e; ++i)
    flatSparseIndices.push_back(getFlattenedIndex(
        {&*std::next(sparseIndexValues.begin(), i * rank), rank}));
  return flatSparseIndices;
}

//===----------------------------------------------------------------------===//
// NamedAttributeList
//===----------------------------------------------------------------------===//

NamedAttributeList::NamedAttributeList(ArrayRef<NamedAttribute> attributes) {
  setAttrs(attributes);
}

ArrayRef<NamedAttribute> NamedAttributeList::getAttrs() const {
  return attrs ? attrs.getValue() : llvm::None;
}

/// Replace the held attributes with ones provided in 'newAttrs'.
void NamedAttributeList::setAttrs(ArrayRef<NamedAttribute> attributes) {
  // Don't create an attribute list if there are no attributes.
  if (attributes.empty())
    attrs = nullptr;
  else
    attrs = DictionaryAttr::get(attributes, attributes[0].second.getContext());
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttributeList::get(StringRef name) const {
  return attrs ? attrs.get(name) : nullptr;
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttributeList::get(Identifier name) const {
  return attrs ? attrs.get(name) : nullptr;
}

/// If the an attribute exists with the specified name, change it to the new
/// value.  Otherwise, add a new attribute with the specified name/value.
void NamedAttributeList::set(Identifier name, Attribute value) {
  assert(value && "attributes may never be null");

  // If we already have this attribute, replace it.
  auto origAttrs = getAttrs();
  SmallVector<NamedAttribute, 8> newAttrs(origAttrs.begin(), origAttrs.end());
  for (auto &elt : newAttrs)
    if (elt.first == name) {
      elt.second = value;
      attrs = DictionaryAttr::get(newAttrs, value.getContext());
      return;
    }

  // Otherwise, add it.
  newAttrs.push_back({name, value});
  attrs = DictionaryAttr::get(newAttrs, value.getContext());
}

/// Remove the attribute with the specified name if it exists.  The return
/// value indicates whether the attribute was present or not.
auto NamedAttributeList::remove(Identifier name) -> RemoveResult {
  auto origAttrs = getAttrs();
  for (unsigned i = 0, e = origAttrs.size(); i != e; ++i) {
    if (origAttrs[i].first == name) {
      // Handle the simple case of removing the only attribute in the list.
      if (e == 1) {
        attrs = nullptr;
        return RemoveResult::Removed;
      }

      SmallVector<NamedAttribute, 8> newAttrs;
      newAttrs.reserve(origAttrs.size() - 1);
      newAttrs.append(origAttrs.begin(), origAttrs.begin() + i);
      newAttrs.append(origAttrs.begin() + i + 1, origAttrs.end());
      attrs = DictionaryAttr::get(newAttrs, newAttrs[0].second.getContext());
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}
