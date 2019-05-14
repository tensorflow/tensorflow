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
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// AttributeStorage
//===----------------------------------------------------------------------===//

AttributeStorage::AttributeStorage(Type type, bool isOrContainsFunctionCache)
    : typeAndContainsFunctionAttrPair(type.getAsOpaquePointer(),
                                      isOrContainsFunctionCache) {}
AttributeStorage::AttributeStorage(bool isOrContainsFunctionCache)
    : AttributeStorage(/*type=*/nullptr, isOrContainsFunctionCache) {}
AttributeStorage::AttributeStorage()
    : AttributeStorage(/*type=*/nullptr, /*isOrContainsFunctionCache=*/false) {}

Type AttributeStorage::getType() const {
  return Type::getFromOpaquePointer(
      typeAndContainsFunctionAttrPair.getPointer());
}
void AttributeStorage::setType(Type type) {
  typeAndContainsFunctionAttrPair.setPointer(type.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the type of this attribute.
Type Attribute::getType() const { return impl->getType(); }

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getType().getContext(); }

/// Get the dialect this attribute is registered to.
const Dialect &Attribute::getDialect() const { return impl->getDialect(); }

bool Attribute::isOrContainsFunction() const {
  return impl->isOrContainsFunctionCache();
}

// Given an attribute that could refer to a function attribute in the remapping
// table, walk it and rewrite it to use the mapped function.  If it doesn't
// refer to anything in the table, then it is returned unmodified.
Attribute Attribute::remapFunctionAttrs(
    const llvm::DenseMap<Attribute, FunctionAttr> &remappingTable) const {
  // Most attributes are trivially unrelated to function attributes, skip them
  // rapidly.
  if (!isOrContainsFunction())
    return *this;

  // If we have a function attribute, remap it.
  if (auto fnAttr = this->dyn_cast<FunctionAttr>()) {
    auto it = remappingTable.find(fnAttr);
    return it != remappingTable.end() ? it->second : *this;
  }

  // Otherwise, we must have an array attribute, remap the elements.
  auto arrayAttr = this->cast<ArrayAttr>();
  SmallVector<Attribute, 8> remappedElts;
  bool anyChange = false;
  for (auto elt : arrayAttr.getValue()) {
    auto newElt = elt.remapFunctionAttrs(remappingTable);
    remappedElts.push_back(newElt);
    anyChange |= (elt != newElt);
  }

  if (!anyChange)
    return *this;

  return ArrayAttr::get(remappedElts, getContext());
}

//===----------------------------------------------------------------------===//
// BoolAttr
//===----------------------------------------------------------------------===//

BoolAttr BoolAttr::get(bool value, MLIRContext *context) {
  // Note: The context is also used within the BoolAttrStorage.
  return Base::get(context, StandardAttributes::Bool, context, value);
}

bool BoolAttr::getValue() const { return getImpl()->value; }

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
static LogicalResult verifyFloatTypeInvariants(llvm::Optional<Location> loc,
                                               Type type) {
  if (!type.isa<FloatType>()) {
    if (loc)
      type.getContext()->emitError(*loc, "expected floating point type");
    return failure();
  }
  return success();
}

LogicalResult FloatAttr::verifyConstructionInvariants(
    llvm::Optional<Location> loc, MLIRContext *ctx, Type type, double value) {
  return verifyFloatTypeInvariants(loc, type);
}

LogicalResult
FloatAttr::verifyConstructionInvariants(llvm::Optional<Location> loc,
                                        MLIRContext *ctx, Type type,
                                        const APFloat &value) {
  // Verify that the type is correct.
  if (failed(verifyFloatTypeInvariants(loc, type)))
    return failure();

  // Verify that the type semantics match that of the value.
  if (&type.cast<FloatType>().getFloatSemantics() != &value.getSemantics()) {
    if (loc)
      ctx->emitError(
          *loc, "FloatAttr type doesn't match the type implied by its value");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StringAttr
//===----------------------------------------------------------------------===//

StringAttr StringAttr::get(StringRef bytes, MLIRContext *context) {
  return Base::get(context, StandardAttributes::String, bytes);
}

StringRef StringAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

ArrayAttr ArrayAttr::get(ArrayRef<Attribute> value, MLIRContext *context) {
  return Base::get(context, StandardAttributes::Array, value);
}

ArrayRef<Attribute> ArrayAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// AffineMapAttr
//===----------------------------------------------------------------------===//

AffineMapAttr AffineMapAttr::get(AffineMap value) {
  return Base::get(value.getResult(0).getContext(),
                   StandardAttributes::AffineMap, value);
}

AffineMap AffineMapAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// IntegerSetAttr
//===----------------------------------------------------------------------===//

IntegerSetAttr IntegerSetAttr::get(IntegerSet value) {
  return Base::get(value.getConstraint(0).getContext(),
                   StandardAttributes::IntegerSet, value);
}

IntegerSet IntegerSetAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

TypeAttr TypeAttr::get(Type value) {
  return Base::get(value.getContext(), StandardAttributes::Type, value);
}

Type TypeAttr::getValue() const { return getImpl()->value; }

//===----------------------------------------------------------------------===//
// FunctionAttr
//===----------------------------------------------------------------------===//

FunctionAttr FunctionAttr::get(Function *value) {
  assert(value && "Cannot get FunctionAttr for a null function");
  return Base::get(value->getContext(), StandardAttributes::Function, value);
}

/// This function is used by the internals of the Function class to null out
/// attributes referring to functions that are about to be deleted.
void FunctionAttr::dropFunctionReference(Function *value) {
  AttributeUniquer::erase<FunctionAttr>(value->getContext(),
                                        StandardAttributes::Function, value);
}

Function *FunctionAttr::getValue() const { return getImpl()->value; }

FunctionType FunctionAttr::getType() const {
  return Attribute::getType().cast<FunctionType>();
}

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

VectorOrTensorType ElementsAttr::getType() const {
  return Attribute::getType().cast<VectorOrTensorType>();
}

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute ElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  switch (getKind()) {
  case StandardAttributes::SplatElements:
    return cast<SplatElementsAttr>().getValue();
  case StandardAttributes::DenseFPElements:
  case StandardAttributes::DenseIntElements:
    return cast<DenseElementsAttr>().getValue(index);
  case StandardAttributes::OpaqueElements:
    return cast<OpaqueElementsAttr>().getValue(index);
  case StandardAttributes::SparseElements:
    return cast<SparseElementsAttr>().getValue(index);
  default:
    llvm_unreachable("unknown ElementsAttr kind");
  }
}

//===----------------------------------------------------------------------===//
// SplatElementsAttr
//===----------------------------------------------------------------------===//

SplatElementsAttr SplatElementsAttr::get(VectorOrTensorType type,
                                         Attribute elt) {
  assert(elt.getType() == type.getElementType() &&
         "value should be of the given element type");
  return Base::get(type.getContext(), StandardAttributes::SplatElements, type,
                   elt);
}

Attribute SplatElementsAttr::getValue() const { return getImpl()->elt; }

//===----------------------------------------------------------------------===//
// RawElementIterator
//===----------------------------------------------------------------------===//

static size_t getDenseElementBitwidth(Type eltType) {
  // FIXME(b/121118307): using 64 bits for BF16 because it is currently stored
  // with double semantics.
  return eltType.isBF16() ? 64 : eltType.getIntOrFloatBitWidth();
}

/// Constructs a new iterator.
DenseElementsAttr::RawElementIterator::RawElementIterator(
    DenseElementsAttr attr, size_t index)
    : rawData(attr.getRawData().data()), index(index),
      bitWidth(getDenseElementBitwidth(attr.getType().getElementType())) {}

/// Accesses the raw APInt value at this iterator position.
APInt DenseElementsAttr::RawElementIterator::operator*() const {
  return readBits(rawData, index * bitWidth, bitWidth);
}

//===----------------------------------------------------------------------===//
// DenseElementsAttr
//===----------------------------------------------------------------------===//

DenseElementsAttr DenseElementsAttr::get(VectorOrTensorType type,
                                         ArrayRef<char> data) {
  assert((static_cast<uint64_t>(type.getSizeInBits()) <=
          data.size() * APInt::APINT_WORD_SIZE) &&
         "Input data bit size should be larger than that type requires");
  switch (type.getElementType().getKind()) {
  case StandardTypes::BF16:
  case StandardTypes::F16:
  case StandardTypes::F32:
  case StandardTypes::F64:
    return AttributeUniquer::get<DenseFPElementsAttr>(
        type.getContext(), StandardAttributes::DenseFPElements, type, data);
  case StandardTypes::Integer:
    return AttributeUniquer::get<DenseIntElementsAttr>(
        type.getContext(), StandardAttributes::DenseIntElements, type, data);
  default:
    llvm_unreachable("unexpected element type");
  }
}

DenseElementsAttr DenseElementsAttr::get(VectorOrTensorType type,
                                         ArrayRef<Attribute> values) {
  assert(type.getElementType().isIntOrFloat() &&
         "expected int or float element type");
  assert(values.size() == type.getNumElements() &&
         "expected 'values' to contain the same number of elements as 'type'");

  // FIXME(b/121118307): using 64 bits for BF16 because it is currently stored
  // with double semantics.
  auto eltType = type.getElementType();
  size_t bitWidth = eltType.isBF16() ? 64 : eltType.getIntOrFloatBitWidth();

  // Compress the attribute values into a character buffer.
  SmallVector<char, 8> data(APInt::getNumWords(bitWidth * values.size()) *
                            APInt::APINT_WORD_SIZE);
  APInt intVal;
  for (unsigned i = 0, e = values.size(); i < e; ++i) {
    switch (eltType.getKind()) {
    case StandardTypes::BF16:
    case StandardTypes::F16:
    case StandardTypes::F32:
    case StandardTypes::F64:
      assert(eltType == values[i].cast<FloatAttr>().getType() &&
             "expected attribute value to have element type");
      intVal = values[i].cast<FloatAttr>().getValue().bitcastToAPInt();
      break;
    case StandardTypes::Integer:
      assert(eltType == values[i].cast<IntegerAttr>().getType() &&
             "expected attribute value to have element type");
      intVal = values[i].cast<IntegerAttr>().getValue();
      break;
    default:
      llvm_unreachable("unexpected element type");
    }
    assert(intVal.getBitWidth() == bitWidth &&
           "expected value to have same bitwidth as element type");
    writeBits(data.data(), i * bitWidth, intVal);
  }
  return get(type, data);
}

/// Returns the number of elements held by this attribute.
size_t DenseElementsAttr::size() const { return getType().getNumElements(); }

/// Return the value at the given index. If index does not refer to a valid
/// element, then a null attribute is returned.
Attribute DenseElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  auto type = getType();

  // Verify that the rank of the indices matches the held type.
  auto rank = type.getRank();
  if (static_cast<size_t>(rank) != index.size())
    return Attribute();

  // Verify that all of the indices are within the shape dimensions.
  auto shape = type.getShape();
  for (unsigned i = 0; i != rank; ++i)
    if (shape[i] <= static_cast<int64_t>(index[i]))
      return Attribute();

  // Reduce the provided multidimensional index into a 1D index.
  uint64_t valueIndex = 0;
  uint64_t dimMultiplier = 1;
  for (int i = rank - 1; i >= 0; --i) {
    valueIndex += index[i] * dimMultiplier;
    dimMultiplier *= shape[i];
  }

  // Return the element stored at the 1D index.
  auto elementType = getType().getElementType();
  size_t bitWidth = getDenseElementBitwidth(elementType);
  APInt rawValueData =
      readBits(getRawData().data(), valueIndex * bitWidth, bitWidth);

  // Convert the raw value data to an attribute value.
  switch (getKind()) {
  case StandardAttributes::DenseIntElements:
    return IntegerAttr::get(elementType, rawValueData);
  case StandardAttributes::DenseFPElements:
    return FloatAttr::get(
        elementType, APFloat(elementType.cast<FloatType>().getFloatSemantics(),
                             rawValueData));
  default:
    llvm_unreachable("unexpected element type");
  }
}

void DenseElementsAttr::getValues(SmallVectorImpl<Attribute> &values) const {
  auto elementType = getType().getElementType();
  switch (getKind()) {
  case StandardAttributes::DenseIntElements: {
    // Get the raw APInt values.
    SmallVector<APInt, 8> intValues;
    cast<DenseIntElementsAttr>().getValues(intValues);

    // Convert each to an IntegerAttr.
    for (auto &intVal : intValues)
      values.push_back(IntegerAttr::get(elementType, intVal));
    return;
  }
  case StandardAttributes::DenseFPElements: {
    // Get the raw APFloat values.
    SmallVector<APFloat, 8> floatValues;
    cast<DenseFPElementsAttr>().getValues(floatValues);

    // Convert each to an FloatAttr.
    for (auto &floatVal : floatValues)
      values.push_back(FloatAttr::get(elementType, floatVal));
    return;
  }
  default:
    llvm_unreachable("unexpected element type");
  }
}

ArrayRef<char> DenseElementsAttr::getRawData() const {
  return static_cast<ImplType *>(impl)->data;
}

// Constructs a dense elements attribute from an array of raw APInt values.
// Each APInt value is expected to have the same bitwidth as the element type
// of 'type'.
DenseElementsAttr DenseElementsAttr::get(VectorOrTensorType type,
                                         ArrayRef<APInt> values) {
  assert(values.size() == type.getNumElements() &&
         "expected 'values' to contain the same number of elements as 'type'");

  size_t bitWidth = getDenseElementBitwidth(type.getElementType());
  std::vector<char> elementData(APInt::getNumWords(bitWidth * values.size()) *
                                APInt::APINT_WORD_SIZE);
  for (unsigned i = 0, e = values.size(); i != e; ++i) {
    assert(values[i].getBitWidth() == bitWidth);
    writeBits(elementData.data(), i * bitWidth, values[i]);
  }
  return get(type, elementData);
}

/// Writes value to the bit position `bitPos` in array `rawData`. 'rawData' is
/// expected to be a 64-bit aligned storage address.
void DenseElementsAttr::writeBits(char *rawData, size_t bitPos, APInt value) {
  size_t bitWidth = value.getBitWidth();

  // If the bitwidth is 1 we just toggle the specific bit.
  if (bitWidth == 1) {
    auto *rawIntData = reinterpret_cast<uint64_t *>(rawData);
    if (value.isOneValue())
      APInt::tcSetBit(rawIntData, bitPos);
    else
      APInt::tcClearBit(rawIntData, bitPos);
    return;
  }

  // If the bit position and width are byte aligned, write the storage directly
  // to the data.
  if ((bitWidth % 8) == 0 && (bitPos % 8) == 0) {
    std::copy_n(reinterpret_cast<const char *>(value.getRawData()),
                bitWidth / 8, rawData + (bitPos / 8));
    return;
  }

  // Otherwise, convert the raw data into an APInt and insert the value at the
  // specified bit position.
  size_t totalWords = APInt::getNumWords((bitPos % 64) + bitWidth);
  llvm::MutableArrayRef<uint64_t> rawIntData(
      reinterpret_cast<uint64_t *>(rawData) + (bitPos / 64), totalWords);
  APInt tempStorage(totalWords * 64, rawIntData);
  tempStorage.insertBits(value, bitPos % 64);

  // Copy the value back to the raw data.
  std::copy_n(tempStorage.getRawData(), rawIntData.size(), rawIntData.data());
}

/// Reads the next `bitWidth` bits from the bit position `bitPos` in array
/// `rawData`. 'rawData' is expected to be a 64-bit aligned storage address.
APInt DenseElementsAttr::readBits(const char *rawData, size_t bitPos,
                                  size_t bitWidth) {
  // Reinterpret the raw data as a uint64_t word array and extract the value
  // starting at 'bitPos'.
  APInt result(bitWidth, 0);
  const uint64_t *intData = reinterpret_cast<const uint64_t *>(rawData);
  APInt::tcExtract(const_cast<uint64_t *>(result.getRawData()),
                   result.getNumWords(), intData, bitWidth, bitPos);
  return result;
}

//===----------------------------------------------------------------------===//
// DenseIntElementsAttr
//===----------------------------------------------------------------------===//

/// Constructs a dense integer elements attribute from an array of APInt
/// values. Each APInt value is expected to have the same bitwidth as the
/// element type of 'type'.
DenseIntElementsAttr DenseIntElementsAttr::get(VectorOrTensorType type,
                                               ArrayRef<APInt> values) {
  return DenseElementsAttr::get(type, values).cast<DenseIntElementsAttr>();
}

/// Constructs a dense integer elements attribute from an array of integer
/// values. Each value is expected to be within the bitwidth of the element
/// type of 'type'.
DenseIntElementsAttr DenseIntElementsAttr::get(VectorOrTensorType type,
                                               ArrayRef<int64_t> values) {
  auto eltType = type.getElementType();
  size_t bitWidth = eltType.isBF16() ? 64 : eltType.getIntOrFloatBitWidth();

  // Convert the raw integer values to APInt.
  SmallVector<APInt, 8> apIntValues;
  apIntValues.reserve(values.size());
  for (auto value : values)
    apIntValues.emplace_back(APInt(bitWidth, value));
  return get(type, apIntValues);
}

void DenseIntElementsAttr::getValues(SmallVectorImpl<APInt> &values) const {
  values.reserve(size());
  values.assign(raw_begin(), raw_end());
}

//===----------------------------------------------------------------------===//
// DenseFPElementsAttr
//===----------------------------------------------------------------------===//

DenseFPElementsAttr::ElementIterator::ElementIterator(
    const llvm::fltSemantics &smt, RawElementIterator it)
    : llvm::mapped_iterator<RawElementIterator,
                            std::function<APFloat(const APInt &)>>(
          it, [&](const APInt &val) { return APFloat(smt, val); }) {}

// Constructs a dense float elements attribute from an array of APFloat
// values. Each APFloat value is expected to have the same bitwidth as the
// element type of 'type'.
DenseFPElementsAttr DenseFPElementsAttr::get(VectorOrTensorType type,
                                             ArrayRef<APFloat> values) {
  // Convert the APFloat values to APInt and create a dense elements attribute.
  std::vector<APInt> intValues(values.size());
  for (unsigned i = 0, e = values.size(); i != e; ++i)
    intValues[i] = values[i].bitcastToAPInt();
  return DenseElementsAttr::get(type, intValues).cast<DenseFPElementsAttr>();
}

void DenseFPElementsAttr::getValues(SmallVectorImpl<APFloat> &values) const {
  values.reserve(size());
  values.assign(begin(), end());
}

/// Iterator access to the float element values.
DenseFPElementsAttr::iterator DenseFPElementsAttr::begin() const {
  auto elementType = getType().getElementType().cast<FloatType>();
  const auto &elementSemantics = elementType.getFloatSemantics();
  return {elementSemantics, raw_begin()};
}
DenseFPElementsAttr::iterator DenseFPElementsAttr::end() const {
  auto elementType = getType().getElementType().cast<FloatType>();
  const auto &elementSemantics = elementType.getFloatSemantics();
  return {elementSemantics, raw_end()};
}

//===----------------------------------------------------------------------===//
// OpaqueElementsAttr
//===----------------------------------------------------------------------===//

OpaqueElementsAttr OpaqueElementsAttr::get(Dialect *dialect,
                                           VectorOrTensorType type,
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

SparseElementsAttr SparseElementsAttr::get(VectorOrTensorType type,
                                           DenseIntElementsAttr indices,
                                           DenseElementsAttr values) {
  assert(indices.getType().getElementType().isInteger(64) &&
         "expected sparse indices to be 64-bit integer values");
  return Base::get(type.getContext(), StandardAttributes::SparseElements, type,
                   indices, values);
}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return getImpl()->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return getImpl()->values;
}

/// Return the value of the element at the given index.
Attribute SparseElementsAttr::getValue(ArrayRef<uint64_t> index) const {
  auto type = getType();

  // Verify that the rank of the indices matches the held type.
  size_t rank = type.getRank();
  if (rank != index.size())
    return Attribute();

  // The sparse indices are 64-bit integers, so we can reinterpret the raw data
  // as a 1-D index array.
  auto sparseIndices = getIndices();
  const uint64_t *sparseIndexValues =
      reinterpret_cast<const uint64_t *>(sparseIndices.getRawData().data());

  // Build a mapping between known indices and the offset of the stored element.
  llvm::SmallDenseMap<llvm::ArrayRef<uint64_t>, size_t> mappedIndices;
  auto numSparseIndices = sparseIndices.getType().getDimSize(0);
  for (size_t i = 0, e = numSparseIndices; i != e; ++i)
    mappedIndices.try_emplace({sparseIndexValues + (i * rank), rank}, i);

  // Look for the provided index key within the mapped indices. If the provided
  // index is not found, then return a zero attribute.
  auto it = mappedIndices.find(index);
  if (it == mappedIndices.end()) {
    auto eltType = type.getElementType();
    if (eltType.isa<FloatType>())
      return FloatAttr::get(eltType, 0);
    assert(eltType.isa<IntegerType>() && "unexpected element type");
    return IntegerAttr::get(eltType, 0);
  }

  // Otherwise, return the held sparse value element.
  return getValues().getValue(it->second);
}

//===----------------------------------------------------------------------===//
// NamedAttributeList
//===----------------------------------------------------------------------===//

NamedAttributeList::NamedAttributeList(ArrayRef<NamedAttribute> attributes) {
  setAttrs(attributes);
}

/// Return all of the attributes on this operation.
ArrayRef<NamedAttribute> NamedAttributeList::getAttrs() const {
  return attrs ? attrs->getElements() : llvm::None;
}

/// Replace the held attributes with ones provided in 'newAttrs'.
void NamedAttributeList::setAttrs(ArrayRef<NamedAttribute> attributes) {
  // Don't create an attribute list if there are no attributes.
  if (attributes.empty()) {
    attrs = nullptr;
    return;
  }

  assert(llvm::all_of(attributes,
                      [](const NamedAttribute &attr) { return attr.second; }) &&
         "attributes cannot have null entries");
  attrs = AttributeListStorage::get(attributes);
}

/// Return the specified attribute if present, null otherwise.
Attribute NamedAttributeList::get(StringRef name) const {
  for (auto elt : getAttrs())
    if (elt.first.is(name))
      return elt.second;
  return nullptr;
}
Attribute NamedAttributeList::get(Identifier name) const {
  for (auto elt : getAttrs())
    if (elt.first == name)
      return elt.second;
  return nullptr;
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
      attrs = AttributeListStorage::get(newAttrs);
      return;
    }

  // Otherwise, add it.
  newAttrs.push_back({name, value});
  attrs = AttributeListStorage::get(newAttrs);
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
      attrs = AttributeListStorage::get(newAttrs);
      return RemoveResult::Removed;
    }
  }
  return RemoveResult::NotFound;
}
