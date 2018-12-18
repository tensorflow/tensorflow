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
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::detail;

Attribute::Kind Attribute::getKind() const { return attr->kind; }

bool Attribute::isOrContainsFunction() const {
  return attr->isOrContainsFunctionCache;
}

// Given an attribute that could refer to a function attribute in the remapping
// table, walk it and rewrite it to use the mapped function.  If it doesn't
// refer to anything in the table, then it is returned unmodified.
Attribute Attribute::remapFunctionAttrs(
    const llvm::DenseMap<Attribute, FunctionAttr> &remappingTable,
    MLIRContext *context) const {
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
    auto newElt = elt.remapFunctionAttrs(remappingTable, context);
    remappedElts.push_back(newElt);
    anyChange |= (elt != newElt);
  }

  if (!anyChange)
    return *this;

  return ArrayAttr::get(remappedElts, context);
}

BoolAttr::BoolAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

bool BoolAttr::getValue() const { return static_cast<ImplType *>(attr)->value; }

IntegerAttr::IntegerAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

APInt IntegerAttr::getValue() const {
  return static_cast<ImplType *>(attr)->getValue();
}

int64_t IntegerAttr::getInt() const { return getValue().getSExtValue(); }

Type IntegerAttr::getType() const {
  return static_cast<ImplType *>(attr)->type;
}

FloatAttr::FloatAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

APFloat FloatAttr::getValue() const {
  return static_cast<ImplType *>(attr)->getValue();
}

Type FloatAttr::getType() const { return static_cast<ImplType *>(attr)->type; }

double FloatAttr::getDouble() const { return getValue().convertToDouble(); }

StringAttr::StringAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

StringRef StringAttr::getValue() const {
  return static_cast<ImplType *>(attr)->value;
}

ArrayAttr::ArrayAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

ArrayRef<Attribute> ArrayAttr::getValue() const {
  return static_cast<ImplType *>(attr)->value;
}

AffineMapAttr::AffineMapAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

AffineMap AffineMapAttr::getValue() const {
  return static_cast<ImplType *>(attr)->value;
}

IntegerSetAttr::IntegerSetAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

IntegerSet IntegerSetAttr::getValue() const {
  return static_cast<ImplType *>(attr)->value;
}

TypeAttr::TypeAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

Type TypeAttr::getValue() const { return static_cast<ImplType *>(attr)->value; }

FunctionAttr::FunctionAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

Function *FunctionAttr::getValue() const {
  return static_cast<ImplType *>(attr)->value;
}

FunctionType FunctionAttr::getType() const { return getValue()->getType(); }

ElementsAttr::ElementsAttr(Attribute::ImplType *ptr) : Attribute(ptr) {}

VectorOrTensorType ElementsAttr::getType() const {
  return static_cast<ImplType *>(attr)->type;
}

SplatElementsAttr::SplatElementsAttr(Attribute::ImplType *ptr)
    : ElementsAttr(ptr) {}

Attribute SplatElementsAttr::getValue() const {
  return static_cast<ImplType *>(attr)->elt;
}

DenseElementsAttr::DenseElementsAttr(Attribute::ImplType *ptr)
    : ElementsAttr(ptr) {}

void DenseElementsAttr::getValues(SmallVectorImpl<Attribute> &values) const {
  switch (getKind()) {
  case Attribute::Kind::DenseIntElements:
    cast<DenseIntElementsAttr>().getValues(values);
    return;
  case Attribute::Kind::DenseFPElements:
    cast<DenseFPElementsAttr>().getValues(values);
    return;
  default:
    llvm_unreachable("unexpected element type");
  }
}

ArrayRef<char> DenseElementsAttr::getRawData() const {
  return static_cast<ImplType *>(attr)->data;
}

/// Writes the lowest `bitWidth` bits of `value` to bit position `bitPos`
/// starting from `rawData`.
void DenseElementsAttr::writeBits(char *data, size_t bitPos, size_t bitWidth,
                                  uint64_t value) {
  // Read the destination bytes which will be written to.
  uint64_t dst = 0;
  auto dstData = reinterpret_cast<char *>(&dst);
  auto endPos = bitPos + bitWidth;
  auto start = data + bitPos / 8;
  auto end = data + endPos / 8 + (endPos % 8 != 0);
  std::copy(start, end, dstData);

  // Clean up the invalid bits in the destination bytes.
  dst &= ~(-1UL << (bitPos % 8));

  // Get the valid bits of the source value, shift them to right position,
  // then add them to the destination bytes.
  value <<= bitPos % 8;
  dst |= value;

  // Write the destination bytes back.
  ArrayRef<char> range({dstData, (size_t)(end - start)});
  std::copy(range.begin(), range.end(), start);
}

/// Reads the next `bitWidth` bits from the bit position `bitPos` of `rawData`
/// and put them in the lowest bits.
uint64_t DenseElementsAttr::readBits(const char *rawData, size_t bitPos,
                                     size_t bitsWidth) {
  uint64_t dst = 0;
  auto dstData = reinterpret_cast<char *>(&dst);
  auto endPos = bitPos + bitsWidth;
  auto start = rawData + bitPos / 8;
  auto end = rawData + endPos / 8 + (endPos % 8 != 0);
  std::copy(start, end, dstData);

  dst >>= bitPos % 8;
  dst &= ~(-1UL << bitsWidth);
  return dst;
}

DenseIntElementsAttr::DenseIntElementsAttr(Attribute::ImplType *ptr)
    : DenseElementsAttr(ptr) {}

void DenseIntElementsAttr::getValues(SmallVectorImpl<Attribute> &values) const {
  auto bitsWidth = static_cast<ImplType *>(attr)->bitsWidth;
  auto elementNum = getType().getNumElements();
  values.reserve(elementNum);
  if (bitsWidth == 64) {
    ArrayRef<int64_t> vs(
        {reinterpret_cast<const int64_t *>(getRawData().data()),
         getRawData().size() / 8});
    for (auto value : vs) {
      auto attr = IntegerAttr::get(getType().getElementType(), value);
      values.push_back(attr);
    }
  } else {
    const auto *rawData = getRawData().data();
    for (size_t pos = 0; pos < elementNum * bitsWidth; pos += bitsWidth) {
      uint64_t bits = readBits(rawData, pos, bitsWidth);
      APInt value(bitsWidth, bits, /*isSigned=*/true);
      auto attr =
          IntegerAttr::get(getType().getElementType(), value.getSExtValue());
      values.push_back(attr);
    }
  }
}

DenseFPElementsAttr::DenseFPElementsAttr(Attribute::ImplType *ptr)
    : DenseElementsAttr(ptr) {}

// Construct a FloatAttr wrapping a float value of `elementType` type from its
// bit representation.  The APFloat stored in the attribute will have the
// semantics defined by the float semantics of the element type.
static inline FloatAttr makeFloatAttrFromBits(size_t bitWidth, uint64_t bits,
                                              FloatType elementType) {
  auto apint = APInt(bitWidth, bits);
  auto apfloat = APFloat(elementType.getFloatSemantics(), apint);
  return FloatAttr::get(elementType, apfloat);
}

void DenseFPElementsAttr::getValues(SmallVectorImpl<Attribute> &values) const {
  auto elementNum = getType().getNumElements();
  auto elementType = getType().getElementType().dyn_cast<FloatType>();
  assert(elementType && "non-float type in FP attribute");
  // FIXME: using 64 bits for BF16 because it is currently stored with double
  // semantics.
  size_t bitWidth =
      elementType.isBF16() ? 64 : elementType.getIntOrFloatBitWidth();

  values.reserve(elementNum);
  if (bitWidth == 64) {
    ArrayRef<int64_t> vs(
        {reinterpret_cast<const int64_t *>(getRawData().data()),
         getRawData().size() / 8});
    for (auto bitValue : vs) {
      values.push_back(makeFloatAttrFromBits(64, bitValue, elementType));
    }
    return;
  }
  for (unsigned i = 0; i < elementNum; ++i) {
    uint64_t bits = readBits(getRawData().data(), i * bitWidth, bitWidth);
    values.push_back(makeFloatAttrFromBits(bitWidth, bits, elementType));
  }
}

OpaqueElementsAttr::OpaqueElementsAttr(Attribute::ImplType *ptr)
    : ElementsAttr(ptr) {}

StringRef OpaqueElementsAttr::getValue() const {
  return static_cast<ImplType *>(attr)->bytes;
}

SparseElementsAttr::SparseElementsAttr(Attribute::ImplType *ptr)
    : ElementsAttr(ptr) {}

DenseIntElementsAttr SparseElementsAttr::getIndices() const {
  return static_cast<ImplType *>(attr)->indices;
}

DenseElementsAttr SparseElementsAttr::getValues() const {
  return static_cast<ImplType *>(attr)->values;
}
