//===- Types.cpp - MLIR Type Classes --------------------------------------===//
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

#include "mlir/IR/Types.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/STLExtras.h"
using namespace mlir;

IntegerType::IntegerType(unsigned width, MLIRContext *context)
  : Type(Kind::Integer, context), width(width) {
    assert(width <= kMaxWidth && "admissible integer bitwidth exceeded");
}

FloatType::FloatType(Kind kind, MLIRContext *context) : Type(kind, context) {}

OtherType::OtherType(Kind kind, MLIRContext *context) : Type(kind, context) {}

FunctionType::FunctionType(Type *const *inputsAndResults, unsigned numInputs,
                           unsigned numResults, MLIRContext *context)
  : Type(Kind::Function, context, numInputs),
    numResults(numResults), inputsAndResults(inputsAndResults) {
}

VectorOrTensorType::VectorOrTensorType(Kind kind, MLIRContext *context,
                                       Type *elementType, unsigned subClassData)
    : Type(kind, context, subClassData), elementType(elementType) {}

/// If this is ranked tensor or vector type, return the rank. If it is an
/// unranked tensor, return -1.
int VectorOrTensorType::getRank() const {
  switch (getKind()) {
  default:
    llvm_unreachable("not a VectorOrTensorType");
  case Kind::Vector:
  case Kind::RankedTensor:
    return getShape().size();
  case Kind::UnrankedTensor:
    return -1;
  }
}

int VectorOrTensorType::getDimSize(unsigned i) const {
  switch (getKind()) {
  case Kind::Vector:
  case Kind::RankedTensor:
    return getShape()[i];
  default:
    llvm_unreachable("not a VectorOrTensorType");
  }
}

ArrayRef<int> VectorOrTensorType::getShape() const {
  switch (getKind()) {
  case Kind::Vector:
    return cast<VectorType>(this)->getShape();
  case Kind::RankedTensor:
    return cast<RankedTensorType>(this)->getShape();
  case Kind::UnrankedTensor:
    return cast<RankedTensorType>(this)->getShape();
  default:
    llvm_unreachable("not a VectorOrTensorType");
  }
}

bool VectorOrTensorType::hasStaticShape() const {
  auto dims = getShape();
  return !std::any_of(dims.begin(), dims.end(), [](int i) { return i < 0; });
}

VectorType::VectorType(ArrayRef<int> shape, Type *elementType,
                       MLIRContext *context)
    : VectorOrTensorType(Kind::Vector, context, elementType, shape.size()),
      shapeElements(shape.data()) {}

TensorType::TensorType(Kind kind, Type *elementType, MLIRContext *context)
    : VectorOrTensorType(kind, context, elementType) {
  assert(isValidTensorElementType(elementType));
}

RankedTensorType::RankedTensorType(ArrayRef<int> shape, Type *elementType,
                                   MLIRContext *context)
  : TensorType(Kind::RankedTensor, elementType, context),
    shapeElements(shape.data()) {
  setSubclassData(shape.size());
}

UnrankedTensorType::UnrankedTensorType(Type *elementType, MLIRContext *context)
  : TensorType(Kind::UnrankedTensor, elementType, context) {
}

MemRefType::MemRefType(ArrayRef<int> shape, Type *elementType,
                       ArrayRef<AffineMap> affineMapList, unsigned memorySpace,
                       MLIRContext *context)
    : Type(Kind::MemRef, context, shape.size()), elementType(elementType),
      shapeElements(shape.data()), numAffineMaps(affineMapList.size()),
      affineMapList(affineMapList.data()), memorySpace(memorySpace) {}

ArrayRef<AffineMap> MemRefType::getAffineMaps() const {
  return ArrayRef<AffineMap>(affineMapList, numAffineMaps);
}

unsigned MemRefType::getNumDynamicDims() const {
  unsigned numDynamicDims = 0;
  for (int dimSize : getShape()) {
    if (dimSize == -1)
      ++numDynamicDims;
  }
  return numDynamicDims;
}
