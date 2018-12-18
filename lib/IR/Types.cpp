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
#include "TypeDetail.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::detail;

Type::Kind Type::getKind() const { return type->kind; }

MLIRContext *Type::getContext() const { return type->context; }

unsigned Type::getSubclassData() const { return type->getSubclassData(); }
void Type::setSubclassData(unsigned val) { type->setSubclassData(val); }

IndexType::IndexType(Type::ImplType *ptr) : Type(ptr) {}

IntegerType::IntegerType(Type::ImplType *ptr) : Type(ptr) {}

unsigned IntegerType::getWidth() const {
  return static_cast<ImplType *>(type)->width;
}

FloatType::FloatType(Type::ImplType *ptr) : Type(ptr) {}

unsigned FloatType::getWidth() const {
  switch (getKind()) {
  case Type::Kind::BF16:
  case Type::Kind::F16:
    return 16;
  case Type::Kind::F32:
    return 32;
  case Type::Kind::F64:
    return 64;
  default:
    llvm_unreachable("unexpected type");
  }
}

/// Returns the floating semantics for the given type.
const llvm::fltSemantics &FloatType::getFloatSemantics() const {
  if (isBF16())
    // Treat BF16 like a double. This is unfortunate but BF16 fltSemantics is
    // not defined in LLVM.
    // TODO(jpienaar): add BF16 to LLVM? fltSemantics are internal to APFloat.cc
    // else one could add it.
    //  static const fltSemantics semBF16 = {127, -126, 8, 16};
    return APFloat::IEEEdouble();
  if (isF16())
    return APFloat::IEEEhalf();
  if (isF32())
    return APFloat::IEEEsingle();
  if (isF64())
    return APFloat::IEEEdouble();
  llvm_unreachable("non-floating point type used");
}

unsigned Type::getIntOrFloatBitWidth() const {
  assert(isIntOrFloat() && "only ints and floats have a bitwidth");
  if (auto intType = dyn_cast<IntegerType>()) {
    return intType.getWidth();
  }

  auto floatType = cast<FloatType>();
  return floatType.getWidth();
}

OtherType::OtherType(Type::ImplType *ptr) : Type(ptr) {}

FunctionType::FunctionType(Type::ImplType *ptr) : Type(ptr) {}

ArrayRef<Type> FunctionType::getInputs() const {
  return static_cast<ImplType *>(type)->getInputs();
}

unsigned FunctionType::getNumResults() const {
  return static_cast<ImplType *>(type)->numResults;
}

ArrayRef<Type> FunctionType::getResults() const {
  return static_cast<ImplType *>(type)->getResults();
}

VectorOrTensorType::VectorOrTensorType(Type::ImplType *ptr) : Type(ptr) {}

Type VectorOrTensorType::getElementType() const {
  return static_cast<ImplType *>(type)->elementType;
}

unsigned VectorOrTensorType::getElementTypeBitWidth() const {
  return getElementType().getIntOrFloatBitWidth();
}

unsigned VectorOrTensorType::getNumElements() const {
  switch (getKind()) {
  case Kind::Vector:
  case Kind::RankedTensor: {
    auto shape = getShape();
    unsigned num = 1;
    for (auto dim : shape)
      num *= dim;
    return num;
  }
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

/// If this is ranked tensor or vector type, return the rank. If it is an
/// unranked tensor, return -1.
int VectorOrTensorType::getRank() const {
  switch (getKind()) {
  case Kind::Vector:
  case Kind::RankedTensor:
    return getShape().size();
  case Kind::UnrankedTensor:
    return -1;
  default:
    llvm_unreachable("not a VectorOrTensorType");
  }
}

int VectorOrTensorType::getDimSize(unsigned i) const {
  switch (getKind()) {
  case Kind::Vector:
  case Kind::RankedTensor:
    return getShape()[i];
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

// Get the number of number of bits require to store a value of the given vector
// or tensor types.  Compute the value recursively since tensors are allowed to
// have vectors as elements.
long VectorOrTensorType::getSizeInBits() const {
  assert(hasStaticShape() &&
         "cannot get the bit size of an aggregate with a dynamic shape");

  auto elementType = getElementType();
  if (elementType.isIntOrFloat())
    return elementType.getIntOrFloatBitWidth() * getNumElements();

  // Tensors can have vectors and other tensors as elements, vectors cannot.
  assert(!isa<VectorType>() && "unsupported vector element type");
  auto elementVectorOrTensorType = elementType.dyn_cast<VectorOrTensorType>();
  assert(elementVectorOrTensorType && "unsupported tensor element type");
  return getNumElements() * elementVectorOrTensorType.getSizeInBits();
}

ArrayRef<int> VectorOrTensorType::getShape() const {
  switch (getKind()) {
  case Kind::Vector:
    return cast<VectorType>().getShape();
  case Kind::RankedTensor:
    return cast<RankedTensorType>().getShape();
  default:
    llvm_unreachable("not a VectorOrTensorType or not ranked");
  }
}

bool VectorOrTensorType::hasStaticShape() const {
  if (isa<UnrankedTensorType>())
    return false;
  auto dims = getShape();
  return !std::any_of(dims.begin(), dims.end(), [](int i) { return i < 0; });
}

VectorType::VectorType(Type::ImplType *ptr) : VectorOrTensorType(ptr) {}

ArrayRef<int> VectorType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

TensorType::TensorType(Type::ImplType *ptr) : VectorOrTensorType(ptr) {}

RankedTensorType::RankedTensorType(Type::ImplType *ptr) : TensorType(ptr) {}

ArrayRef<int> RankedTensorType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

UnrankedTensorType::UnrankedTensorType(Type::ImplType *ptr) : TensorType(ptr) {}

MemRefType::MemRefType(Type::ImplType *ptr) : Type(ptr) {}

ArrayRef<int> MemRefType::getShape() const {
  return static_cast<ImplType *>(type)->getShape();
}

Type MemRefType::getElementType() const {
  return static_cast<ImplType *>(type)->elementType;
}

ArrayRef<AffineMap> MemRefType::getAffineMaps() const {
  return static_cast<ImplType *>(type)->getAffineMaps();
}

unsigned MemRefType::getMemorySpace() const {
  return static_cast<ImplType *>(type)->memorySpace;
}

unsigned MemRefType::getNumDynamicDims() const {
  unsigned numDynamicDims = 0;
  for (int dimSize : getShape()) {
    if (dimSize == -1)
      ++numDynamicDims;
  }
  return numDynamicDims;
}
