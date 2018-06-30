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
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/STLExtras.h"
using namespace mlir;

PrimitiveType::PrimitiveType(TypeKind kind, MLIRContext *context)
  : Type(kind, context) {
}

IntegerType::IntegerType(unsigned width, MLIRContext *context)
  : Type(TypeKind::Integer, context), width(width) {
}

FunctionType::FunctionType(Type *const *inputsAndResults, unsigned numInputs,
                           unsigned numResults, MLIRContext *context)
  : Type(TypeKind::Function, context, numInputs),
    numResults(numResults), inputsAndResults(inputsAndResults) {
}

VectorType::VectorType(ArrayRef<unsigned> shape, PrimitiveType *elementType,
                       MLIRContext *context)
  : Type(TypeKind::Vector, context, shape.size()),
    shapeElements(shape.data()), elementType(elementType) {
}

RankedTensorType::RankedTensorType(ArrayRef<int> shape, Type *elementType,
                                   MLIRContext *context)
  : TensorType(TypeKind::RankedTensor, elementType, context),
    shapeElements(shape.data()) {
  setSubclassData(shape.size());
}

UnrankedTensorType::UnrankedTensorType(Type *elementType, MLIRContext *context)
  : TensorType(TypeKind::UnrankedTensor, elementType, context) {
}

void Type::print(raw_ostream &os) const {
  switch (getKind()) {
  case TypeKind::AffineInt: os << "affineint"; return;
  case TypeKind::BF16: os << "bf16"; return;
  case TypeKind::F16:  os << "f16"; return;
  case TypeKind::F32:  os << "f32"; return;
  case TypeKind::F64:  os << "f64"; return;

  case TypeKind::Integer: {
    auto *integer = cast<IntegerType>(this);
    os << 'i' << integer->getWidth();
    return;
  }
  case TypeKind::Function: {
    auto *func = cast<FunctionType>(this);
    os << '(';
    interleave(func->getInputs(),
               [&](Type *type) { os << *type; },
               [&]() { os << ", "; });
    os << ") -> ";
    auto results = func->getResults();
    if (results.size() == 1)
      os << *results[0];
    else {
      os << '(';
      interleave(results,
                 [&](Type *type) { os << *type; },
                 [&]() { os << ", "; });
      os << ")";
    }
    return;
  }
  case TypeKind::Vector: {
    auto *v = cast<VectorType>(this);
    os << "vector<";
    for (auto dim : v->getShape())
      os << dim << 'x';
    os << *v->getElementType() << '>';
    return;
  }
  case TypeKind::RankedTensor: {
    auto *v = cast<RankedTensorType>(this);
    os << "tensor<";
    for (auto dim : v->getShape()) {
      if (dim < 0)
        os << '?';
      else
        os << dim;
      os << 'x';
    }
    os << *v->getElementType() << '>';
    return;
  }
  case TypeKind::UnrankedTensor: {
    auto *v = cast<UnrankedTensorType>(this);
    os << "tensor<??" << *v->getElementType() << '>';
    return;
  }
  }
}

void Type::dump() const {
  print(llvm::errs());
}
