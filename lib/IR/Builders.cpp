//===- Builders.cpp - Helpers for constructing MLIR Classes ---------------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
using namespace mlir;

Builder::Builder(Module *module) : context(module->getContext()) {}

// Types.
PrimitiveType *Builder::getAffineIntType() {
  return Type::getAffineInt(context);
}

PrimitiveType *Builder::getBF16Type() { return Type::getBF16(context); }

PrimitiveType *Builder::getF16Type() { return Type::getF16(context); }

PrimitiveType *Builder::getF32Type() { return Type::getF32(context); }

PrimitiveType *Builder::getF64Type() { return Type::getF64(context); }

IntegerType *Builder::getIntegerType(unsigned width) {
  return Type::getInteger(width, context);
}

FunctionType *Builder::getFunctionType(ArrayRef<Type *> inputs,
                                       ArrayRef<Type *> results) {
  return FunctionType::get(inputs, results, context);
}

VectorType *Builder::getVectorType(ArrayRef<unsigned> shape,
                                   Type *elementType) {
  return VectorType::get(shape, elementType);
}

RankedTensorType *Builder::getTensorType(ArrayRef<int> shape,
                                         Type *elementType) {
  return RankedTensorType::get(shape, elementType);
}

UnrankedTensorType *Builder::getTensorType(Type *elementType) {
  return UnrankedTensorType::get(elementType);
}
