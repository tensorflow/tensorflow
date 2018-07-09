//===- Builders.h - Helpers for constructing MLIR Classes -------*- C++ -*-===//
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

#ifndef MLIR_IR_BUILDERS_H
#define MLIR_IR_BUILDERS_H

#include "mlir/IR/CFGFunction.h"

namespace mlir {
class MLIRContext;
class Module;
class Type;
class PrimitiveType;
class IntegerType;
class FunctionType;
class VectorType;
class RankedTensorType;
class UnrankedTensorType;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(Module *module);

  MLIRContext *getContext() const { return context; }

  // Types.
  PrimitiveType *getAffineIntType();
  PrimitiveType *getBF16Type();
  PrimitiveType *getF16Type();
  PrimitiveType *getF32Type();
  PrimitiveType *getF64Type();
  IntegerType *getIntegerType(unsigned width);
  FunctionType *getFunctionType(ArrayRef<Type *> inputs,
                                ArrayRef<Type *> results);
  VectorType *getVectorType(ArrayRef<unsigned> shape, Type *elementType);
  RankedTensorType *getTensorType(ArrayRef<int> shape, Type *elementType);
  UnrankedTensorType *getTensorType(Type *elementType);

  // TODO: Helpers for affine map/exprs, etc.
  // TODO: Helpers for attributes.
  // TODO: Identifier
  // TODO: createModule()
protected:
  MLIRContext *context;
};

/// This class helps build a CFGFunction.  Instructions that are created are
/// automatically inserted at an insertion point or added to the current basic
/// block.
class CFGFuncBuilder : public Builder {
public:
  CFGFuncBuilder(BasicBlock *block)
      : Builder(block->getFunction()->getContext()),
        function(block->getFunction()) {
    setInsertionPoint(block);
  }
  CFGFuncBuilder(CFGFunction *function)
      : Builder(function->getContext()), function(function) {}

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = BasicBlock::iterator();
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPoint(BasicBlock *block) {
    this->block = block;
    insertPoint = block->end();
  }

  OperationInst *createOperation(Identifier name,
                                 ArrayRef<NamedAttribute> attributes) {
    auto op = new OperationInst(name, attributes, context);
    block->getOperations().push_back(op);
    return op;
  }

  // Terminators.

  ReturnInst *createReturnInst() { return insertTerminator(new ReturnInst()); }

  BranchInst *createBranchInst(BasicBlock *dest) {
    return insertTerminator(new BranchInst(dest));
  }

private:
  template <typename T>
  T *insertTerminator(T *term) {
    block->setTerminator(term);
    return term;
  }

  CFGFunction *function;
  BasicBlock *block = nullptr;
  BasicBlock::iterator insertPoint;
};

// TODO: MLFuncBuilder

} // namespace mlir

#endif
