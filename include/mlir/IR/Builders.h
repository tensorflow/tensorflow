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
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Statements.h"

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
class BoolAttr;
class IntegerAttr;
class FloatAttr;
class StringAttr;
class ArrayAttr;
class AffineMapAttr;
class AffineMap;
class AffineExpr;
class AffineConstantExpr;
class AffineDimExpr;
class AffineSymbolExpr;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(Module *module);

  MLIRContext *getContext() const { return context; }

  Identifier getIdentifier(StringRef str);
  Module *createModule();

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

  // Attributes.
  BoolAttr *getBoolAttr(bool value);
  IntegerAttr *getIntegerAttr(int64_t value);
  FloatAttr *getFloatAttr(double value);
  StringAttr *getStringAttr(StringRef bytes);
  ArrayAttr *getArrayAttr(ArrayRef<Attribute *> value);
  AffineMapAttr *getAffineMapAttr(AffineMap *value);

  // Affine Expressions and Affine Map.
  AffineMap *getAffineMap(unsigned dimCount, unsigned symbolCount,
                          ArrayRef<AffineExpr *> results,
                          ArrayRef<AffineExpr *> rangeSizes);
  AffineDimExpr *getDimExpr(unsigned position);
  AffineSymbolExpr *getSymbolExpr(unsigned position);
  AffineConstantExpr *getConstantExpr(int64_t constant);
  AffineExpr *getAddExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getSubExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getMulExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getModExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getFloorDivExpr(AffineExpr *lhs, AffineExpr *rhs);
  AffineExpr *getCeilDivExpr(AffineExpr *lhs, AffineExpr *rhs);

  // TODO: Helpers for affine map/exprs, etc.
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

  // Add new basic block and set the insertion point to the end of it.
  BasicBlock *createBlock();

  // Instructions.
  OperationInst *createOperation(Identifier name, ArrayRef<CFGValue *> operands,
                                 ArrayRef<Type *> resultTypes,
                                 ArrayRef<NamedAttribute> attributes) {
    auto op =
        OperationInst::create(name, operands, resultTypes, attributes, context);
    block->getOperations().push_back(op);
    return op;
  }

  // Terminators.

  ReturnInst *createReturnInst(ArrayRef<CFGValue *> operands) {
    return insertTerminator(ReturnInst::create(operands));
  }

  BranchInst *createBranchInst(BasicBlock *dest) {
    return insertTerminator(BranchInst::create(dest));
  }

  CondBranchInst *createCondBranchInst(CFGValue *condition,
                                       BasicBlock *trueDest,
                                       BasicBlock *falseDest) {
    return insertTerminator(
        CondBranchInst::create(condition, trueDest, falseDest));
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

/// This class helps build an MLFunction.  Statements that are created are
/// automatically inserted at an insertion point or added to the current
/// statement block.
class MLFuncBuilder : public Builder {
public:
  MLFuncBuilder(MLFunction *function) : Builder(function->getContext()) {}

  MLFuncBuilder(StmtBlock *block) : MLFuncBuilder(block->getFunction()) {
    setInsertionPoint(block);
  }

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = StmtBlock::iterator();
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPoint(StmtBlock *block) {
    this->block = block;
    insertPoint = block->end();
  }

  OperationStmt *createOperation(Identifier name,
                                 ArrayRef<NamedAttribute> attributes) {
    auto op = new OperationStmt(name, attributes, context);
    block->getStatements().push_back(op);
    return op;
  }

  // Creates for statement. When step is not specified, it is set to 1.
  ForStmt *createFor(AffineConstantExpr *lowerBound,
                     AffineConstantExpr *upperBound,
                     AffineConstantExpr *step = nullptr);

  IfStmt *createIf() {
    auto stmt = new IfStmt();
    block->getStatements().push_back(stmt);
    return stmt;
  }

private:
  StmtBlock *block = nullptr;
  StmtBlock::iterator insertPoint;
};

} // namespace mlir

#endif
