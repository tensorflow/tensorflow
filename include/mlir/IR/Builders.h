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

#include "mlir/IR/Attributes.h"
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
  FloatType *getBF16Type();
  FloatType *getF16Type();
  FloatType *getF32Type();
  FloatType *getF64Type();

  OtherType *getAffineIntType();
  OtherType *getTFControlType();
  OtherType *getTFStringType();
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
  CFGFuncBuilder(BasicBlock *block, BasicBlock::iterator insertPoint)
      : Builder(block->getFunction()->getContext()),
        function(block->getFunction()) {
    setInsertionPoint(block, insertPoint);
  }

  CFGFuncBuilder(OperationInst *insertBefore)
      : CFGFuncBuilder(insertBefore->getBlock(),
                       BasicBlock::iterator(insertBefore)) {}

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

  /// Set the insertion point to the specified location.
  void setInsertionPoint(BasicBlock *block, BasicBlock::iterator insertPoint) {
    assert(block->getFunction() == function &&
           "can't move to a different function");
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Set the insertion point to the specified operation.
  void setInsertionPoint(OperationInst *inst) {
    setInsertionPoint(inst->getBlock(), BasicBlock::iterator(inst));
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPoint(BasicBlock *block) {
    setInsertionPoint(block, block->end());
  }

  // Add new basic block and set the insertion point to the end of it.
  BasicBlock *createBlock();

  // Create an operation at the current insertion point.
  OperationInst *createOperation(Identifier name, ArrayRef<CFGValue *> operands,
                                 ArrayRef<Type *> resultTypes,
                                 ArrayRef<NamedAttribute> attributes) {
    auto op =
        OperationInst::create(name, operands, resultTypes, attributes, context);
    block->getOperations().insert(insertPoint, op);
    return op;
  }

  OperationInst *cloneOperation(const OperationInst &srcOpInst) {
    auto *op = srcOpInst.clone();
    block->getOperations().insert(insertPoint, op);
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
  /// Create ML function builder and set insertion point to the given
  /// statement block, that is, given ML function, for statement or if statement
  /// clause.
  MLFuncBuilder(StmtBlock *block)
      : Builder(block->findFunction()->getContext()) {
    setInsertionPoint(block);
  }

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = StmtBlock::iterator();
  }

  /// Set the insertion point to the specified location.
  /// Unlike CFGFuncBuilder, MLFuncBuilder allows to set insertion
  /// point to a different function.
  void setInsertionPoint(StmtBlock *block, StmtBlock::iterator insertPoint) {
    // TODO: check that insertPoint is in this rather than some other block.
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Set the insertion point to the specified operation.
  void setInsertionPoint(OperationStmt *stmt) {
    setInsertionPoint(stmt->getBlock(), StmtBlock::iterator(stmt));
  }

  /// Set the insertion point to the end of the specified block.
  void setInsertionPoint(StmtBlock *block) {
    this->block = block;
    insertPoint = block->end();
  }

  /// Set the insertion point at the beginning of the specified block.
  void setInsertionPointAtStart(StmtBlock *block) {
    this->block = block;
    insertPoint = block->begin();
  }

  OperationStmt *createOperation(Identifier name, ArrayRef<MLValue *> operands,
                                 ArrayRef<Type *> resultTypes,
                                 ArrayRef<NamedAttribute> attributes) {
    auto *op =
        OperationStmt::create(name, operands, resultTypes, attributes, context);
    block->getStatements().insert(insertPoint, op);
    return op;
  }

  OperationStmt *cloneOperation(const OperationStmt &srcOpStmt) {
    auto *op = srcOpStmt.clone();
    block->getStatements().insert(insertPoint, op);
    return op;
  }

  // Creates for statement. When step is not specified, it is set to 1.
  ForStmt *createFor(AffineConstantExpr *lowerBound,
                     AffineConstantExpr *upperBound,
                     AffineConstantExpr *step = nullptr);

  IfStmt *createIf() {
    auto *stmt = new IfStmt();
    block->getStatements().insert(insertPoint, stmt);
    return stmt;
  }

  // TODO: subsume with a generate create<ConstantInt>() method.
  OperationStmt *createConstInt32Op(int value) {
    std::pair<Identifier, Attribute *> namedAttr(
        Identifier::get("value", context), getIntegerAttr(value));
    auto *mlconst = createOperation(Identifier::get("constant", context), {},
                                    {getIntegerType(32)}, {namedAttr});
    return mlconst;
  }

private:
  StmtBlock *block = nullptr;
  StmtBlock::iterator insertPoint;
};

} // namespace mlir

#endif
