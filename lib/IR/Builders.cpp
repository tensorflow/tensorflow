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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Types.h"
using namespace mlir;

Builder::Builder(Module *module) : context(module->getContext()) {}

Identifier Builder::getIdentifier(StringRef str) {
  return Identifier::get(str, context);
}

Module *Builder::createModule() { return new Module(context); }

//===----------------------------------------------------------------------===//
// Locations.
//===----------------------------------------------------------------------===//

UnknownLoc *Builder::getUnknownLoc() { return UnknownLoc::get(context); }

UniquedFilename Builder::getUniquedFilename(StringRef filename) {
  return UniquedFilename::get(filename, context);
}

FileLineColLoc *Builder::getFileLineColLoc(UniquedFilename filename,
                                           unsigned line, unsigned column) {
  return FileLineColLoc::get(filename, line, column, context);
}

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

FloatType *Builder::getBF16Type() { return Type::getBF16(context); }

FloatType *Builder::getF16Type() { return Type::getF16(context); }

FloatType *Builder::getF32Type() { return Type::getF32(context); }

FloatType *Builder::getF64Type() { return Type::getF64(context); }

OtherType *Builder::getAffineIntType() { return Type::getAffineInt(context); }

OtherType *Builder::getTFControlType() { return Type::getTFControl(context); }

OtherType *Builder::getTFResourceType() { return Type::getTFResource(context); }

OtherType *Builder::getTFVariantType() { return Type::getTFVariant(context); }

OtherType *Builder::getTFComplex64Type() {
  return Type::getTFComplex64(context);
}

OtherType *Builder::getTFComplex128Type() {
  return Type::getTFComplex128(context);
}

OtherType *Builder::getTFF32REFType() { return Type::getTFF32REF(context); }

OtherType *Builder::getTFStringType() { return Type::getTFString(context); }

IntegerType *Builder::getIntegerType(unsigned width) {
  return Type::getInteger(width, context);
}

FunctionType *Builder::getFunctionType(ArrayRef<Type *> inputs,
                                       ArrayRef<Type *> results) {
  return FunctionType::get(inputs, results, context);
}

MemRefType *Builder::getMemRefType(ArrayRef<int> shape, Type *elementType,
                                   ArrayRef<AffineMap *> affineMapComposition,
                                   unsigned memorySpace) {
  return MemRefType::get(shape, elementType, affineMapComposition, memorySpace);
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

//===----------------------------------------------------------------------===//
// Attributes.
//===----------------------------------------------------------------------===//

BoolAttr *Builder::getBoolAttr(bool value) {
  return BoolAttr::get(value, context);
}

IntegerAttr *Builder::getIntegerAttr(int64_t value) {
  return IntegerAttr::get(value, context);
}

FloatAttr *Builder::getFloatAttr(double value) {
  return FloatAttr::get(value, context);
}

StringAttr *Builder::getStringAttr(StringRef bytes) {
  return StringAttr::get(bytes, context);
}

ArrayAttr *Builder::getArrayAttr(ArrayRef<Attribute *> value) {
  return ArrayAttr::get(value, context);
}

AffineMapAttr *Builder::getAffineMapAttr(AffineMap *map) {
  return AffineMapAttr::get(map, context);
}

TypeAttr *Builder::getTypeAttr(Type *type) {
  return TypeAttr::get(type, context);
}

FunctionAttr *Builder::getFunctionAttr(const Function *value) {
  return FunctionAttr::get(value, context);
}

//===----------------------------------------------------------------------===//
// Affine Expressions, Affine Maps, and Integet Sets.
//===----------------------------------------------------------------------===//

AffineMap *Builder::getAffineMap(unsigned dimCount, unsigned symbolCount,
                                 ArrayRef<AffineExprRef> results,
                                 ArrayRef<AffineExprRef> rangeSizes) {
  return AffineMap::get(dimCount, symbolCount, results, rangeSizes, context);
}

AffineExprRef Builder::getDimExpr(unsigned position) {
  return AffineDimExpr::get(position, context);
}

AffineExprRef Builder::getSymbolExpr(unsigned position) {
  return AffineSymbolExpr::get(position, context);
}

AffineExprRef Builder::getConstantExpr(int64_t constant) {
  return AffineConstantExpr::get(constant, context);
}

AffineExprRef Builder::getAddExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::Add, lhs, rhs, context);
}

AffineExprRef Builder::getAddExpr(AffineExprRef lhs, int64_t rhs) {
  return AffineBinaryOpExpr::getAdd(lhs, rhs, context);
}

AffineExprRef Builder::getMulExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::Mul, lhs, rhs, context);
}

// Most multiply expressions are pure affine (rhs is a constant).
AffineExprRef Builder::getMulExpr(AffineExprRef lhs, int64_t rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::Mul, lhs,
                                 getConstantExpr(rhs), context);
}

AffineExprRef Builder::getSubExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return getAddExpr(lhs, getMulExpr(rhs, getConstantExpr(-1)));
}

AffineExprRef Builder::getSubExpr(AffineExprRef lhs, int64_t rhs) {
  return AffineBinaryOpExpr::getAdd(lhs, -rhs, context);
}

AffineExprRef Builder::getModExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::Mod, lhs, rhs, context);
}

// Most modulo expressions are pure affine.
AffineExprRef Builder::getModExpr(AffineExprRef lhs, uint64_t rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::Mod, lhs,
                                 getConstantExpr(rhs), context);
}

AffineExprRef Builder::getFloorDivExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::FloorDiv, lhs, rhs, context);
}

// Most floordiv expressions are pure affine.
AffineExprRef Builder::getFloorDivExpr(AffineExprRef lhs, uint64_t rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::FloorDiv, lhs,
                                 getConstantExpr(rhs), context);
}

AffineExprRef Builder::getCeilDivExpr(AffineExprRef lhs, AffineExprRef rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::CeilDiv, lhs, rhs, context);
}

// Most ceildiv expressions are pure affine.
AffineExprRef Builder::getCeilDivExpr(AffineExprRef lhs, uint64_t rhs) {
  return AffineBinaryOpExpr::get(AffineExpr::Kind::CeilDiv, lhs,
                                 getConstantExpr(rhs), context);
}

IntegerSet *Builder::getIntegerSet(unsigned dimCount, unsigned symbolCount,
                                   ArrayRef<AffineExprRef> constraints,
                                   ArrayRef<bool> isEq) {
  return IntegerSet::get(dimCount, symbolCount, constraints, isEq, context);
}

AffineMap *Builder::getConstantAffineMap(int64_t val) {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                        {getConstantExpr(val)}, {}, context);
}

AffineMap *Builder::getDimIdentityMap() {
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {getDimExpr(0)}, {},
                        context);
}

AffineMap *Builder::getSymbolIdentityMap() {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1, {getSymbolExpr(0)},
                        {}, context);
}

AffineMap *Builder::getSingleDimShiftAffineMap(int64_t shift) {
  // expr = d0 + shift.
  auto expr = getDimExpr(0) + shift;
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {expr}, {}, context);
}

AffineMap *Builder::getShiftedAffineMap(AffineMap *map, int64_t shift) {
  SmallVector<AffineExprRef, 4> shiftedResults;
  shiftedResults.reserve(map->getNumResults());
  for (auto resultExpr : map->getResults()) {
    shiftedResults.push_back(getAddExpr(resultExpr, shift));
  }
  return AffineMap::get(map->getNumDims(), map->getNumSymbols(), shiftedResults,
                        map->getRangeSizes(), context);
}

//===----------------------------------------------------------------------===//
// CFG function elements.
//===----------------------------------------------------------------------===//

/// Add new basic block and set the insertion point to the end of it.  If an
/// 'insertBefore' basic block is passed, the block will be placed before the
/// specified block.  If not, the block will be appended to the end of the
/// current function.
BasicBlock *CFGFuncBuilder::createBlock(BasicBlock *insertBefore) {
  BasicBlock *b = new BasicBlock();

  // If we are supposed to insert before a specific block, do so, otherwise add
  // the block to the end of the function.
  if (insertBefore)
    function->getBlocks().insert(CFGFunction::iterator(insertBefore), b);
  else
    function->push_back(b);

  setInsertionPoint(b);
  return b;
}

/// Create an operation given the fields represented as an OperationState.
OperationInst *CFGFuncBuilder::createOperation(const OperationState &state) {
  SmallVector<CFGValue *, 8> operands;
  operands.reserve(state.operands.size());
  for (auto elt : state.operands)
    operands.push_back(cast<CFGValue>(elt));

  auto *op = OperationInst::create(state.location, state.name, operands,
                                   state.types, state.attributes, context);
  block->getOperations().insert(insertPoint, op);
  return op;
}

//===----------------------------------------------------------------------===//
// Statements.
//===----------------------------------------------------------------------===//

/// Create an operation given the fields represented as an OperationState.
OperationStmt *MLFuncBuilder::createOperation(const OperationState &state) {
  SmallVector<MLValue *, 8> operands;
  operands.reserve(state.operands.size());
  for (auto elt : state.operands)
    operands.push_back(cast<MLValue>(elt));

  auto *op = OperationStmt::create(state.location, state.name, operands,
                                   state.types, state.attributes, context);
  block->getStatements().insert(insertPoint, op);
  return op;
}

ForStmt *MLFuncBuilder::createFor(Location *location,
                                  ArrayRef<MLValue *> lbOperands,
                                  AffineMap *lbMap,
                                  ArrayRef<MLValue *> ubOperands,
                                  AffineMap *ubMap, int64_t step) {
  auto *stmt = ForStmt::create(location, lbOperands, lbMap, ubOperands, ubMap,
                               step, context);
  block->getStatements().insert(insertPoint, stmt);
  return stmt;
}

ForStmt *MLFuncBuilder::createFor(Location *location, int64_t lb, int64_t ub,
                                  int64_t step) {
  auto *lbMap = AffineMap::getConstantMap(lb, context);
  auto *ubMap = AffineMap::getConstantMap(ub, context);
  return createFor(location, {}, lbMap, {}, ubMap, step);
}

IfStmt *MLFuncBuilder::createIf(Location *location,
                                ArrayRef<MLValue *> operands, IntegerSet *set) {
  auto *stmt = IfStmt::create(location, operands, set);
  block->getStatements().insert(insertPoint, stmt);
  return stmt;
}
