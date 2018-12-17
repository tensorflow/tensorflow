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

UnknownLoc Builder::getUnknownLoc() { return UnknownLoc::get(context); }

UniquedFilename Builder::getUniquedFilename(StringRef filename) {
  return UniquedFilename::get(filename, context);
}

FileLineColLoc Builder::getFileLineColLoc(UniquedFilename filename,
                                          unsigned line, unsigned column) {
  return FileLineColLoc::get(filename, line, column, context);
}

Location Builder::getFusedLoc(ArrayRef<Location> locs, Attribute metadata) {
  return FusedLoc::get(locs, metadata, context);
}

//===----------------------------------------------------------------------===//
// Types.
//===----------------------------------------------------------------------===//

FloatType Builder::getBF16Type() { return Type::getBF16(context); }

FloatType Builder::getF16Type() { return Type::getF16(context); }

FloatType Builder::getF32Type() { return Type::getF32(context); }

FloatType Builder::getF64Type() { return Type::getF64(context); }

IndexType Builder::getIndexType() { return Type::getIndex(context); }

OtherType Builder::getTFControlType() { return Type::getTFControl(context); }

OtherType Builder::getTFResourceType() { return Type::getTFResource(context); }

OtherType Builder::getTFVariantType() { return Type::getTFVariant(context); }

OtherType Builder::getTFComplex64Type() {
  return Type::getTFComplex64(context);
}

OtherType Builder::getTFComplex128Type() {
  return Type::getTFComplex128(context);
}

OtherType Builder::getTFF32REFType() { return Type::getTFF32REF(context); }

OtherType Builder::getTFStringType() { return Type::getTFString(context); }

IntegerType Builder::getI1Type() { return Type::getInteger(1, context); }

IntegerType Builder::getIntegerType(unsigned width) {
  return Type::getInteger(width, context);
}

FunctionType Builder::getFunctionType(ArrayRef<Type> inputs,
                                      ArrayRef<Type> results) {
  return FunctionType::get(inputs, results, context);
}

MemRefType Builder::getMemRefType(ArrayRef<int> shape, Type elementType,
                                  ArrayRef<AffineMap> affineMapComposition,
                                  unsigned memorySpace) {
  return MemRefType::get(shape, elementType, affineMapComposition, memorySpace);
}

VectorType Builder::getVectorType(ArrayRef<int> shape, Type elementType) {
  return VectorType::get(shape, elementType);
}

RankedTensorType Builder::getTensorType(ArrayRef<int> shape, Type elementType) {
  return RankedTensorType::get(shape, elementType);
}

UnrankedTensorType Builder::getTensorType(Type elementType) {
  return UnrankedTensorType::get(elementType);
}

//===----------------------------------------------------------------------===//
// Attributes.
//===----------------------------------------------------------------------===//

BoolAttr Builder::getBoolAttr(bool value) {
  return BoolAttr::get(value, context);
}

IntegerAttr Builder::getIntegerAttr(int64_t value) {
  return IntegerAttr::get(getIntegerType(64), APInt(64, value));
}

IntegerAttr Builder::getIntegerAttr(Type type, int64_t value) {
  if (type.isIndex())
    return IntegerAttr::get(type, APInt(64, value));
  return IntegerAttr::get(type, APInt(type.getBitWidth(), value));
}

IntegerAttr Builder::getIntegerAttr(Type type, const APInt &value) {
  return IntegerAttr::get(type, value);
}

FloatAttr Builder::getFloatAttr(double value) {
  return FloatAttr::get(getF64Type(), APFloat(value));
}

FloatAttr Builder::getFloatAttr(float value) {
  return FloatAttr::get(getF32Type(), APFloat(value));
}

FloatAttr Builder::getFloatAttr(Type type, double value) {
  return FloatAttr::get(type, value);
}

FloatAttr Builder::getFloatAttr(Type type, const APFloat &value) {
  return FloatAttr::get(type, value);
}

StringAttr Builder::getStringAttr(StringRef bytes) {
  return StringAttr::get(bytes, context);
}

ArrayAttr Builder::getArrayAttr(ArrayRef<Attribute> value) {
  return ArrayAttr::get(value, context);
}

AffineMapAttr Builder::getAffineMapAttr(AffineMap map) {
  return AffineMapAttr::get(map);
}

IntegerSetAttr Builder::getIntegerSetAttr(IntegerSet set) {
  return IntegerSetAttr::get(set);
}

TypeAttr Builder::getTypeAttr(Type type) {
  return TypeAttr::get(type, context);
}

FunctionAttr Builder::getFunctionAttr(const Function *value) {
  return FunctionAttr::get(value, context);
}

ElementsAttr Builder::getSplatElementsAttr(VectorOrTensorType type,
                                           Attribute elt) {
  return SplatElementsAttr::get(type, elt);
}

ElementsAttr Builder::getDenseElementsAttr(VectorOrTensorType type,
                                           ArrayRef<char> data) {
  return DenseElementsAttr::get(type, data);
}

ElementsAttr Builder::getSparseElementsAttr(VectorOrTensorType type,
                                            DenseIntElementsAttr indices,
                                            DenseElementsAttr values) {
  return SparseElementsAttr::get(type, indices, values);
}

ElementsAttr Builder::getOpaqueElementsAttr(VectorOrTensorType type,
                                            StringRef bytes) {
  return OpaqueElementsAttr::get(type, bytes);
}

//===----------------------------------------------------------------------===//
// Affine Expressions, Affine Maps, and Integet Sets.
//===----------------------------------------------------------------------===//

AffineMap Builder::getAffineMap(unsigned dimCount, unsigned symbolCount,
                                ArrayRef<AffineExpr> results,
                                ArrayRef<AffineExpr> rangeSizes) {
  return AffineMap::get(dimCount, symbolCount, results, rangeSizes);
}

AffineExpr Builder::getAffineDimExpr(unsigned position) {
  return mlir::getAffineDimExpr(position, context);
}

AffineExpr Builder::getAffineSymbolExpr(unsigned position) {
  return mlir::getAffineSymbolExpr(position, context);
}

AffineExpr Builder::getAffineConstantExpr(int64_t constant) {
  return mlir::getAffineConstantExpr(constant, context);
}

IntegerSet Builder::getIntegerSet(unsigned dimCount, unsigned symbolCount,
                                  ArrayRef<AffineExpr> constraints,
                                  ArrayRef<bool> isEq) {
  return IntegerSet::get(dimCount, symbolCount, constraints, isEq);
}

AffineMap Builder::getConstantAffineMap(int64_t val) {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
                        {getAffineConstantExpr(val)}, {});
}

AffineMap Builder::getDimIdentityMap() {
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                        {getAffineDimExpr(0)}, {});
}

AffineMap Builder::getMultiDimIdentityMap(unsigned rank) {
  SmallVector<AffineExpr, 4> dimExprs;
  dimExprs.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    dimExprs.push_back(getAffineDimExpr(i));
  return AffineMap::get(/*dimCount=*/rank, /*symbolCount=*/0, dimExprs, {});
}

AffineMap Builder::getSymbolIdentityMap() {
  return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/1,
                        {getAffineSymbolExpr(0)}, {});
}

AffineMap Builder::getSingleDimShiftAffineMap(int64_t shift) {
  // expr = d0 + shift.
  auto expr = getAffineDimExpr(0) + shift;
  return AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0, {expr}, {});
}

AffineMap Builder::getShiftedAffineMap(AffineMap map, int64_t shift) {
  SmallVector<AffineExpr, 4> shiftedResults;
  shiftedResults.reserve(map.getNumResults());
  for (auto resultExpr : map.getResults()) {
    shiftedResults.push_back(resultExpr + shift);
  }
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), shiftedResults,
                        map.getRangeSizes());
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
Instruction *CFGFuncBuilder::createOperation(const OperationState &state) {
  SmallVector<CFGValue *, 8> operands;
  operands.reserve(state.operands.size());
  // Allow null operands as they act as sentinal barriers between successor
  // operand lists.
  for (auto elt : state.operands)
    operands.push_back(elt ? cast<CFGValue>(elt) : nullptr);

  auto *op =
      Instruction::create(state.location, state.name, operands, state.types,
                          state.attributes, state.successors, context);
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

/// Create an operation given the fields.
OperationStmt *MLFuncBuilder::createOperation(Location location,
                                              OperationName name,
                                              ArrayRef<MLValue *> operands,
                                              ArrayRef<Type> types,
                                              ArrayRef<NamedAttribute> attrs) {
  auto *op = OperationStmt::create(location, name, operands, types, attrs,
                                   getContext());
  block->getStatements().insert(insertPoint, op);
  return op;
}

ForStmt *MLFuncBuilder::createFor(Location location,
                                  ArrayRef<MLValue *> lbOperands,
                                  AffineMap lbMap,
                                  ArrayRef<MLValue *> ubOperands,
                                  AffineMap ubMap, int64_t step) {
  auto *stmt =
      ForStmt::create(location, lbOperands, lbMap, ubOperands, ubMap, step);
  block->getStatements().insert(insertPoint, stmt);
  return stmt;
}

ForStmt *MLFuncBuilder::createFor(Location location, int64_t lb, int64_t ub,
                                  int64_t step) {
  auto lbMap = AffineMap::getConstantMap(lb, context);
  auto ubMap = AffineMap::getConstantMap(ub, context);
  return createFor(location, {}, lbMap, {}, ubMap, step);
}

IfStmt *MLFuncBuilder::createIf(Location location, ArrayRef<MLValue *> operands,
                                IntegerSet set) {
  auto *stmt = IfStmt::create(location, operands, set);
  block->getStatements().insert(insertPoint, stmt);
  return stmt;
}
