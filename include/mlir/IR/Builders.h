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

#include "mlir/IR/Function.h"
#include "mlir/IR/Instruction.h"

namespace mlir {

class AffineExpr;
class BlockAndValueMapping;
class Module;
class UnknownLoc;
class UniquedFilename;
class FileLineColLoc;
class Type;
class PrimitiveType;
class IntegerType;
class FunctionType;
class MemRefType;
class VectorType;
class RankedTensorType;
class UnrankedTensorType;
class BoolAttr;
class IntegerAttr;
class FloatAttr;
class StringAttr;
class TypeAttr;
class ArrayAttr;
class FunctionAttr;
class ElementsAttr;
class DenseElementsAttr;
class DenseIntElementsAttr;
class AffineMapAttr;
class AffineMap;

/// This class is a general helper class for creating context-global objects
/// like types, attributes, and affine expressions.
class Builder {
public:
  explicit Builder(MLIRContext *context) : context(context) {}
  explicit Builder(Module *module);

  MLIRContext *getContext() const { return context; }

  Identifier getIdentifier(StringRef str);
  Module *createModule();

  // Locations.
  UnknownLoc getUnknownLoc();
  UniquedFilename getUniquedFilename(StringRef filename);
  FileLineColLoc getFileLineColLoc(UniquedFilename filename, unsigned line,
                                   unsigned column);
  Location getFusedLoc(ArrayRef<Location> locs,
                       Attribute metadata = Attribute());

  // Types.
  FloatType getBF16Type();
  FloatType getF16Type();
  FloatType getF32Type();
  FloatType getF64Type();

  IndexType getIndexType();

  IntegerType getI1Type();
  IntegerType getIntegerType(unsigned width);
  FunctionType getFunctionType(ArrayRef<Type> inputs, ArrayRef<Type> results);
  MemRefType getMemRefType(ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<AffineMap> affineMapComposition = {},
                           unsigned memorySpace = 0);
  VectorType getVectorType(ArrayRef<int64_t> shape, Type elementType);
  RankedTensorType getTensorType(ArrayRef<int64_t> shape, Type elementType);
  UnrankedTensorType getTensorType(Type elementType);

  /// Get or construct an instance of the type 'ty' with provided arguments.
  template <typename Ty, typename... Args> Ty getType(Args... args) {
    return Ty::get(context, args...);
  }

  // Attributes.
  BoolAttr getBoolAttr(bool value);
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  IntegerAttr getIntegerAttr(Type type, const APInt &value);
  FloatAttr getFloatAttr(Type type, double value);
  FloatAttr getFloatAttr(Type type, const APFloat &value);
  StringAttr getStringAttr(StringRef bytes);
  ArrayAttr getArrayAttr(ArrayRef<Attribute> value);
  AffineMapAttr getAffineMapAttr(AffineMap map);
  IntegerSetAttr getIntegerSetAttr(IntegerSet set);
  TypeAttr getTypeAttr(Type type);
  FunctionAttr getFunctionAttr(const Function *value);
  ElementsAttr getSplatElementsAttr(VectorOrTensorType type, Attribute elt);
  ElementsAttr getDenseElementsAttr(VectorOrTensorType type,
                                    ArrayRef<char> data);
  ElementsAttr getDenseElementsAttr(VectorOrTensorType type,
                                    ArrayRef<Attribute> values);
  ElementsAttr getSparseElementsAttr(VectorOrTensorType type,
                                     DenseIntElementsAttr indices,
                                     DenseElementsAttr values);
  ElementsAttr getOpaqueElementsAttr(VectorOrTensorType type, StringRef bytes);
  // Returns a 0-valued attribute of the given `type`. This function only
  // supports boolean, integer, and 32-/64-bit float types, and vector or ranked
  // tensor of them. Returns null attribute otherwise.
  Attribute getZeroAttr(Type type);

  // Convenience methods for fixed types.
  FloatAttr getF32FloatAttr(float value);
  FloatAttr getF64FloatAttr(double value);
  IntegerAttr getI32IntegerAttr(int32_t value);
  IntegerAttr getI64IntegerAttr(int64_t value);

  // Affine expressions and affine maps.
  AffineExpr getAffineDimExpr(unsigned position);
  AffineExpr getAffineSymbolExpr(unsigned position);
  AffineExpr getAffineConstantExpr(int64_t constant);

  AffineMap getAffineMap(unsigned dimCount, unsigned symbolCount,
                         ArrayRef<AffineExpr> results,
                         ArrayRef<AffineExpr> rangeSizes);

  // Special cases of affine maps and integer sets
  /// Returns a single constant result affine map with 0 dimensions and 0
  /// symbols.  One constant result: () -> (val).
  AffineMap getConstantAffineMap(int64_t val);
  // One dimension id identity map: (i) -> (i).
  AffineMap getDimIdentityMap();
  // Multi-dimensional identity map: (d0, d1, d2) -> (d0, d1, d2).
  AffineMap getMultiDimIdentityMap(unsigned rank);
  // One symbol identity map: ()[s] -> (s).
  AffineMap getSymbolIdentityMap();

  /// Returns a map that shifts its (single) input dimension by 'shift'.
  /// (d0) -> (d0 + shift)
  AffineMap getSingleDimShiftAffineMap(int64_t shift);

  /// Returns an affine map that is a translation (shift) of all result
  /// expressions in 'map' by 'shift'.
  /// Eg: input: (d0, d1)[s0] -> (d0, d1 + s0), shift = 2
  ///   returns:    (d0, d1)[s0] -> (d0 + 2, d1 + s0 + 2)
  AffineMap getShiftedAffineMap(AffineMap map, int64_t shift);

  // Integer set.
  IntegerSet getIntegerSet(unsigned dimCount, unsigned symbolCount,
                           ArrayRef<AffineExpr> constraints,
                           ArrayRef<bool> isEq);
  // TODO: Helpers for affine map/exprs, etc.
protected:
  MLIRContext *context;
};

/// This class helps build a Function.  Instructions that are created are
/// automatically inserted at an insertion point.  The builder is copyable.
class FuncBuilder : public Builder {
public:
  /// Create a function builder and set the insertion point to the start of
  /// the function.
  FuncBuilder(Function *func) : Builder(func->getContext()), function(func) {
    if (!func->empty())
      setInsertionPoint(&func->front(), func->front().begin());
    else
      clearInsertionPoint();
  }

  /// Create a function builder and set insertion point to the given
  /// instruction, which will cause subsequent insertions to go right before it.
  FuncBuilder(Instruction *inst) : FuncBuilder(inst->getFunction()) {
    setInsertionPoint(inst);
  }

  FuncBuilder(Block *block) : FuncBuilder(block->getFunction()) {
    setInsertionPoint(block, block->end());
  }

  FuncBuilder(Block *block, Block::iterator insertPoint)
      : FuncBuilder(block->getFunction()) {
    setInsertionPoint(block, insertPoint);
  }

  /// Return the function this builder is referring to.
  Function *getFunction() const { return function; }

  /// Reset the insertion point to no location.  Creating an operation without a
  /// set insertion point is an error, but this can still be useful when the
  /// current insertion point a builder refers to is being removed.
  void clearInsertionPoint() {
    this->block = nullptr;
    insertPoint = Block::iterator();
  }

  /// Set the insertion point to the specified location.
  void setInsertionPoint(Block *block, Block::iterator insertPoint) {
    // TODO: check that insertPoint is in this rather than some other block.
    this->block = block;
    this->insertPoint = insertPoint;
  }

  /// Sets the insertion point to the specified operation, which will cause
  /// subsequent insertions to go right before it.
  void setInsertionPoint(Instruction *inst) {
    setInsertionPoint(inst->getBlock(), Block::iterator(inst));
  }

  /// Sets the insertion point to the start of the specified block.
  void setInsertionPointToStart(Block *block) {
    setInsertionPoint(block, block->begin());
  }

  /// Sets the insertion point to the end of the specified block.
  void setInsertionPointToEnd(Block *block) {
    setInsertionPoint(block, block->end());
  }

  /// Return the block the current insertion point belongs to.  Note that the
  /// the insertion point is not necessarily the end of the block.
  Block *getInsertionBlock() const { return block; }

  /// Returns the current insertion point of the builder.
  Block::iterator getInsertionPoint() const { return insertPoint; }

  /// Add new block and set the insertion point to the end of it.  If an
  /// 'insertBefore' block is passed, the block will be placed before the
  /// specified block.  If not, the block will be appended to the end of the
  /// current function.
  Block *createBlock(Block *insertBefore = nullptr);

  /// Returns the current block of the builder.
  Block *getBlock() const { return block; }

  /// Creates an operation given the fields represented as an OperationState.
  OperationInst *createOperation(const OperationState &state);

  /// Create operation of specific op type at the current insertion point.
  template <typename OpTy, typename... Args>
  OpPointer<OpTy> create(Location location, Args... args) {
    OperationState state(getContext(), location, OpTy::getOperationName());
    OpTy::build(this, &state, args...);
    auto *inst = createOperation(state);
    auto result = inst->dyn_cast<OpTy>();
    assert(result && "Builder didn't return the right type");
    return result;
  }

  /// Creates a deep copy of the specified instruction, remapping any operands
  /// that use values outside of the instruction using the map that is provided
  /// ( leaving them alone if no entry is present).  Replaces references to
  /// cloned sub-instructions to the corresponding instruction that is copied,
  /// and adds those mappings to the map.
  Instruction *clone(const Instruction &inst, BlockAndValueMapping &mapper) {
    Instruction *cloneInst = inst.clone(mapper, getContext());
    block->getInstructions().insert(insertPoint, cloneInst);
    return cloneInst;
  }
  Instruction *clone(const Instruction &inst) {
    Instruction *cloneInst = inst.clone(getContext());
    block->getInstructions().insert(insertPoint, cloneInst);
    return cloneInst;
  }

private:
  Function *function;
  Block *block = nullptr;
  Block::iterator insertPoint;
};

} // namespace mlir

#endif
