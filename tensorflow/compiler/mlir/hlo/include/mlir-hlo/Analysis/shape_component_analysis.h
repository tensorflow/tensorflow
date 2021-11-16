/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_SHAPE_COMPONENT_ANALYSIS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_SHAPE_COMPONENT_ANALYSIS_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

// Analysis to infer shape information.
//
// This lazily analyzes the individual components of a shape or shape tensor.
// Results are cached but the cache is not consistent across IR mutations and
// needs to be reset in that case.
class ShapeComponentAnalysis {
 public:
  // Represents either the shape of a tensor or value of a tensor.
  class ShapeOrValueOfTensor {
    llvm::PointerIntPair<Value, 1, bool> p;

    explicit ShapeOrValueOfTensor(decltype(p) p) : p(p) {}
    ShapeOrValueOfTensor(Value v, bool isShapeTensor) : p(v, isShapeTensor) {}

   public:
    static ShapeOrValueOfTensor getShapeOf(Value v) { return {v, false}; }
    static ShapeOrValueOfTensor getValueOf(Value v) { return {v, true}; }
    Value value() const { return p.getPointer(); }
    bool isShapeTensor() const { return p.getInt(); }

    bool operator==(ShapeOrValueOfTensor rhs) const { return p == rhs.p; }
    bool operator!=(ShapeOrValueOfTensor rhs) const { return !(*this == rhs); }

    // Forward p's DenseMapInfo.
    struct DenseMapInfo {
      using PairInfo = llvm::DenseMapInfo<decltype(p)>;
      static inline ShapeOrValueOfTensor getEmptyKey() {
        return ShapeOrValueOfTensor(PairInfo::getEmptyKey());
      }
      static inline ShapeOrValueOfTensor getTombstoneKey() {
        return ShapeOrValueOfTensor(PairInfo::getTombstoneKey());
      }
      static unsigned getHashValue(ShapeOrValueOfTensor val) {
        return PairInfo::getHashValue(val.p);
      }
      static bool isEqual(ShapeOrValueOfTensor lhs, ShapeOrValueOfTensor rhs) {
        return lhs == rhs;
      }
    };
  };

  // Represents a dimension of a shape or shape tensor in the input. This is
  // used for the symbols of a symbolic dimension.
  struct Symbol {
    ShapeOrValueOfTensor source;
    size_t index;

    bool operator==(const Symbol &rhs) const {
      return source == rhs.source && index == rhs.index;
    }
    bool operator!=(const Symbol &rhs) const { return !(*this == rhs); }
  };

  // Represents the computed dimension of a shape or shape tensor. This can be a
  // constant or a combination of different symbols as described by an affine
  // expression.
  struct SymbolicDimension {
    SmallVector<Symbol, 1> symbols;
    AffineExpr expr;

    // Return true if this dimension is a constant equal to `value`.
    bool isConstant(int64_t value) const;
    // Returns true if this dimension is known to be not `-1`. This is useful
    // for reshapes.
    bool isKnownNotNegativeOne() const;
    // If this is a reference to a singular symbol, return it.
    Optional<Symbol> singleton() const;

    bool operator==(const SymbolicDimension &rhs) const {
      return expr == expr && symbols == rhs.symbols;
    }
    bool operator!=(const SymbolicDimension &rhs) const {
      return !(*this == rhs);
    }

    void dump(llvm::raw_ostream &os = llvm::outs()) const;
  };

  using DimensionsMap =
      DenseMap<ShapeOrValueOfTensor, std::vector<SymbolicDimension>,
               ShapeOrValueOfTensor::DenseMapInfo>;
  using ConstraintsMap = DenseMap<int, Symbol>;

 private:
  // Mapping from value to an array of symbolic dimensions.
  DimensionsMap dimensions;

  // Mapping of constraints derived from argument attributes.
  ConstraintsMap symbolicShapeConstraints;

  // Run the analysis on a value.
  void compute(ShapeOrValueOfTensor v);

 public:
  // Return the computed dimensions for a shape of a value.
  Optional<ArrayRef<SymbolicDimension>> dimensionsForShape(Value value);
  // Return the computed dimensions for a shape tensor.
  Optional<ArrayRef<SymbolicDimension>> dimensionsForShapeTensor(Value shape);

  // Clear analysis data structures.
  void reset();
};
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_SHAPE_COMPONENT_ANALYSIS_H_
