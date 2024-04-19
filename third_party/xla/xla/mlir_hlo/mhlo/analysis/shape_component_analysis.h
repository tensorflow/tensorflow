/* Copyright 2021 The OpenXLA Authors.

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

#ifndef MLIR_HLO_MHLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H
#define MLIR_HLO_MHLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H

#include <optional>

#include "llvm/Support/raw_ostream.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"

namespace mlir {

// Analysis to infer shape information.
//
// This lazily analyzes the individual components of a shape (e.g., the
// dimensions of a tensor) or value (e.g, the elements of a shape tensor).
// Results are cached but the cache is not consistent across IR mutations and
// needs to be reset in that case.
class ShapeComponentAnalysis {
 public:
  // Represents the analysis request for a specific value. We are either
  // interested in the shape of a value or the value itself.
  class ShapeOrValueInfo {
    llvm::PointerIntPair<Value, 1, bool> p;

    explicit ShapeOrValueInfo(decltype(p) p) : p(p) {}
    ShapeOrValueInfo(Value v, bool isValueInfo) : p(v, isValueInfo) {}

   public:
    static ShapeOrValueInfo getShapeInfoOf(Value v) { return {v, false}; }
    static ShapeOrValueInfo getValueInfoOf(Value v) { return {v, true}; }
    Value value() const { return p.getPointer(); }
    bool isValueInfo() const { return p.getInt(); }
    bool isShapeInfo() const { return !isValueInfo(); }

    bool operator==(ShapeOrValueInfo rhs) const { return p == rhs.p; }
    bool operator!=(ShapeOrValueInfo rhs) const { return !(*this == rhs); }

    // Forward p's DenseMapInfo.
    struct DenseMapInfo {
      using PairInfo = llvm::DenseMapInfo<decltype(p)>;
      static inline ShapeOrValueInfo getEmptyKey() {
        return ShapeOrValueInfo(PairInfo::getEmptyKey());
      }
      static inline ShapeOrValueInfo getTombstoneKey() {
        return ShapeOrValueInfo(PairInfo::getTombstoneKey());
      }
      static unsigned getHashValue(ShapeOrValueInfo val) {
        return PairInfo::getHashValue(val.p);
      }
      static bool isEqual(ShapeOrValueInfo lhs, ShapeOrValueInfo rhs) {
        return lhs == rhs;
      }
    };
  };

  // Symbolically represents one component of a shape (e.g., the dimensions of a
  // tensor) or value (e.g, the elements of a shape tensor). This is used to tie
  // symbolic expressions to components of shapes or values.
  struct Symbol {
    ShapeOrValueInfo source;
    size_t index;

    bool operator==(const Symbol &rhs) const {
      return source == rhs.source && index == rhs.index;
    }
    bool operator!=(const Symbol &rhs) const { return !(*this == rhs); }
  };

  // Represents the analysis result for a one component of a shape (e.g., the
  // dimensions of a tensor) or value (e.g, the elements of a shape tensor).
  // This can be a constant or an expression over symbols.
  struct SymbolicExpr {
    SmallVector<Symbol, 1> symbols;
    AffineExpr expr;

    // Returns true if this symbolic expression is known to be a constant equal
    // to `value`.
    bool isConstant(int64_t value) const;
    // Returns true if this symbolic expression is known to be different from
    // `-1`. This is useful for reshapes.
    bool isKnownNotNegativeOne() const;
    // Returns true if thus symbolic expression is known to be different from
    // `1`. This is useful for broadcasts.
    bool isKnownNotOne() const;
    // If this is a reference to a singular symbol, return it.
    std::optional<Symbol> singleton() const;

    bool operator==(const SymbolicExpr &rhs) const {
      return expr == rhs.expr && symbols == rhs.symbols;
    }
    bool operator!=(const SymbolicExpr &rhs) const { return !(*this == rhs); }

    void dump(llvm::raw_ostream &os = llvm::outs()) const;
  };

  using SymbolicExprsMap = DenseMap<ShapeOrValueInfo, std::vector<SymbolicExpr>,
                                    ShapeOrValueInfo::DenseMapInfo>;
  using SymbolicShapeConstraintsMap = DenseMap<int, Symbol>;

 private:
  // Mapping from the analysis requests to the results, i.e. to an array of
  // symbolic expressions. This is essentially a cache for all the results of
  // this analysis.
  SymbolicExprsMap symbolicExprsMap;

  // Mapping from symbolic shape constraints, derived from the argument
  // attributes, to the symbols used in this analysis.
  SymbolicShapeConstraintsMap symbolicShapeConstraintsMap;

  // Run the analysis to request either shape or value information.
  void compute(ShapeOrValueInfo v);

 public:
  // Return the computed components for the shape of a value, e.g., the
  // dimensions of a tensor.
  std::optional<ArrayRef<SymbolicExpr>> GetShapeInfo(Value value);
  // Return the computed components for the value of a value, e.g, the elements
  // of a shape tensor.
  std::optional<ArrayRef<SymbolicExpr>> GetValueInfo(Value shape);

  // Clear analysis data structures.
  void reset();
};
}  // namespace mlir

namespace llvm {

template <>
struct DenseMapInfo<mlir::ShapeComponentAnalysis::Symbol> {
  static inline mlir::ShapeComponentAnalysis::Symbol getEmptyKey() {
    return {mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
                getEmptyKey(),
            llvm::DenseMapInfo<size_t>::getEmptyKey()};
  }
  static inline mlir::ShapeComponentAnalysis::Symbol getTombstoneKey() {
    return {mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
                getTombstoneKey(),
            llvm::DenseMapInfo<size_t>::getTombstoneKey()};
  }
  static unsigned getHashValue(mlir::ShapeComponentAnalysis::Symbol symbol) {
    return llvm::hash_combine(
        mlir::ShapeComponentAnalysis::ShapeOrValueInfo::DenseMapInfo::
            getHashValue(symbol.source),
        llvm::DenseMapInfo<size_t>::getHashValue(symbol.index));
  }
  static bool isEqual(mlir::ShapeComponentAnalysis::Symbol lhs,
                      mlir::ShapeComponentAnalysis::Symbol rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // MLIR_HLO_MHLO_ANALYSIS_SHAPE_COMPONENT_ANALYSIS_H
