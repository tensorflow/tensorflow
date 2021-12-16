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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_FUSION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_FUSION_UTILS_H_

#include <memory>
#include <vector>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project

// This file implements some helper functions and classes used to do fusion
// & code generation.

namespace mlir {
namespace lmhlo {

// kLoop fusion template satisfies:
//   - all ops in the fusion pattern are element-wise.
//   - all the shapes of outputs of fusion pattern are same or have same number
//   of elements, and thus can fit into a same parallel loop.
//
// kInput fusion template satisfies:
//   - any op in the fusion pattern is either element-wise or a reduction.
//   - if a op is a reduction, its output cannot be consumed by other
//     ops in the same fusion pattern.
//   - all the effective shapes of outputs of fusion pattern are same.
//     - For element-wise op, its effective shape is its output shape.
//     - For reduction op, its effective shape is its operand shape.
//   - currently our downstreaming codegen engine only support 2d -> 1d tensor
//   reduction. TODO: lift this limitation.
//     - 2D row reduction: out[i] = sum({in[i][j] for all j})
//     - 2D column reduction: out[j] = sum({in[i][j] for all i})
enum FusionType {
  // Not a fusion pattern
  kNone,
  // kLoop fusion pattern
  kLoop,
  // kInput fusion pattern and all reduce ops of the fused pattern are row
  // reduction
  kRowReduction,
  // kInput fusion pattern and all reduce ops of the fused pattern are column
  // reduction
  kColReduction,
};

// Returns true if the op is an elementwise unary lmhlo op.
// TODO: use fusibility interface
bool isElementWiseUnary(Operation* op);

// Returns true if the op is an elementwise binary lmhlo op.
// TODO: use fusibility interface
bool isElementWiseBinary(Operation* op);

// Returns true if the op is an elementwise lmhlo op.
// TODO: use fusibility interface
bool isElementWise(Operation* op);

// Returns true if this op is a rank-2 row reduction.
bool isRank2RowReduction(Operation* op);

// Returns true if this op is a rank-2 column reduction.
bool isRank2ColReduction(Operation* op);

// Returns true if the op is supported by the downstreaming fusion codegen
// engine.
bool isFusible(Operation* op);

// Returns the number of operands that are supposed to be written.
// For some ops (e.g. lmhlo ops), some operands are the output memrefs
// Thus these operands are supposed to be updated.
int getNumResultOperands(Operation* op);

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v);

// Represents a list of lmhlo ops that are going to be fused.
class FusionPattern {
 public:
  using FusionOpList = SmallVector<Operation*, 4>;
  using FusionValueList = SmallVector<Value, 4>;

  // Create a new fusion pattern from a single op.
  FusionPattern(Operation* op);

  // Create a new fusion pattern from the ops inside the lmhlo fusion op.
  FusionPattern(lmhlo::FusionOp op);

  // Returns the op list this fusion pattern represents.
  FusionOpList& getOpList() { return op_list_; }

  // Returns the dominant op of this fusion pattern.
  // For kLoop fusion, a dominant op may be any op that has external users.
  // For kInput fusion, a dominant op may be a row reduction (if exists), or
  // a column reduction op.
  Operation* getDominantOp() { return dominant_op_; }

  // Sets the dominant op to the op provided.
  void setDominantOp(Operation* op) { dominant_op_ = op; }

  // Returns the fusion kind of the fusion pattern.
  FusionType getFusionType() { return fusion_type_; }

  // Sets the fusion type to the the type provided.
  void setFusionType(FusionType type) { fusion_type_ = type; }

  // Returns true if this a fusible fusion pattern.
  bool isFusible() { return getFusionType() != FusionType::kNone; }

  // Returns true if this fusion pattern is a kLoop fusion.
  bool isKLoopFusion() { return getFusionType() == FusionType::kLoop; }

  // Returns true if this fusion pattern is a kInput fusion.
  bool isKInputFusion() {
    return (getFusionType() == FusionType::kRowReduction ||
            getFusionType() == FusionType::kColReduction);
  }

  // Returns true if two fusion patterns can be merged into one bigger fusion
  // pattern.
  bool isMergeable(FusionPattern& other);

  // Merges two fusion patterns and returns the merged pattern. The original
  // pattern remains unmodified.
  FusionPattern merge(FusionPattern& other);

  // Merges two fusion patterns and returns the merged pattern. Replaces the
  // original pattern with new merged pattern.
  FusionPattern& mergeInplace(FusionPattern& other);

  // Returns values that are consumed by the lmhlo ops inside the fusion
  // pattern.
  FusionValueList& getOperands() { return operands_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  FusionValueList& getResults() { return results_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // have consumers outside the fusion pattern.
  SmallVector<Operation*, 4>& getRootOps() { return root_ops_; }

  // Returns values that are outputs of any lmhlo op in the fused pattern and
  // are only consumed by the lmhlo ops inside the fused pattern.
  FusionValueList& getInternalResults() { return internal_results_; }

  // Returns the size of the ops this fusion pattern contains.
  int size() { return op_list_.size(); }

  // Returns the effective size (e.g. not counting const ops) of the ops this
  // fusion pattern contains.
  int effectiveSize();

  // Sorts the ops inside the fusion pattern according to the keys provided.
  void sortFusionOpListBy(DenseMap<Operation*, int>& op_to_idx);

 private:
  FusionPattern(SmallVectorImpl<Operation*>& op_list);

 private:
  // Calculates the inputs and outputs of the fusion pattern.
  void calculateOperandsAndResults();

 private:
  FusionOpList op_list_;
  Operation* dominant_op_ = nullptr;
  FusionType fusion_type_ = FusionType::kNone;
  FusionValueList operands_;
  FusionValueList results_;
  FusionValueList internal_results_;
  SmallVector<Operation*, 4> root_ops_;
};

// Represents a list of disjoint fusion patterns for a block.
using FusionPlan = std::vector<FusionPattern>;

using llvm::EquivalenceClasses;

// Supports using EquivalenceClasses for Value
class ValueWrapper {
 public:
  explicit ValueWrapper(Value value) : value_(std::move(value)) {}

  Value getValue() const { return value_; }

  bool operator==(const ValueWrapper& rhs) const {
    return getValue() == rhs.getValue();
  }

 private:
  Value value_;
};

bool operator<(const ValueWrapper& lhs, const ValueWrapper& rhs);

// This is a simple shape constraint analysis, which is used to
// guide fusion decision (e.g. we only fuse shape-compatible ops).
//
// Currently, We only consider shape equality and same-number-elements equality
// propagation based on the shape constraint traits of elementwise ops (assuming
// that implicit shape broadcast is forbidden).
class ShapeConstraintAnalysis {
 public:
  explicit ShapeConstraintAnalysis(const SmallVectorImpl<Operation*>& op_list) {
    PropagateEquality(op_list);
  }

  // Returns true if `lhs` and `rhs` are supposed to have same shape.
  bool HasSameShape(Value lhs, Value rhs) {
    return same_shape_impl_.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs));
  }

  // Returns true if `lhs` and `rhs` are supposed to have same number of
  // elements.
  bool HasSameNumElements(Value lhs, Value rhs) {
    return same_num_elements_impl_.isEquivalent(ValueWrapper(lhs),
                                                ValueWrapper(rhs));
  }

  Value GetLeaderValueWithSameShape(Value val) const {
    if (same_shape_impl_.findLeader(ValueWrapper(val)) ==
        same_shape_impl_.member_end()) {
      return nullptr;
    }
    return same_shape_impl_.getLeaderValue(ValueWrapper(val)).getValue();
  }

 private:
  // shape equality propagation based on the shape constrains of
  // elementwise ops.
  void PropagateEquality(const SmallVectorImpl<Operation*>& op_list);

  // a UnionFind set
  EquivalenceClasses<ValueWrapper> same_shape_impl_;
  EquivalenceClasses<ValueWrapper> same_num_elements_impl_;
};

}  // namespace lmhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_LHLO_TRANSFORMS_FUSION_UTILS_H_
