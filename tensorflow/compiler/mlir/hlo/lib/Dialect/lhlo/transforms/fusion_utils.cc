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

#include "mlir-hlo/Dialect/lhlo/transforms/fusion_utils.h"

#include <algorithm>

#include "mlir/Dialect/Shape/IR/Shape.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"          // TF:llvm-project
#include "mlir/IR/Matchers.h"

// This file implements some helper functions and classes used to do fusion
// & code generation.

namespace mlir {
namespace lmhlo {

// Returns true if the op is an elementwise unary lmhlo op.
// TODO(disc): use fusibility interface
// TODO(disc): Unify with disc_supported_list.h and Elementwise Trait
bool isElementWiseUnary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AbsOp,
    lmhlo::CeilOp,
    lmhlo::ConvertOp,
    lmhlo::CopyOp,
    lmhlo::CosOp,
    lmhlo::ExpOp,
    lmhlo::FloorOp,
    lmhlo::IsFiniteOp,
    lmhlo::LogOp,
    lmhlo::NegOp,
    lmhlo::NotOp,
    lmhlo::RsqrtOp,
    lmhlo::SignOp,
    lmhlo::SqrtOp,
    lmhlo::TanhOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise binary lmhlo op.
// TODO(disc): use fusibility interface
bool isElementWiseBinary(Operation* op) {
  // clang-format off
  return isa<
    lmhlo::AddOp,
    lmhlo::AndOp,
    lmhlo::CompareOp,
    lmhlo::DivOp,
    lmhlo::MaxOp,
    lmhlo::MinOp,
    lmhlo::MulOp,
    lmhlo::OrOp,
    lmhlo::PowOp,
    lmhlo::SubOp
  >(op);
  // clang-format on
}

// Returns true if the op is an elementwise lmhlo op.
// TODO(disc): use fusibility interface
bool isElementWise(Operation* op) {
  return isElementWiseUnary(op) || isElementWiseBinary(op);
}

// Returns true if this op is a rank-2 row reduction.
bool isRank2RowReduction(Operation* op) {
  auto reduce_op = dyn_cast<lmhlo::ReduceOp>(op);
  if (!reduce_op || reduce_op.dimensions().getNumElements() != 1) return false;

  int rank = op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.dimensions().getValues<int64_t>();
  return ((*dimensions.begin() == 1) && (rank == 2));
}

// Returns true if this op is a rank-2 column reduction.
bool isRank2ColReduction(Operation* op) {
  auto reduce_op = dyn_cast<lmhlo::ReduceOp>(op);
  if (!reduce_op || reduce_op.dimensions().getNumElements() != 1) return false;

  int rank = op->getOperand(0).getType().cast<MemRefType>().getRank();
  auto dimensions = reduce_op.dimensions().getValues<int64_t>();
  return ((*dimensions.begin() == 0) && (rank == 2));
}

// Returns true if the op is supported by the downstreaming fusion codegen
// engine.
bool isFusible(Operation* op) {
  // Only scalar const are supported by the fusion codegen engine a.t.m.
  if (dyn_cast<lmhlo::ConstOp>(op)) {
    MemRefType type = op->getOperand(0).getType().cast<MemRefType>();
    return (type.getRank() == 0);
  }

  // All element ops are supported by the fusion codegen engine.
  if (isElementWise(op)) return true;

  // Only rank-2 tensor -> rank-1 tensor reduction are supported now.
  if (isRank2RowReduction(op) || isRank2ColReduction(op)) return true;

  // clang-format off
  return isa<
    lmhlo::BroadcastInDimOp,
    lmhlo::BroadcastOp,
    lmhlo::ConcatenateOp,
    lmhlo::DynamicBroadcastInDimOp,
    lmhlo::DynamicGatherOp,
    lmhlo::DynamicIotaOp,
    lmhlo::DynamicPadOp,
    lmhlo::DynamicReshapeOp,
    lmhlo::GatherOp,
    lmhlo::RealDynamicSliceOp,
    lmhlo::ReshapeOp,
    lmhlo::SelectOp,
    lmhlo::SliceOp,
    lmhlo::TransposeOp
  >(op);
  // clang-format on
}

// Returns the number of operands that are supposed to be written.
// For some ops (e.g. lmhlo ops), some operands are the output memrefs
// Thus these operands are supposed to be updated.
int getNumResultOperands(Operation* op) {
  if (!isa<LmhloOp>(op)) {
    return 0;
  }

  auto isWritable = [&](Value operand) -> bool {
    llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
    MemoryEffectOpInterface interface = dyn_cast<MemoryEffectOpInterface>(op);
    // Suppose that operands of op without `MemoryEffectOpInterface` are
    // readonly.
    if (!interface) return false;

    interface.getEffectsOnValue(operand, effects);
    return llvm::any_of(
        effects, [](const mlir::MemoryEffects::EffectInstance& instance) {
          return mlir::isa<mlir::MemoryEffects::Write>(instance.getEffect());
        });
  };

  return llvm::count_if(op->getOperands(),
                        [&](Value v) { return isWritable(v); });
}

// Returns data users of the value and its aliases (e.g. memref.cast).
// Here non-data users means DimOp, DeallocOp and ShapeOfOp.
SmallVector<Operation*, 4> getValueUsers(Value v) {
  SmallVector<Operation*, 4> users;
  SmallVector<Value, 4> worklist;
  worklist.push_back(v);
  while (!worklist.empty()) {
    Value curr = worklist.back();
    worklist.pop_back();
    for (Operation* user : curr.getUsers()) {
      // Skip non-data users
      if (isa<memref::DimOp, memref::DeallocOp, shape::ShapeOfOp>(user)) {
        continue;
      }
      // alias value
      if (isa<memref::CastOp>(user)) {
        worklist.push_back(user->getResult(0));
      } else {
        users.push_back(user);
      }
    }
  }
  return users;
}

// Create a new fusion pattern from a single op.
FusionPattern::FusionPattern(Operation* op) {
  op_list_.push_back(op);
  if (isRank2RowReduction(op)) {
    fusion_type_ = FusionType::kRowReduction;
  } else if (isRank2ColReduction(op)) {
    fusion_type_ = FusionType::kColReduction;
  } else if (mlir::lmhlo::isFusible(op)) {
    fusion_type_ = FusionType::kLoop;
  } else {
    fusion_type_ = FusionType::kNone;
  }
  dominant_op_ = op;
  calculateOperandsAndResults();
}

// Create a new fusion pattern from the ops inside the lmhlo fusion op.
FusionPattern::FusionPattern(lmhlo::FusionOp fusion_op) {
  for (Operation& op : fusion_op.region().getBlocks().front()) {
    op_list_.push_back(&op);
  }

  // Figure out fusion type and dominant op for the fusion pattern.
  for (Operation* op : op_list_) {
    if (isRank2RowReduction(op)) {
      fusion_type_ = FusionType::kRowReduction;
      dominant_op_ = op;
    } else if (isRank2ColReduction(op)) {
      if (fusion_type_ != FusionType::kRowReduction) {
        fusion_type_ = FusionType::kColReduction;
        dominant_op_ = op;
      }
    } else if (lmhlo::isFusible(op)) {
      // Ignore if already a kRowReduction or kColReduction, otherwise update
      // the fusion type to kLoop and dominant op to current op. This supposes
      // that the last op inside the block is a valid candidate dominant op if
      // the fusion pattern is a kLoop.
      if (fusion_type_ == FusionType::kNone ||
          fusion_type_ == FusionType::kLoop) {
        fusion_type_ = FusionType::kLoop;
        dominant_op_ = op;
      }
    } else if (!isa<lmhlo::TerminatorOp>(op)) {
      // Not a supported fusionOp, early stop.
      fusion_type_ = FusionType::kNone;
      dominant_op_ = nullptr;
      break;
    }
  }

  if (isFusible()) calculateOperandsAndResults();
}

// Create a new fusion pattern from a valid fusion op list.
FusionPattern::FusionPattern(SmallVectorImpl<Operation*>& op_list)
    : op_list_(op_list.begin(), op_list.end()) {
  calculateOperandsAndResults();
}

// Returns true if two fusion patterns can be merged into one bigger fusion
// pattern.
bool FusionPattern::isMergeable(FusionPattern& other) {
  if (!this->isFusible() || !other.isFusible()) return false;
  return true;
}

// Merges two fusion patterns and returns the merged pattern. The original
// pattern remains unmodified.
FusionPattern FusionPattern::merge(FusionPattern& other) {
  assert(isMergeable(other));
  FusionOpList new_op_list = op_list_;
  new_op_list.insert(new_op_list.end(), other.getOpList().begin(),
                     other.getOpList().end());
  FusionPattern new_fusion_pattern{new_op_list};

  FusionType newType = FusionType::kLoop;
  Operation* newDominant = getDominantOp();

  // kRowReduction + (kRowReduction | kColReduction | kLoop) = kRowReduction
  // kColReduction + (kColReduction | kLoop) = kColReduction
  // kLoop + kLoop = kLoop
  if (getFusionType() == FusionType::kRowReduction ||
      other.getFusionType() == FusionType::kRowReduction) {
    newType = FusionType::kRowReduction;
    if (getFusionType() != FusionType::kRowReduction)
      newDominant = other.getDominantOp();
  } else if (getFusionType() == FusionType::kColReduction ||
             other.getFusionType() == FusionType::kColReduction) {
    newType = FusionType::kColReduction;
    if (getFusionType() != FusionType::kColReduction)
      newDominant = other.getDominantOp();
  }

  new_fusion_pattern.setDominantOp(newDominant);
  new_fusion_pattern.setFusionType(newType);
  return new_fusion_pattern;
}

// Merges two fusion patterns and returns the merged pattern. Replaces the
// original pattern with new merged pattern.
FusionPattern& FusionPattern::mergeInplace(FusionPattern& other) {
  *this = merge(other);
  return *this;
}

// Returns the effective size (e.g. not counting const ops) of the ops this
// fusion pattern contains.
int FusionPattern::effectiveSize() {
  return llvm::count_if(
      op_list_, [](Operation* op) { return !matchPattern(op, m_Constant()); });
}

// Sorts the ops inside the fusion pattern according to the keys provided.
void FusionPattern::sortFusionOpListBy(DenseMap<Operation*, int>& op_to_idx) {
  std::sort(op_list_.begin(), op_list_.end(),
            [&](Operation* lhs, Operation* rhs) {
              return op_to_idx[lhs] < op_to_idx[rhs];
            });
}

// Calculates the inputs and outputs of the fusion pattern.
void FusionPattern::calculateOperandsAndResults() {
  DenseSet<Value> input_set;
  DenseSet<Value> result_set;
  DenseSet<Value> internal_result_set;
  DenseSet<Operation*> op_set(op_list_.begin(), op_list_.end());

  DenseMap<Value, Operation*> last_writer;
  for (Operation* op : op_list_) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value v : op->getOperands().drop_front(num_input_operand)) {
      bool inserted = last_writer.try_emplace(v, op).second;
      (void)inserted;
      assert(inserted);

      bool has_external_user = false;
      for (Operation* user : getValueUsers(v)) {
        if (!op_set.contains(user)) {
          has_external_user = true;
          break;
        }
      }

      if (has_external_user) {
        results_.push_back(v);
        root_ops_.push_back(op);
      } else {
        internal_results_.push_back(v);
      }
    }
  }

  for (Operation* op : op_list_) {
    int num_input_operand = op->getNumOperands() - getNumResultOperands(op);
    for (Value value : op->getOperands().take_front(num_input_operand)) {
      if (last_writer.find(value) != last_writer.end()) {
        // skip if defining op is in the pattern
        continue;
      }
      input_set.insert(value);
    }
  }

  for (Value v : input_set) operands_.push_back(v);
}

// Supports using EquivalenceClasses for Value
bool operator<(const ValueWrapper& lhs, const ValueWrapper& rhs) {
  auto lhs_value = lhs.getValue().getAsOpaquePointer();
  auto rhs_value = rhs.getValue().getAsOpaquePointer();
  return lhs_value < rhs_value;
}

// shape equality propagation based on the shape constrains of
// elementwise ops.
void ShapeConstraintAnalysis::PropagateEquality(
    const SmallVectorImpl<Operation*>& op_list) {
  bool converged = true;
  do {
    converged = true;
    auto update = [&](Value lhs, Value rhs,
                      EquivalenceClasses<ValueWrapper>& impl) {
      if (!impl.isEquivalent(ValueWrapper(lhs), ValueWrapper(rhs))) {
        converged = false;
        impl.unionSets(ValueWrapper(lhs), ValueWrapper(rhs));
      }
    };
    for (Operation* op : op_list) {
      int num_operand = op->getNumOperands();
      // Propagates same num_elements equality, and shape equality
      if (isElementWise(op)) {
        Value lhs = op->getOperand(0);
        for (Value rhs : op->getOperands().drop_front()) {
          update(lhs, rhs, same_num_elements_impl_);
          update(lhs, rhs, same_shape_impl_);
        }
      }
      // Propagates same num_elements equality, not shape equality
      if (isa<lmhlo::DynamicReshapeOp, lmhlo::ReshapeOp, lmhlo::TransposeOp>(
              op)) {
        Value input = op->getOperand(0);
        // The last operand is the output memref by design
        Value output = op->getOperand(num_operand - 1);
        update(input, output, same_num_elements_impl_);
      }
    }
  } while (!converged);
}

}  // namespace lmhlo
}  // namespace mlir
