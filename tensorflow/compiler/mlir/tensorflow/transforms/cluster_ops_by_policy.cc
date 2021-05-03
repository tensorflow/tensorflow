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

#include "tensorflow/compiler/mlir/tensorflow/transforms/cluster_ops_by_policy.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace TFDevice {

// -------------------------------------------------------------------------- //
// ValueConstraint.
// -------------------------------------------------------------------------- //

ValueConstraint Merge(ValueConstraint a, ValueConstraint b) {
  return a > b ? a : b;
}

raw_ostream& operator<<(raw_ostream& os, const ValueConstraint& constraint) {
  auto str = [](ValueConstraint constraint) -> StringRef {
    switch (constraint) {
      case ValueConstraint::kRank:
        return "rank";
      case ValueConstraint::kShape:
        return "shape";
      case ValueConstraint::kValue:
        return "value";
      default:
        llvm_unreachable("unknown value constraint");
    }
  };

  os << str(constraint);
  return os;
}

// -------------------------------------------------------------------------- //
// ValuesConstraintSet.
// -------------------------------------------------------------------------- //

void ValuesConstraintSet::Insert(ValueRange values,
                                 ValueConstraint constraint) {
  for (Value value : values) Insert(value, constraint);
}

std::pair<ValueConstraint, bool> ValuesConstraintSet::Insert(
    Value value, ValueConstraint constraint) {
  auto emplaced = constraints_.try_emplace(value, constraint);
  ValueConstraint persisted = emplaced.first->getSecond();

  // We've just inserted a new constraint for the value.
  if (emplaced.second) return {persisted, true};

  // Update existing constraint with a new one.
  auto merged = Merge(constraint, persisted);
  return {constraints_[value] = merged, merged != persisted};
}

void ValuesConstraintSet::Walk(
    llvm::function_ref<void(Value, ValueConstraint)> walk) const {
  for (auto& kv : constraints_) walk(kv.getFirst(), kv.getSecond());
}

Optional<ValueConstraint> ValuesConstraintSet::GetConstraint(
    Value value) const {
  auto it = constraints_.find(value);
  if (it == constraints_.end()) return None;
  return it->getSecond();
}

bool ValuesConstraintSet::HasConstraint(Value value) const {
  return GetConstraint(value).hasValue();
}

ValuesConstraintSet& ValuesConstraintSet::Reset() {
  constraints_.clear();
  return *this;
}

size_t ValuesConstraintSet::Size() const { return constraints_.size(); }

bool ValuesConstraintSet::Empty() const { return constraints_.empty(); }

// -------------------------------------------------------------------------- //
// PropagateValueConstraints.
// -------------------------------------------------------------------------- //

mlir::LogicalResult PropagateValuesConstraints(
    mlir::Region& region, const ClusteringPolicySet& policies,
    ValuesConstraintSet& constraints) {
  // A set of constraints for operation results.
  llvm::DenseMap<Operation*, ValuesConstraintSet> op_results_constraints;

  // Use initial constraints to initialize op results constraints.
  for (std::pair<Value, ValueConstraint> pair : constraints) {
    Value value = pair.first;
    ValueConstraint constraint = pair.second;

    // Value must be defined by an operation.
    Operation* op = value.getDefiningOp();
    assert(op && "value must be defined by an operation");
    if (!op) return failure();

    // Operation must be in the region.
    Block* ancestorBlock = region.findAncestorBlockInRegion(*op->getBlock());
    assert(ancestorBlock && "operation must be in the region");
    if (!ancestorBlock) return failure();

    op_results_constraints[op].Insert(value, constraint);
  }

  // Keep a worklist of operations that need their constraints to be updated.
  llvm::SetVector<Operation*> worklist;
  region.walk([&](Operation* op) { worklist.insert(op); });  // process all ops

  while (!worklist.empty()) {
    Operation* op = worklist.pop_back_val();

    // Use results constraints to infer operands constraints.
    const ValuesConstraintSet& results = op_results_constraints[op];
    ValuesConstraintSet operands;

    // Walk through all policies until we find one that matches the operation.
    bool updated = false;
    for (auto& policy : policies.policies()) {
      auto matched =
          policy->MatchAndUpdateConstraints(op, results, operands.Reset());
      if (succeeded(matched)) {
        updated = true;
        break;
      }
    }

    // Signal a failure if could not propagate non-empty constraints on the
    // operation results to the operands.
    if (!updated && !results.Empty()) {
      op->emitError("failed to propagate results constraints");
      return failure();
    }

    // Update results constraints based on inferred operands constraints.
    operands.Walk([&](Value value, ValueConstraint constraint) {
      // Update constraint for a value.
      auto updated = constraints.Insert(value, constraint);
      if (!updated.second) return;

      // Maybe update constaint on the operation result, but do not follow
      // operations that are outside of the `region`.
      Operation* op = value.getDefiningOp();
      if (!op || !region.findAncestorBlockInRegion(*op->getBlock())) return;

      // Add updated operation to the worklist.
      auto inserted = op_results_constraints[op].Insert(value, updated.first);
      if (inserted.second) worklist.insert(op);
    });
  }

  return success();
}

void EmitValueConstraintsRemarks(const ValuesConstraintSet& constraints) {
  constraints.Walk([](Value value, ValueConstraint constraint) {
    for (OpOperand& operand : value.getUses())
      operand.getOwner()->emitRemark(
          llvm::formatv("operand #{0} constrained to: {1}",
                        operand.getOperandNumber(), constraint));
  });
}

LogicalResult InferFunctionBodyValuesConstraints(
    FuncOp func, ValuesConstraintSet& constraints) {
  for (unsigned i = 0; i < func.getNumResults(); ++i) {
    auto str = func.getResultAttrOfType<StringAttr>(i, "tf.constraint");
    if (!str) continue;

    ValueConstraint constraint = StringSwitch<ValueConstraint>(str.getValue())
                                     .Case("rank", ValueConstraint::kRank)
                                     .Case("shape", ValueConstraint::kShape)
                                     .Case("value", ValueConstraint::kValue);

    // Propagate constraints through function return operations.
    for (Block& block : func.body()) {
      ReturnOp ret = dyn_cast<ReturnOp>(block.back());
      if (ret) constraints.Insert(ret.getOperand(i), constraint);
    }
  }

  return success();
}

}  // namespace TFDevice
}  // namespace mlir
