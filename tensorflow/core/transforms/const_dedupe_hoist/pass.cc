/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/const_dedupe_hoist/pass.h"

#include <forward_list>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {

namespace {

struct DedupeAndHoistConstantPass
    : DedupeAndHoistConstantBase<DedupeAndHoistConstantPass> {
  LogicalResult initialize(MLIRContext* context) override {
    tfg_const = StringAttr::get(context, "tfg.Const");
    value_id = StringAttr::get(context, "value");
    name_id = StringAttr::get(context, TFGraphDialect::getNameAttrKey());
    return success();
  }
  void runOnOperation() override;

  void RunOnGraphOrFuncOp(Operation* op);

  // Propagate all control deps of op to its users.
  void PropagateEdges(Operation* op);

  // Returns whether identity op is required.
  bool RequiresIdentity(Operation* op);

  FunctionTable* function_table;

  // Identifiers used for operation type & attributes checked.
  StringAttr tfg_const;
  StringAttr value_id;
  StringAttr name_id;
};

}  // namespace

// Checking ConstOp's for equivalence skipping names.
struct EquivalentConst : public llvm::DenseMapInfo<Operation*> {
  static unsigned getHashValue(const Operation* op_c) {
    auto* op = const_cast<Operation*>(op_c);
    auto hash = llvm::hash_value("");
    // We know only TFG ConstOp will be here, so can query the name attribute
    // from it.
    StringAttr name_id =
        cast<TFGraphDialect>(op->getDialect())->getNameAttrIdentifier();
    for (auto attr : op->getAttrs()) {
      // Skip name from hash.
      if (attr.getName() == name_id) continue;
      hash = llvm::hash_combine(hash, attr.getValue());
    }
    return hash;
  }

  static bool isEqual(const Operation* lhs_c, const Operation* rhs_c) {
    auto* lhs = const_cast<Operation*>(lhs_c);
    auto* rhs = const_cast<Operation*>(rhs_c);
    if (lhs == rhs) return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Attributes are stored sorted by name.
    StringAttr name_id =
        cast<TFGraphDialect>(lhs->getDialect())->getNameAttrIdentifier();
    for (auto it : llvm::zip(lhs->getAttrs(), rhs->getAttrs())) {
      NamedAttribute lhs_attr = std::get<0>(it);
      NamedAttribute rhs_attr = std::get<1>(it);
      if (lhs_attr.getName() != rhs_attr.getName()) return false;
      if (lhs_attr.getValue() != rhs_attr.getValue()) {
        if (lhs_attr.getName() != name_id) return false;
      }
    }
    return true;
  }
};

void DedupeAndHoistConstantPass::PropagateEdges(Operation* op) {
  SmallVector<Operation*> users(op->getUsers());
  Value new_const = op->getResult(1);
  // ConstOp's only have control operands, so any operand of the op is a
  // control operand.
  for (Operation* user : users) {
    SetVector<Value> operands;
    auto add_ctl_operands = [&](Operation* operation) {
      // Filter out where there is a control edge already.
      auto op_operands =
          llvm::make_filter_range(TFOp(operation).getControlOperands(),
                                  [&](Value v) { return v == new_const; });
      operands.insert(op_operands.begin(), op_operands.end());
    };
    add_ctl_operands(user);
    add_ctl_operands(op);
    // Erase all control operands (effectively deduping control operands).
    // TODO(jpienaar): This could be optimized by avoiding cases where we don't
    // need to dedupe etc.
    TFOp tf_user(user);
    user->eraseOperands(tf_user.getNonControlOperands().size(),
                        tf_user.getControlOperands().size());
    user->insertOperands(user->getNumOperands(), operands.takeVector());
  }
}

bool DedupeAndHoistConstantPass::RequiresIdentity(Operation* op) {
  for (Operation* user : op->getUsers())
    if (function_table->MaybeCall(user)) return true;
  return false;
}

static Operation* BuildIdentity(Operation* input, Operation* value,
                                Attribute value_id) {
  OperationState state(input->getLoc(), "tfg.Identity");
  state.addTypes(input->getResultTypes());
  state.addOperands({value->getResult(0)});

  SetVector<Value> operands;
  auto op_operands = TFOp(input).getControlOperands();
  operands.insert(op_operands.begin(), op_operands.end());
  state.addOperands(operands.takeVector());

  // All attributes except for value.
  // TODO(jpienaar): Consider reducing name of the identity op.
  auto attrs = llvm::to_vector(llvm::make_filter_range(
      input->getAttrs(),
      [&](NamedAttribute attr) { return attr.getName() == value_id; }));
  state.addAttributes(attrs);
  return OpBuilder(input).createOperation(state);
}

void DedupeAndHoistConstantPass::RunOnGraphOrFuncOp(Operation* op) {
  DenseMap<Operation*, std::vector<Operation*>, EquivalentConst> constant_ops;

  // Collect all small constant ops grouped by attributes.
  getOperation()->walk([&](Operation* op) {
    if (op->getName().getIdentifier() != tfg_const) return;

    ElementsAttr val = op->getAttr(value_id).cast<ElementsAttr>();
    if (val.getNumElements() > max_size_) return;
    constant_ops[op].push_back(op);
  });

  // Iterate over all constant ops and perform constant deduping.
  for (const auto& it : constant_ops) {
    if (it.second.size() > 1) {
      Operation* top = OpBuilder(it.second.front()).clone(*it.second.front());
      top->eraseOperands(0, top->getNumOperands());

      for (auto jt : it.second) {
        if (!assume_strict_calls_ && RequiresIdentity(jt)) {
          // Create a new identity node with all the control deps of the node
          // being replaced that forwards the value of top.
          Operation* id = BuildIdentity(jt, top, value_id);
          jt->replaceAllUsesWith(id);
        } else {
          // Just propagate control deps from the duplicated op to its users and
          // then replace uses with top.
          PropagateEdges(jt);
          jt->replaceAllUsesWith(top);
        }
        jt->erase();
      }
    }
  }
}

void DedupeAndHoistConstantPass::runOnOperation() {
  markAnalysesPreserved<FunctionTable>();

  ModuleOp module = getOperation();
  if (!assume_strict_calls_) {
    function_table = &getAnalysis<FunctionTable>();
    assume_strict_calls_ = function_table->empty();
  } else {
    assume_strict_calls_ = true;
  }

  for (auto& op : module.getOps())
    // Only hoist inside Graph or GraphFunc ops.
    if (isa<GraphFuncOp, GraphOp>(op)) RunOnGraphOrFuncOp(&op);
}

}  // namespace tfg
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::tfg::CreateDedupeAndHoistConstantPass() {
  return std::make_unique<mlir::tfg::DedupeAndHoistConstantPass>();
}
