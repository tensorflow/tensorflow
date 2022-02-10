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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
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

  // All the functions in the graph.
  DenseSet<StringAttr> functions;

  // Whether to consider all calls as strict. In TensorFlow calls can be strict
  // (all operands to call should have been executed before op is) or non-strict
  // (execute whatever is possible given known values, so part of a function
  // could be evaluated). This member indicates that all calls in the module can
  // be treated as strict.
  bool strict_calls = false;

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

void DedupeAndHoistConstantPass::RunOnGraphOrFuncOp(Operation* op) {
  DenseMap<Operation*, std::vector<Operation*>, EquivalentConst> constant_ops;

  // Collect all small constant ops grouped by attributes.
  getOperation()->walk([&](Operation* op) {
    if (op->getName().getIdentifier() != tfg_const) return;

    ElementsAttr val = op->getAttr(value_id).cast<ElementsAttr>();
    if (val.getNumElements() > max_size_) return;
    constant_ops[op].push_back(op);
  });

  // Propagate the control deps on the op to users of the op.
  auto propagate_edges = [](Operation* op) {
    SmallVector<Operation*> users(op->getUsers());
    // ConstOp's only have control operands, so any operand of the op is a
    // control operand.
    for (Operation* user : users) {
      SetVector<Value> operands;
      auto add_ctl_operands = [&](Operation* operation) {
        auto op_operands = operation->getOperands();
        operands.insert(op_operands.begin(), op_operands.end());
      };
      add_ctl_operands(user);
      // Record the number of unique control deps here as the original op could
      // have had duplicates.
      int existing = operands.getArrayRef().size();
      add_ctl_operands(op);

      auto remaining = operands.getArrayRef().drop_front(existing);
      if (!remaining.empty())
        user->insertOperands(user->getNumOperands(), remaining);
    }
  };

  // Iterate over all constant ops and perform constant deduping.
  for (const auto& it : constant_ops) {
    if (it.second.size() > 1) {
      Operation* top = it.second.front();
      if (top->getNumOperands() > 0) {
        propagate_edges(top);
        top->eraseOperands(0, top->getNumOperands());
      }
      for (auto jt : llvm::drop_begin(it.second)) {
        jt->replaceAllUsesWith(top);
        propagate_edges(jt);
        jt->erase();
      }
    }
  }
}

void DedupeAndHoistConstantPass::runOnOperation() {
  ModuleOp module = dyn_cast<ModuleOp>(getOperation());
  if (!module) return;

  // Collect function names (to be used for disambiguating legacy call
  // behavior).
  for (auto& op : module.getOps()) {
    if (auto func = dyn_cast<GraphFuncOp>(op))
      functions.insert(func.getNameAttr());
  }
  strict_calls = functions.empty();

  // This only supports the strict calls case for now, which is satisfied if
  // there are no functions to call.
  // TODO(jpienaar): Expand this to check for calls/function references and in
  // those cases insert identity nodes.
  if (!strict_calls) {
    LOG(WARNING)
        << "Skipping deduping conservatively because functions are present";
    return;
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
