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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

using ResourceIdMap =
    absl::flat_hash_map<std::pair<std::string, std::string>, int>;

using ResourceMap = absl::flat_hash_map<int, std::vector<VarHandleOp>>;

Type GetQuantizedTypeFromReadVariableOp(VarHandleOp var_handle_op) {
  Type ref_qtype = nullptr;
  for (auto *var_handle_user : var_handle_op.getResult().getUsers()) {
    auto read_variable_op = dyn_cast_or_null<ReadVariableOp>(var_handle_user);
    if (!read_variable_op) continue;
    for (auto *read_variable_user : read_variable_op.getResult().getUsers()) {
      auto q_op = dyn_cast_or_null<QuantizeOp>(read_variable_user);
      if (!q_op || ref_qtype) continue;
      ref_qtype = q_op.getResult().getType();
    }
  }
  return ref_qtype;
}

Type GetDequantizedTypeFromAssigneVariableOp(VarHandleOp var_handle_op) {
  Type ref_qtype = nullptr;
  for (auto *var_handle_user : var_handle_op.getResult().getUsers()) {
    auto assign_variable_op =
        dyn_cast_or_null<AssignVariableOp>(var_handle_user);
    if (!assign_variable_op) continue;
    auto value_op = assign_variable_op.getValue().getDefiningOp();
    auto dq_op = dyn_cast_or_null<DequantizeOp>(value_op);
    if (!dq_op || ref_qtype) continue;
    ref_qtype = dq_op.getInput().getType();
  }
  return ref_qtype;
}

class QuantizeVariablesPass
    : public QuantizeVariablesPassBase<QuantizeVariablesPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(QuantizeVariablesPass)
  explicit QuantizeVariablesPass() = default;

  void runOnOperation() override;

 private:
  // Outlines the regions of the WhileOp's cond and body and insert function
  // calls instead.
  void QuantizeVariable(OpBuilder &builder,
                        const std::vector<VarHandleOp> &var_handle_op);
};

void QuantizeVariablesPass::QuantizeVariable(
    OpBuilder &builder, const std::vector<VarHandleOp> &var_handle_ops) {
  // TODO(b/261940892): Refactoring quantize_variables.cc
  Type ref_qtype = nullptr;
  for (VarHandleOp var_handle_op : var_handle_ops) {
    if (ref_qtype) break;
    ref_qtype = GetQuantizedTypeFromReadVariableOp(var_handle_op);
    if (ref_qtype) break;
    ref_qtype = GetDequantizedTypeFromAssigneVariableOp(var_handle_op);
  }
  if (!ref_qtype) return;

  for (VarHandleOp var_handle_op : var_handle_ops) {
    for (auto *var_handle_user :
         llvm::make_early_inc_range(var_handle_op.getResult().getUsers())) {
      auto read_variable_op = dyn_cast_or_null<ReadVariableOp>(var_handle_user);
      if (!read_variable_op) continue;
      // Add dequantize.
      builder.setInsertionPointAfter(read_variable_op);
      auto new_read_variable_op =
          builder.create<ReadVariableOp>(read_variable_op.getLoc(), ref_qtype,
                                         read_variable_op.getResourceId());
      auto new_dq_op = builder.create<DequantizeOp>(
          read_variable_op.getLoc(), read_variable_op.getResult().getType(),
          new_read_variable_op.getResult());
      read_variable_op->replaceAllUsesWith(new_dq_op);
      read_variable_op.erase();
    }
    for (auto *var_handle_user :
         llvm::make_early_inc_range(var_handle_op.getResult().getUsers())) {
      auto assign_variable_op =
          dyn_cast_or_null<AssignVariableOp>(var_handle_user);
      if (!assign_variable_op) continue;
      auto *value_op = assign_variable_op.getValue().getDefiningOp();
      auto dq_op = dyn_cast_or_null<DequantizeOp>(value_op);
      if (dq_op) {
        Type output_type = dq_op.getInput().getType();
        auto qtype = quant::QuantizedType::getQuantizedElementType(output_type);
        if (qtype == quant::QuantizedType::getQuantizedElementType(ref_qtype)) {
          // Same quantization parameters, remove it.
          builder.setInsertionPoint(assign_variable_op);
          auto new_assign_variable_op = builder.create<AssignVariableOp>(
              assign_variable_op.getLoc(), assign_variable_op.getResourceId(),
              dq_op.getInput());
          assign_variable_op->replaceAllUsesWith(new_assign_variable_op);
        } else {
          // Otherwise, apply re-quantization.
          builder.setInsertionPoint(assign_variable_op);
          auto new_q_op = builder.create<QuantizeOp>(
              assign_variable_op.getLoc(), ref_qtype, dq_op.getInput(),
              TypeAttr::get(ref_qtype));
          auto new_assign_variable_op = builder.create<AssignVariableOp>(
              assign_variable_op.getLoc(), assign_variable_op.getResourceId(),
              new_q_op.getResult());
          assign_variable_op->replaceAllUsesWith(new_assign_variable_op);
        }
        assign_variable_op.erase();
        dq_op.erase();
      } else {
        // Add quantize op.
        builder.setInsertionPoint(assign_variable_op);
        auto new_q_op = builder.create<QuantizeOp>(
            assign_variable_op.getLoc(), ref_qtype,
            assign_variable_op.getValue(), TypeAttr::get(ref_qtype));
        auto new_assign_variable_op = builder.create<AssignVariableOp>(
            assign_variable_op.getLoc(), assign_variable_op.getResourceId(),
            new_q_op.getResult());
        assign_variable_op->replaceAllUsesWith(new_assign_variable_op);
        assign_variable_op.erase();
      }
    }
  }
  // Update resource tensors.
  for (VarHandleOp var_handle_op : var_handle_ops) {
    builder.setInsertionPoint(var_handle_op);
    auto output_type = UnrankedTensorType::get(TF::ResourceType::get(
        {ref_qtype.cast<TensorType>()}, builder.getContext()));
    auto new_var_handle_op = builder.create<VarHandleOp>(
        var_handle_op.getLoc(), output_type, var_handle_op.getContainer(),
        var_handle_op.getSharedName());
    var_handle_op->replaceAllUsesWith(new_var_handle_op);
    var_handle_op.erase();
  }
}

void QuantizeVariablesPass::runOnOperation() {
  ResourceIdMap resource_id_map;
  ResourceMap resource_map;

  // Collect all resource identities.
  getOperation().walk([&](TFL::VarHandleOp var_handle_op) {
    auto identity = std::make_pair(var_handle_op.getContainer().str(),
                                   var_handle_op.getSharedName().str());
    resource_id_map.insert(
        std::make_pair(identity, static_cast<int>(resource_id_map.size())));
    int resource_id = resource_id_map[identity];
    resource_map[resource_id].push_back(var_handle_op);
  });

  OpBuilder builder(getOperation().getContext());

  for (const auto &[identity, var_handle_op] : resource_map) {
    QuantizeVariable(builder, var_handle_op);
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect Quantize Variables pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePrepareQuantizeVariablesPass() {
  return std::make_unique<QuantizeVariablesPass>();
}

}  // namespace TFL
}  // namespace mlir
