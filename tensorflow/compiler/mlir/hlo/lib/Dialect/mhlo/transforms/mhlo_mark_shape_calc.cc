/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"       // TF:local_config_mlir

namespace mlir {

using hlo::kCpu;
using hlo::kDiscShapeCalcAttr;

namespace mhlo {
namespace {

// Check Op if it is a mhlo Op.
bool isMhloDialect(Operation* op) {
  return (op->getDialect()->getTypeID() == TypeID::get<mhlo::MhloDialect>());
}

// This pass explicitly marks the shape calculating Op by adding an Attr. Nested
// FuncOps should be taken into consideration.
// Following Ops are shape Ops:
//  - i64 Scalar output
//  - Shape Op's operands
//  - Shape operands according to kShapeCalcOperandMap
// Following Ops regard as shape Ops:
//  - GetDimensionSizeOp, PrintOp
//  - ConstOp, SelectOp, IotaOp, DynamicIotaOp if type is i32
//  - mhlo.dynamic_gather and mhlo.gather if operand_0's type is i32
//  - Date operands but type is i32 according to kShapeCalcOperandMap
class MarkShapeCalc : public MarkShapeCalculationPassBase<MarkShapeCalc> {
 public:
  using MarkShapeCalculationPassBase<
      MarkShapeCalc>::MarkShapeCalculationPassBase;

  MarkShapeCalc() = default;
  MarkShapeCalc(const MarkShapeCalc& o) = default;

  LogicalResult initialize(MLIRContext* context) final {
    // Cache these during initialization to enable pointer comparison during
    // pass execution.
    cpu_placement_attr_ = StringAttr::get(context, kCpu);
    output_placement_attr_key_ =
        Identifier::get(hlo::kOutputPlacementAttr, context);
    true_attr_ = BoolAttr::get(context, true);
    return success();
  }
  void runOnOperation() final;

 private:
  // Mark shape calculation subgraph
  void MarkShapeCalcOps();

  // Regard any mhlo Ops that calculates I32 as shape calculation Ops
  void MarkRegardAsShapeCalcOps();

  // for rule based placement strategy, the placement of the op in the list
  // is up to the placement of the dominant operand
  const DenseMap<TypeID, /*dominant operand index*/ int> kPlaceRuleMap = {
      {TypeID::get<DynamicGatherOp>(), /*operand*/ 0},
      {TypeID::get<GatherOp>(), /*operand*/ 0}};

  const DenseMap<TypeID, SmallVector<int, 3>> kShapeCalcOperandMap = {
      {TypeID::get<RealDynamicSliceOp>(),
       {/*start_indices*/ 1, /*limit_indices*/ 2, /*strides*/ 3}},
      {TypeID::get<DynamicPadOp>(),
       {/*edge_padding_low*/ 2, /*edge_padding_high*/ 3,
        /*interior_padding*/ 4}},
      {TypeID::get<DynamicReshapeOp>(), {/*shape*/ 1}},
      {TypeID::get<DynamicIotaOp>(), {/*shape*/ 0}},
      {TypeID::get<DynamicBroadcastInDimOp>(), {/*out_dim_size*/ 1}},
      {TypeID::get<DynamicGatherOp>(), {/*slice_sizes*/ 2}},
      {TypeID::get<DynamicConvOp>(), {/*paddings*/ 2}},
      {TypeID::get<IfOp>(), {/*pred*/ 0}}};

  // add output OP into marked set if it is a I64 scalar and placment is CPU.
  void markI64ReturnedCpuScalarOps(FuncOp func,
                                   DenseSet<Operation*>& shape_calc_ops);
  // Update marked set.
  // If a OP is in marked set, add all of its operands to marked set.
  // Add some operands of dynamic shape OPs into marked set according to lookup
  // table.
  void markShapeCalculationOps(FuncOp func,
                               DenseSet<Operation*>& shape_calc_ops);

  // Cached context-owned entities for fast pointer-based access.
  StringAttr cpu_placement_attr_;
  Optional<Identifier> output_placement_attr_key_;
  BoolAttr true_attr_;
};

void MarkShapeCalc::runOnOperation() {
  // Mark shape calculation subgraph
  MarkShapeCalcOps();

  // Mark any mhlo Ops that calculates I32 as shape calculation Ops
  MarkRegardAsShapeCalcOps();
}

// Mark the Ops that is the producer of any shape operands
// TODO(disc): handle when TupleOp exists in shape_calc_ops
void MarkShapeCalc::MarkShapeCalcOps() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  llvm::DenseSet<Operation*> shape_calc_ops;

  module.walk([&](FuncOp func) {
    // Mark the i64 Scalar output as shape calculation Op.
    // TODO(disc): revisit this if we have outputs on CPU for TF in the future.
    if (func.getName() == "main")
      markI64ReturnedCpuScalarOps(func, shape_calc_ops);
    // Skip if this function is external
    if (func.isExternal()) return;
    // no target ops
    if (llvm::none_of(func.getBlocks().front(),
                      [](Operation& op) { return isMhloDialect(&op); })) {
      return;
    }
    markShapeCalculationOps(func, shape_calc_ops);
  });

  for (Operation* op : shape_calc_ops) {
    // We suppose that mhlo op only has single output, either having tensor
    // type or tuple type.
    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      // If an op is placed on cpu, then we suppose all its outputs are
      // placed on cpu.
      SmallVector<Attribute> attrs(tp.size(), true_attr_);
      op->setAttr(kDiscShapeCalcAttr, ArrayAttr::get(tp.getContext(), attrs));
    } else {
      op->setAttr(kDiscShapeCalcAttr, true_attr_);
    }
  }
}

// Regard any mhlo Ops that calculates i32 as shape Ops. This is an rule based
// optimization that mimicking the behavior of tensorflow
void MarkShapeCalc::MarkRegardAsShapeCalcOps() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  module.walk([&](Operation* op) {
    if (!isMhloDialect(op)) return;
    if (isa<mhlo::TupleOp, mhlo::GetTupleElementOp, mhlo::WhileOp, mhlo::IfOp,
            mhlo::ReturnOp>(op))
      return;

    // Skip the Op that is already marked shape Op
    auto attr = op->getAttrOfType<BoolAttr>(kDiscShapeCalcAttr);
    if ((attr != nullptr) && (attr.getValue() == true)) return;

    if (isa<mhlo::GetDimensionSizeOp, mhlo::PrintOp>(op)) {
      op->setAttr(kDiscShapeCalcAttr, true_attr_);
      return;
    }

    // Ops that only cares about the output element type
    if (isa<mhlo::ConstOp, mhlo::SelectOp, mhlo::IotaOp, mhlo::DynamicIotaOp>(
            op)) {
      auto result_ty = op->getResult(0).getType().dyn_cast<RankedTensorType>();
      assert(result_ty && "unexpected non ranked type for ConstOp");
      auto elem_type = result_ty.getElementType();
      if (elem_type.isInteger(32)) {
        op->setAttr(kDiscShapeCalcAttr, true_attr_);
      }
      return;
    }

    auto op_type_id = op->getRegisteredInfo()->getTypeID();
    bool is_shape_calc_op = false;
    // Follow the rule of kPlaceRuleMap exist, or else follow
    // kShapeCalcOperandMap
    auto it = kPlaceRuleMap.find(op_type_id);
    if (it != kPlaceRuleMap.end()) {
      auto dominant_idx = it->second;
      auto operand_ty =
          op->getOperand(dominant_idx).getType().dyn_cast<RankedTensorType>();
      assert(operand_ty && "unexpected non unranked type of operand");
      if (operand_ty.getElementType().isInteger(32)) {
        is_shape_calc_op = true;
      }
    } else {
      auto iter = kShapeCalcOperandMap.find(op_type_id);
      if (iter != kShapeCalcOperandMap.end()) {
        const SmallVector<int, 3>& shape_operand_indices = iter->second;
        for (int idx : shape_operand_indices) {
          auto operand_ty =
              op->getOperand(idx).getType().dyn_cast<RankedTensorType>();
          if (!operand_ty) continue;
          auto elem_type = operand_ty.getElementType();
          if (elem_type.isInteger(32)) {
            is_shape_calc_op = true;
            break;
          }
        }
      }
    }
    // Set attr if it is a shape Op
    if (is_shape_calc_op) {
      if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(), true_attr_);
        op->setAttr(kDiscShapeCalcAttr, ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op->setAttr(kDiscShapeCalcAttr, true_attr_);
      }
    }
    return;
  });
}

void MarkShapeCalc::markI64ReturnedCpuScalarOps(
    FuncOp func, llvm::DenseSet<Operation*>& shape_calc_ops) {
  assert(func.getName() == "main");
  auto return_op = func.front().getTerminator();
  if (!isa<mlir::ReturnOp>(return_op)) return;
  auto result_attrs = func.getAllResultAttrs();
  if (!result_attrs) return;
  auto returned_ops = return_op->getOperands();
  assert(returned_ops.size() == result_attrs.size());
  for (auto output : llvm::enumerate(returned_ops)) {
    Operation* op = output.value().getDefiningOp();
    if (!op || !isMhloDialect(op)) continue;
    int idx = output.index();
    if (auto type = op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
      if (type.getElementType().isInteger(64) && (type.getRank() == 0) &&
          (result_attrs[idx].cast<DictionaryAttr>().getAs<StringAttr>(
               *output_placement_attr_key_) == cpu_placement_attr_))
        shape_calc_ops.insert(op);
    }
  }
}

void MarkShapeCalc::markShapeCalculationOps(
    FuncOp func, llvm::DenseSet<Operation*>& shape_calc_ops) {
  auto& block = func.getBlocks().front();
  for (Operation& op : block) {
    if (!isMhloDialect(&op)) continue;

    // If the op is already in shape calculation op set, insert all of its
    // operands into shape calculation op set
    if (shape_calc_ops.contains(&op)) {
      for (auto operand_value : op.getOperands()) {
        Operation* operand = operand_value.getDefiningOp();
        if (operand == nullptr || !isMhloDialect(operand)) continue;
        shape_calc_ops.insert(operand);
      }
    } else {
      // Mark operands into shape calculation set according to the lookup table.
      auto op_type_id = op.getRegisteredInfo()->getTypeID();
      auto iter = kShapeCalcOperandMap.find(op_type_id);
      if (iter != kShapeCalcOperandMap.end()) {
        for (auto operand_idx : iter->second) {
          auto operand = op.getOperand(operand_idx).getDefiningOp();
          if (operand == nullptr || !isMhloDialect(operand)) continue;
          shape_calc_ops.insert(operand);
        }
      }
    }
    // TODO(disc): If the operand of the op is a nested FuncOp, mark the
    // associated producer in the nested FuncOp
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMarkShapeCalcOpPass() {
  return std::make_unique<MarkShapeCalc>();
}

}  // namespace mhlo
}  // namespace mlir
