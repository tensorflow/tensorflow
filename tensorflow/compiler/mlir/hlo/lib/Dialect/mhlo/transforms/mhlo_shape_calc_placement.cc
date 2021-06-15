
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/IR/MLIRContext.h"              // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:local_config_mlir
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"

using llvm::StringRef;
using std::string;

namespace mlir {

using hlo::DeviceType;
using hlo::getInputPlacement;
using hlo::kDeviceTyAttr;
using hlo::kPlaceRuleMap;
using hlo::kShapeCalcOperandMap;
using hlo::kTypeCPU;
using hlo::kTypeGPU;

namespace mhlo {
namespace {

SmallVector<llvm::StringRef, 4> getOutputPlacements(FuncOp main_func) {
  auto dict_attr =
      main_func->getAttrOfType<DictionaryAttr>("tf.entry_function");
  assert(dict_attr && "main_func must has tf.entry_function attr");
  auto output_placements_attr = dict_attr.get(hlo::kOutputPlacementAttr);
  SmallVector<StringRef, 4> output_placements;
  if (!output_placements_attr) {
    // No placement attr is specified, thus using the inferred placement.
    return output_placements;
  }

  output_placements_attr.cast<mlir::StringAttr>().getValue().split(
      output_placements, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  return output_placements;
}

void markI64ReturnedCPUScalarOps(FuncOp func,
                                 std::map<Operation*, bool>& marked_ops) {
  assert(func.getName() == "main");
  auto return_op = func.front().getTerminator();
  if (!isa<mlir::ReturnOp>(return_op)) return;

  const auto& output_placements = getOutputPlacements(func);
  auto returned_ops = return_op->getOperands();
  assert(returned_ops.size() == output_placements.size());
  for (auto output : llvm::enumerate(returned_ops)) {
    auto idx = output.index();
    auto op = output.value().getDefiningOp();
    if (!op) continue;

    auto dialect_name = op->getDialect()->getNamespace();
    if (dialect_name != "mhlo" && dialect_name != "tensor") continue;

    if (auto type = op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
      if ((output_placements[idx] == kTypeCPU) &&
          type.getElementType().isInteger(64) && (type.getRank() == 0)) {
        marked_ops[op] = true;
      }
    }
  }
}

void markShapeCalculationOps(FuncOp func,
                             std::map<Operation*, bool>& marked_ops) {
  auto& block = func.getBlocks().front();
  block.walk([&](Operation* op) {
    auto dialect_name = op->getDialect()->getNamespace();
    if (dialect_name != "mhlo" && dialect_name != "tensor") return;
    if (op->getParentOp() != func.getOperation()) return;

    // 1. If the op is already marked, mark all of its operands
    //    as shape calculation ops
    if ((marked_ops.find(op) != marked_ops.end()) && marked_ops[op]) {
      for (auto operand_value : op->getOperands()) {
        Operation* operand = operand_value.getDefiningOp();
        if (operand == nullptr) continue;
        auto operand_dialect_name = operand->getDialect()->getNamespace();
        if (operand_dialect_name != "mhlo" &&
            operand_dialect_name != "tensor") {
          continue;
        }
        marked_ops[operand] = true;
      }
    }
    // 2. If the op is not marked, mark the shape operands as
    //    shape calculation ops
    if (((marked_ops.find(op) != marked_ops.end()) && (!marked_ops[op])) ||
        (marked_ops.find(op) == marked_ops.end())) {
      string name_str = op->getName().getStringRef().str();
      if (kShapeCalcOperandMap.find(name_str) != kShapeCalcOperandMap.end()) {
        for (auto operand_idx : kShapeCalcOperandMap.at(name_str)) {
          auto operand = op->getOperand(operand_idx).getDefiningOp();
          if (operand == nullptr) continue;
          auto operand_dialect_name = operand->getDialect()->getNamespace();
          if (operand_dialect_name != "mhlo" &&
              operand_dialect_name != "tensor") {
            continue;
          }
          marked_ops[operand] = true;
        }
      }
    }
    // TODO: 3. If the operand of the op is a nested FuncOp, mark the
    //    associated producer in the nested FuncOp
  });
}

DeviceType getOpPlacement(Operation* op) {
  auto attr = op->getAttrOfType<StringAttr>(hlo::kDeviceTyAttr);
  if ((attr != nullptr) && (attr.getValue() == kTypeCPU)) {
    return DeviceType::kCPU;
  }
  return DeviceType::kGPU;
}

DeviceType getTensorPlacement(Operation* dst, size_t operand_idx) {
  // special case when dst is TupleOp
  if (isa<mhlo::TupleOp>(dst)) {
    auto array_attr = dst->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
    assert(array_attr && "kDeviceTyAttr on Tuple not found");
    if (array_attr[operand_idx].cast<StringAttr>().getValue() == kTypeCPU) {
      return DeviceType::kCPU;
    } else {
      return DeviceType::kGPU;
    }
  }
  // when dst op placed on CPU
  if (getOpPlacement(dst) == DeviceType::kCPU) return DeviceType::kCPU;

  // when dst op placed on GPU
  string name_str = dst->getName().getStringRef().str();
  if (kShapeCalcOperandMap.find(name_str) == kShapeCalcOperandMap.end())
    return DeviceType::kGPU;

  const auto& shape_operand_indices = kShapeCalcOperandMap.at(name_str);
  if (shape_operand_indices.find(operand_idx) != shape_operand_indices.end())
    return DeviceType::kCPU;

  return DeviceType::kGPU;
}

// TODO: Currently, we put H2DOp and D2HOp in mhlo dialect. They should be put
// into a separate dedicated dialect.
void insertMemcpy(Operation* dst, size_t operand_index, bool is_h2d) {
  OpBuilder b(dst);
  Location loc = dst->getLoc();
  auto orig_operand = dst->getOperand(operand_index);
  Value copy_result = nullptr;
  if (is_h2d) {
    copy_result = b.create<mhlo::H2DOp>(loc, orig_operand).getResult();
  } else {
    auto new_copy = b.create<mhlo::D2HOp>(loc, orig_operand);
    new_copy->setAttr(kDeviceTyAttr, b.getStringAttr(kTypeCPU));
    copy_result = new_copy.getResult();
  }
  dst->setOperand(operand_index, copy_result);
}

LogicalResult setAttrForTupleOp(mhlo::TupleOp tuple) {
  auto tuple_ty = tuple.getResult().getType().dyn_cast<TupleType>();
  auto tuple_size = tuple_ty.size();
  SmallVector<Attribute, 4> attrs;
  OpBuilder builder(tuple);
  auto parent = tuple->getParentOp();
  for (size_t operand_idx = 0; operand_idx < tuple_size; ++operand_idx) {
    auto operand = tuple.getOperation()->getOperand(operand_idx);
    auto operand_op = operand.getDefiningOp();
    // in case the operand is a block argument
    if (!operand_op) {
      if (isa<mlir::FuncOp>(parent)) {
        if (getInputPlacement(operand) == DeviceType::kCPU) {
          attrs.push_back(builder.getStringAttr(kTypeCPU));
        } else {
          attrs.push_back(builder.getStringAttr(kTypeGPU));
        }
      } else if (isa<mhlo::WhileOp>(parent)) {
        auto while_operand = parent->getOperand(0).getDefiningOp();
        assert(isa<mhlo::TupleOp>(while_operand));
        auto parent_array_attr =
            while_operand->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
        assert(parent_array_attr &&
               "tuple in the parent block should already be processed");
        attrs.push_back(parent_array_attr[operand_idx].cast<StringAttr>());
      } else if (isa<mhlo::IfOp>(parent)) {
        // if in true_region, map to operand(1) of IfOp
        // if in false_region, map to operand(2) of IfOp
        auto cond = cast<mhlo::IfOp>(parent);
        Operation* def_op = (tuple->getParentRegion() == &cond.true_branch())
                                ? cond.true_arg().getDefiningOp()
                                : cond.false_arg().getDefiningOp();
        DeviceType operand_placement = getOpPlacement(def_op);
        if (operand_placement == DeviceType::kGPU) {
          attrs.push_back(builder.getStringAttr(kTypeGPU));
        } else if (operand_placement == DeviceType::kCPU) {
          attrs.push_back(builder.getStringAttr(kTypeCPU));
        } else {
          tuple.emitError("Unexpected DeviceType");
          return failure();
        }
      } else {
        tuple.emitError("!nexpected tuple");
        return failure();
      }

      continue;
    }

    // in case the tuple_operand is a ordinary op
    DeviceType operand_placement = getOpPlacement(operand_op);
    if (operand_placement == DeviceType::kGPU) {
      attrs.push_back(builder.getStringAttr(kTypeGPU));
    } else if (operand_placement == DeviceType::kCPU) {
      attrs.push_back(builder.getStringAttr(kTypeCPU));
    } else {
      tuple.emitError("Unexpected DeviceType");
      return failure();
    }
  }

  auto array_attr = ArrayAttr::get(tuple.getContext(), attrs);
  tuple.getOperation()->setAttr(kDeviceTyAttr, array_attr);

  return success();
}

LogicalResult setAttrForGTEOp(mhlo::GetTupleElementOp gte) {
  OpBuilder builder(gte);
  auto gte_idx = static_cast<unsigned>(gte.index());
  auto tuple = gte.getOperation()->getOperand(0).getDefiningOp();
  // in case the operand is a block argument
  if (!tuple) {
    auto parent = gte->getParentOp();
    if (isa<mhlo::WhileOp>(parent)) {
      auto while_operand = parent->getOperand(0).getDefiningOp();
      assert(isa<mhlo::TupleOp>(while_operand));
      auto parent_array_attr =
          while_operand->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
      assert(parent_array_attr &&
             "tuple in the parent block should already be processed");
      gte.getOperation()->setAttr(
          kDeviceTyAttr, parent_array_attr[gte_idx].cast<StringAttr>());
    } else if (isa<mhlo::IfOp>(parent)) {
      auto if_op = cast<mhlo::IfOp>(parent);
      Operation* if_operand = (gte->getParentRegion() == &if_op.true_branch())
                                  ? if_op.true_arg().getDefiningOp()
                                  : if_op.false_arg().getDefiningOp();
      assert(isa<mhlo::TupleOp>(if_operand));
      auto parent_array_attr =
          if_operand->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
      assert(parent_array_attr &&
             "tuple in the parent block should already be processed");
      gte.getOperation()->setAttr(
          kDeviceTyAttr, parent_array_attr[gte_idx].cast<StringAttr>());
    } else {
      gte.emitError(
          "unexpected GTE whose operand is a block argument, \
                    but not inside a while/if");
      return failure();
    }
    return success();
  }
  auto array_attr = tuple->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
  assert(array_attr && "kDeviceTyAttr on Tuple not found");
  if (array_attr[gte_idx].cast<StringAttr>().getValue() == kTypeCPU) {
    gte.getOperation()->setAttr(kDeviceTyAttr, builder.getStringAttr(kTypeCPU));
  }
  return success();
}

// This pass explicitly place the shape calculating hlo_ops on host side
// by adding an Attr. Nested FuncOps should be taken into consideration.
// 1, Normally, the type of kDeviceTyAttr is StringAttr;
// 2, In case the result type of an hlo op is TupleType, for example TupleOp
//    or TopKOp, the type of kDeviceTyAttr is an ArrayAttr made of
//    StringAttr.
struct PlaceShapeCalcOnCPU
    : public PassWrapper<PlaceShapeCalcOnCPU, OperationPass<ModuleOp>> {
 public:
  PlaceShapeCalcOnCPU() = default;
  PlaceShapeCalcOnCPU(const PlaceShapeCalcOnCPU& o) {}

  void runOnOperation() override {
    // Phase 1: Place the shape calculation subgraph to CPU
    placeShapeCalcSubgraphOnHost();
    // Phase 2: Place any mhlo ops that calculates I32 on CPU
    placeI32OpsOnHost();
    // Phase 3: Add placement attribute for TupleOp and GetTupleElementOp
    addAttrForTupleAndGTE();
    // Phase 4: Insert h2d and d2h OP on cross device edges
    insertMemcpyNodes();
  }

 private:
  void placeShapeCalcSubgraphOnHost();
  void placeI32OpsOnHost();
  void addAttrForTupleAndGTE();
  void insertMemcpyNodes();
};

// Place the subgraph that is the producer of any shape operands
// on CPU
// TODO: handle when TupleOp exists in shape_calc_ops
void PlaceShapeCalcOnCPU::placeShapeCalcSubgraphOnHost() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  std::map<Operation*, bool> shape_calc_ops;

  // Put the i64 Scalar output on CPU into shape_calc_ops,
  // This is a optimization for PyTorch,
  // TODO: revisit this if we have outputs on CPU for TF in the future.
  module.walk([&](FuncOp func) {
    if (func.getName() == "main") {
      markI64ReturnedCPUScalarOps(func, shape_calc_ops);
    }
  });

  module.walk([&](FuncOp func) {
    // skip empty blocks
    if (func.getBlocks().size() == 0) return;

    // no target ops
    if (llvm::none_of(func.getBlocks().front(), [](Operation& op) {
          auto dialect_name = op.getDialect()->getNamespace();
          return (op.getDialect()->getNamespace() == "mhlo");
        })) {
      return;
    }

    markShapeCalculationOps(func, shape_calc_ops);
  });

  for (auto op : shape_calc_ops) {
    if (op.second) {
      // We suppose that mhlo op only has single output, either having tensor
      // type or tuple type.
      if (auto tp = op.first->getResult(0).getType().dyn_cast<TupleType>()) {
        // If an op is placed on cpu, then we suppose all its outputs are
        // placed on cpu.
        SmallVector<Attribute, 4> attrs(tp.size(),
                                        builder.getStringAttr(kTypeCPU));
        op.first->setAttr(kDeviceTyAttr,
                          ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op.first->setAttr(kDeviceTyAttr, builder.getStringAttr(kTypeCPU));
      }
    }
  }
}

// Place any mhlo ops that calculates i32 on CPU
// this is an rule based optimization that mimicking the behavior of tensorflow
void PlaceShapeCalcOnCPU::placeI32OpsOnHost() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  module.walk([&](Operation* op) {
    auto dialect_name = op->getDialect()->getNamespace();
    if (dialect_name != "mhlo" && dialect_name != "tensor") {
      return;
    }
    if (isa<mhlo::TupleOp>(op) || isa<mhlo::GetTupleElementOp>(op) ||
        isa<mhlo::WhileOp>(op) || isa<mhlo::IfOp>(op) ||
        isa<mhlo::ReturnOp>(op)) {
      return;
    }
    // skip the ops that is already placed on CPU
    auto attr = op->getAttrOfType<StringAttr>(kDeviceTyAttr);
    if ((attr != nullptr) && (attr.getValue() == kTypeCPU)) return;

    if (isa<mhlo::GetDimensionSizeOp>(op) || isa<tensor::FromElementsOp>(op)) {
      op->setAttr(kDeviceTyAttr, builder.getStringAttr(kTypeCPU));
      return;
    }

    // ops that only cares about the output element type
    if (isa<mhlo::ConstOp>(op) || isa<mhlo::SelectOp>(op) ||
        isa<mhlo::IotaOp>(op) || isa<mhlo::DynamicIotaOp>(op)) {
      auto result_ty = op->getResult(0).getType().dyn_cast<RankedTensorType>();
      assert(result_ty && "unexpected non ranked type for ConstOp");
      auto elem_type = result_ty.getElementType();
      if (elem_type.isInteger(32)) {
        op->setAttr(kDeviceTyAttr, builder.getStringAttr(kTypeCPU));
        return;
      } else {
        return;
      }
    }

    string op_name = op->getName().getStringRef().str();
    bool place_on_cpu = false;
    // follow the rule of kPlaceRuleMap exist, or else follow
    // kShapeCalcOperandMap
    if (kPlaceRuleMap.find(op_name) != kPlaceRuleMap.end()) {
      auto dominant_idx = kPlaceRuleMap.at(op_name);
      auto operand_ty =
          op->getOperand(dominant_idx).getType().dyn_cast<RankedTensorType>();
      assert(operand_ty && "unexpected non unranked type of operand");
      if (operand_ty.getElementType().isInteger(32)) {
        place_on_cpu = true;
      }
    } else {
      std::set<int> shape_operand_indices;
      if (kShapeCalcOperandMap.find(op_name) != kShapeCalcOperandMap.end()) {
        shape_operand_indices = kShapeCalcOperandMap.at(op_name);
      }
      for (int idx = 0; idx < op->getNumOperands(); ++idx) {
        // if it is not "shape operand", then it must be "data operand"
        if (shape_operand_indices.find(idx) == shape_operand_indices.end()) {
          auto operand_ty =
              op->getOperand(idx).getType().dyn_cast<RankedTensorType>();
          if (!operand_ty) continue;
          auto elem_type = operand_ty.getElementType();
          if (elem_type.isInteger(32)) {
            place_on_cpu = true;
            break;
          }
        }
      }
    }

    if (!place_on_cpu) {
      // For most ops, we can safely omit to insert placement attribute since
      // currently we suppose ops without placement attribute are placed on gpu.
      // However, ops having tuple outputs should have explicit placement
      // attributes.
      if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(),
                                        builder.getStringAttr(kTypeGPU));
        op->setAttr(kDeviceTyAttr, ArrayAttr::get(tp.getContext(), attrs));
      }
    } else {
      if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(),
                                        builder.getStringAttr(kTypeCPU));
        op->setAttr(kDeviceTyAttr, ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op->setAttr(kDeviceTyAttr, builder.getStringAttr(kTypeCPU));
      }
    }
    return;
  });
}

void setAttrForControlFlowOps(Operation* op, Region& region) {
  auto tuple = region.front().getTerminator()->getOperand(0).getDefiningOp();
  // TODO: support nested while/condition op
  assert(isa<mhlo::TupleOp>(tuple) &&
         "unexpected TupleOp in While/CondOp with TupleType");
  auto array_attr = tuple->getAttrOfType<ArrayAttr>(kDeviceTyAttr);
  assert(array_attr && "TupleOp in the body should already be processed");
  op->setAttr(kDeviceTyAttr, array_attr);
}

void processRegion(Region& region) {
  for (Block& block : region) {
    for (Operation& op : llvm::make_early_inc_range(block)) {
      if (isa<mhlo::TupleOp>(&op)) {
        setAttrForTupleOp(cast<mhlo::TupleOp>(&op));
      } else if (isa<mhlo::GetTupleElementOp>(&op)) {
        setAttrForGTEOp(cast<mhlo::GetTupleElementOp>(&op));
      } else if (isa<mhlo::WhileOp>(&op)) {
        auto while_op = cast<mhlo::WhileOp>(&op);
        processRegion(while_op.cond());
        processRegion(while_op.body());
        Value operand = while_op.getOperand();
        auto defining_op = operand.getDefiningOp();
        // Here we assume that while's operand always from a tupleOp.
        assert(defining_op && isa<mhlo::TupleOp>(defining_op));
        // We set placment attr of while_op to its operand's placement since
        // we require that all inputs and outputs of while_op should have the
        // same placement. This is further ensured in the `insertMemcpyNodes`
        // phase.
        while_op.getOperation()->setAttr(
            kDeviceTyAttr,
            defining_op->getAttrOfType<ArrayAttr>(kDeviceTyAttr));
      } else if (isa<mhlo::IfOp>(&op)) {
        auto if_op = cast<mhlo::IfOp>(&op);
        processRegion(if_op.true_branch());
        processRegion(if_op.false_branch());
        if (if_op.getType().isa<TupleType>()) {
          setAttrForControlFlowOps(&op, if_op.true_branch());
        }
      }
    }
  }
}

// Add placement attribute for TupleOp and GetTupleElementOp
void PlaceShapeCalcOnCPU::addAttrForTupleAndGTE() {
  ModuleOp module = getOperation();
  module.walk(
      [&](mlir::FuncOp func) { processRegion(*func.getCallableRegion()); });
}

void EnforceOutputPlacement(
    Operation* dst, FuncOp main_func,
    SmallVector<std::pair<Operation*, size_t>, 4>& d2h_worklist,
    SmallVector<std::pair<Operation*, size_t>, 4>& h2d_worklist) {
  const auto& output_placements = getOutputPlacements(main_func);
  assert(output_placements.size() == dst->getNumOperands() &&
         "output_placements.size() is not equal to num of outputs");
  for (auto i = 0; i < dst->getNumOperands(); ++i) {
    Value operand = dst->getOperand(i);
    auto operand_op = operand.getDefiningOp();
    auto src_placement =
        operand_op ? getOpPlacement(operand_op) : getInputPlacement(operand);
    DeviceType dst_placement;
    if (output_placements[i] == kTypeCPU) {
      dst_placement = DeviceType::kCPU;
    } else {
      assert(output_placements[i] == kTypeGPU);
      dst_placement = DeviceType::kGPU;
    }
    if (dst_placement == DeviceType::kCPU &&
        src_placement == DeviceType::kGPU) {
      d2h_worklist.push_back(std::make_pair(dst, i));
    } else if (dst_placement == DeviceType::kGPU &&
               src_placement == DeviceType::kCPU) {
      h2d_worklist.push_back(std::make_pair(dst, i));
    }
  }
}

// Insert potential h2d and d2h for cross device edges
void PlaceShapeCalcOnCPU::insertMemcpyNodes() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  SmallVector<std::pair<Operation*, size_t>, 4> d2h_worklist;
  SmallVector<std::pair<Operation*, size_t>, 4> h2d_worklist;
  module.walk([&](Operation* dst) {
    // Enforce output placement specified by the users using attrs.
    if (isa<mlir::ReturnOp>(dst)) {
      auto parent = dst->getParentOp();
      if (!isa<mlir::FuncOp>(parent) ||
          (cast<mlir::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto main_func = dyn_cast<mlir::FuncOp>(parent);
      EnforceOutputPlacement(dst, main_func, d2h_worklist, h2d_worklist);
    }

    if (isa<tensor::ExtractOp>(dst)) {
      auto operand = dst->getOperand(0);
      auto parent = operand.getParentRegion()->getParentOp();
      if (!isa<mlir::FuncOp>(parent) ||
          (cast<mlir::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto defining_op = operand.getDefiningOp();
      if (defining_op) return;
      if (getInputPlacement(operand) == DeviceType::kGPU) {
        d2h_worklist.push_back(std::make_pair(dst, 0));
      }
    }
    auto dialect_name = dst->getDialect()->getNamespace();
    if ((dialect_name != "mhlo" && dialect_name != "tensor") ||
        (isa<mhlo::GetTupleElementOp>(dst)) || (isa<mhlo::ReturnOp>(dst)))
      return;

    // output of the while's cond func should be placed on cpu.
    if (auto while_op = dyn_cast<mhlo::WhileOp>(dst)) {
      {
        assert(while_op.cond().getBlocks().size() == 1 &&
               "only support single block while_op");
        auto& front = while_op.cond().getBlocks().front();
        auto return_op = &*std::prev(front.end());
        assert(isa<mhlo::ReturnOp>(return_op) &&
               return_op->getNumOperands() == 1);
        Value operand = return_op->getOperand(0);
        auto defining_op = operand.getDefiningOp();

        // TODO: support while_op with single tensor input once we finish the
        // refactor the placement pass.
        assert(defining_op && "unsupported while_op with single tensor input");

        if (getOpPlacement(defining_op) == DeviceType::kGPU) {
          d2h_worklist.push_back(std::make_pair(return_op, 0));
        }
      }

      // outputs and inputs of the while's body func should have the same
      // placement.
      // TODO: this is a workaround and should be removed once we refactor the
      // pass.
      {
        auto ensure_input_output_having_same_output = [&]() {
          assert(while_op.body().getBlocks().size() == 1 &&
                 "only support single block while_op");
          auto& front = while_op.body().getBlocks().front();
          auto return_op = &*std::prev(front.end());
          assert(isa<mhlo::ReturnOp>(return_op) &&
                 return_op->getNumOperands() == 1);
          Value operand = return_op->getOperand(0);
          auto defining_op = operand.getDefiningOp();
          // io forwarding, thus already satifies the requirement.
          if (!defining_op) return;

          auto tp = operand.getType().dyn_cast<TupleType>();
          // TODO: support while_op with single tensor input once we finish
          // the refactor the placement pass.
          if (!tp) {
            return;
          }
          assert(dyn_cast<mhlo::TupleOp>(defining_op) &&
                 "unexpected nest while");
          auto output_attr =
              defining_op->getAttrOfType<ArrayAttr>(kDeviceTyAttr);

          auto while_operand = while_op.getOperand();
          auto while_operand_defining_op = while_operand.getDefiningOp();
          assert(while_operand_defining_op && "unexpected nest while");
          auto while_operand_tuple_op =
              dyn_cast<mhlo::TupleOp>(while_operand_defining_op);
          assert(while_operand_tuple_op && "unexpected nest while");
          auto input_attr = while_operand_defining_op->getAttrOfType<ArrayAttr>(
              kDeviceTyAttr);
          for (int i = 0; i < tp.size(); ++i) {
            auto input_placement = input_attr[i].cast<StringAttr>();
            auto output_placement = output_attr[i].cast<StringAttr>();
            if (input_placement.getValue() == output_placement.getValue()) {
              continue;
            }
            if (input_placement.getValue() == kTypeCPU) {
              d2h_worklist.push_back(std::make_pair(defining_op, i));
            } else {
              h2d_worklist.push_back(std::make_pair(defining_op, i));
            }
          }
          defining_op->setAttr(kDeviceTyAttr, input_attr);
        };
        ensure_input_output_having_same_output();
      }
    }

    for (auto indexed_operand : llvm::enumerate(dst->getOperands())) {
      auto index = indexed_operand.index();
      auto operand = indexed_operand.value();
      auto operand_op = operand.getDefiningOp();
      // If operand is a Block Argument and the parent is not the main func,
      // insert the potential memcpy outside the parent Op.

      if (!operand_op) {
        auto parent = operand.getParentRegion()->getParentOp();
        if (!isa<mlir::FuncOp>(parent) ||
            (cast<mlir::FuncOp>(parent).getName() != "main")) {
          continue;
        }
      } else if ((operand_op->getDialect()->getNamespace() != "mhlo" &&
                  operand_op->getDialect()->getNamespace() != "tensor") ||
                 (isa<mhlo::TupleOp>(operand_op))) {
        continue;
      }

      auto dst_placement = getTensorPlacement(dst, index);
      auto src_placement =
          operand_op ? getOpPlacement(operand_op) : getInputPlacement(operand);
      if (dst_placement == DeviceType::kCPU &&
          src_placement == DeviceType::kGPU) {
        d2h_worklist.push_back(std::make_pair(dst, index));
      } else if (dst_placement == DeviceType::kGPU &&
                 src_placement == DeviceType::kCPU) {
        h2d_worklist.push_back(std::make_pair(dst, index));
      }
    }
  });

  for (auto h2d : h2d_worklist) {
    insertMemcpy(h2d.first, h2d.second, /*is_h2d*/ 1);
  }
  for (auto d2h : d2h_worklist) {
    insertMemcpy(d2h.first, d2h.second, /*is_h2d*/ 0);
  }
}

}  // namespace
// createPlaceShapeCalcOnCPUPass
// MhloPlaceShapeCalc
std::unique_ptr<OperationPass<ModuleOp>> createPlaceShapeCalcOnCPUPass() {
  return absl::make_unique<PlaceShapeCalcOnCPU>();
}

}  // namespace mhlo
}  // namespace mlir
