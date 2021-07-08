#include <map>
#include <set>
#include <string>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/IR/MLIRContext.h"              // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:local_config_mlir
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_disc_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"

namespace mlir {

using hlo::kInputPlacementAttr;
using hlo::kPlaceTyAttr;
using hlo::kTypeDevice;
using hlo::kTypeHost;
using hlo::PlacementType;

namespace mhlo {
namespace {

bool isTargetDialect(Operation* op) {
  return (op->getDialect() ==
          op->getContext()->getLoadedDialect<mhlo::MhloDialect>());
}

// This pass explicitly place the shape calculating hlo_ops on host side
// by adding an Attr. Nested FuncOps should be taken into consideration.
// 1, Normally, the type of kPlaceTyAttr is StringAttr;
// 2, In case the result type of an hlo op is TupleType, for example TupleOp
//    or TopKOp, the type of kPlaceTyAttr is an ArrayAttr made of
//    StringAttr.
struct PlaceShapeCalcOnHost
    : public PassWrapper<PlaceShapeCalcOnHost, OperationPass<ModuleOp>> {
 public:
  PlaceShapeCalcOnHost() = default;
  PlaceShapeCalcOnHost(const PlaceShapeCalcOnHost& o){};

  StringRef getArgument() const final { return "place-shape-calc-on-host"; }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mhlo::MhloDialect, mhlo_disc::MhloDiscDialect>();
  }

  void runOnOperation() override {
    // Phase 1: Place the shape calculation subgraph to Host
    placeShapeCalcSubgraphOnHost();
    // Phase 2: Place any mhlo ops that calculates I32 on Host
    placeI32OpsOnHost();

    // TODO(disc): Phase 3: Add placement attribute for TupleOp and
    // GetTupleElementOp.

    // Phase 4: Insert h2d and d2h OP on cross device edges. Host is CPU. Device
    // is GPU or CPU
    addMemcpyNodes();
  };

 private:
  // Place the shape calculation subgraph to Host
  void placeShapeCalcSubgraphOnHost();
  // Place any mhlo ops that calculates I32 on Host
  void placeI32OpsOnHost();
  // Insert h2d and d2h OP on cross device edges.
  void addMemcpyNodes();

  // for rule based placement strategy, the placement of the op in the list
  // is up to the placement of the dominant operand
  const std::unordered_map<std::string, /*dominant operand index*/ int>
      kPlaceRuleMap = {{"mhlo.dynamic_gather", /*operand*/ 0},
                       {"mhlo.gather", /*operand*/ 0}};

  const std::unordered_map<std::string, std::set<int>> kShapeCalcOperandMap = {
      {"mhlo.real_dynamic_slice",
       {/*start_indices*/ 1, /*limit_indices*/ 2, /*strides*/ 3}},
      {"mhlo.dynamic_pad",
       {/*edge_padding_low*/ 2, /*edge_padding_high*/ 3,
        /*interior_padding*/ 4}},
      {"mhlo.dynamic_reshape", {/*shape*/ 1}},
      {"mhlo.dynamic_iota", {/*shape*/ 0}},
      {"mhlo.dynamic_broadcast_in_dim", {/*out_dim_size*/ 1}},
      {"mhlo.dynamic_gather", {/*slice_sizes*/ 2}},
      {"mhlo.dynamic_conv", {/*paddings*/ 2}},
      {"mhlo.if", {/*pred*/ 0}},
      {"mhlo.dynamic_rng_uniform", {/*start*/ 0, /*limit*/ 1, /*shape*/ 2}}};

  // add output OP into marked set if it is a I64 scalar and placment is CPU.
  void markI64ReturnedCPUScalarOps(FuncOp func,
                                   DenseSet<Operation*>& marked_ops);
  // Update marked set.
  // If a OP is in marked set, add all of its operands to marked set.
  // Add some operands of dynamic shape OPs into marked set according to lookup
  // table.
  void markShapeCalculationOps(FuncOp func, DenseSet<Operation*>& marked_ops);

  // Get placement vector of func's output.
  SmallVector<llvm::StringRef, 4> getOutputPlacements(FuncOp main_func);

  // Get Op's placement according to its attr
  PlacementType getOpPlacement(Operation* op);

  // Get tensor's placement
  PlacementType getTensorPlacement(Operation* dst, size_t operand_idx);

  // Get input argument's placment.
  PlacementType getArgumentPlacement(Value arg);

  // Enforce output's placement.
  void enforceOutputPlacement(
      Operation* dst, FuncOp main_func,
      SmallVector<std::pair<Operation*, size_t>, 4>& d2h_worklist,
      SmallVector<std::pair<Operation*, size_t>, 4>& h2d_worklist);

  // insert H2D or D2H Op.
  void insertMemcpy(Operation* dst, size_t operand_index, bool is_h2d);
};

// Place the subgraph that is the producer of any shape operands
// on Host
// TODO(disc): handle when TupleOp exists in shape_calc_ops
void PlaceShapeCalcOnHost::placeShapeCalcSubgraphOnHost() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());
  llvm::DenseSet<Operation*> shape_calc_ops;

  for (FuncOp func : module.getOps<FuncOp>()) {
    // Put the i64 Scalar output on Host(into shape_calc_ops).
    // TODO(disc): revisit this if we have outputs on CPU for TF in the future.
    if (func.getName() == "main") {
      markI64ReturnedCPUScalarOps(func, shape_calc_ops);
    }
    // Skip if this function is external
    if (func.isExternal()) continue;
    // no target ops
    if (llvm::none_of(func.getBlocks().front(),
                      [](Operation& op) { return isTargetDialect(&op); })) {
      continue;
    }
    markShapeCalculationOps(func, shape_calc_ops);
  }

  for (Operation* op : shape_calc_ops) {
    // We suppose that mhlo op only has single output, either having tensor
    // type or tuple type.
    if (auto tp = op->getResult(0).getType().dyn_cast<TupleType>()) {
      // If an op is placed on host, then we suppose all its outputs are
      // placed on host.
      SmallVector<Attribute, 4> attrs(tp.size(),
                                      builder.getStringAttr(kTypeHost));
      op->setAttr(kPlaceTyAttr, ArrayAttr::get(tp.getContext(), attrs));
    } else {
      op->setAttr(kPlaceTyAttr, builder.getStringAttr(kTypeHost));
    }
  }
}

// Place any mhlo ops that calculates i32 on Host
// this is an rule based optimization that mimicking the behavior of
// tensorflow
void PlaceShapeCalcOnHost::placeI32OpsOnHost() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  for (Operation& op : module.getOps()) {
    if (!isTargetDialect(&op)) {
      return;
    }
    if (isa<mhlo::TupleOp, mhlo::GetTupleElementOp, mhlo::WhileOp, mhlo::IfOp,
            mhlo::ReturnOp>(&op)) {
      return;
    }
    // skip the ops that is already placed on Host
    auto attr = op.getAttrOfType<StringAttr>(kPlaceTyAttr);
    if ((attr != nullptr) && (attr.getValue() == kTypeHost)) return;

    if (isa<mhlo::GetDimensionSizeOp, tensor::FromElementsOp>(&op)) {
      op.setAttr(kPlaceTyAttr, builder.getStringAttr(kTypeHost));
      return;
    }

    // ops that only cares about the output element type
    if (isa<mhlo::ConstOp, mhlo::SelectOp, mhlo::IotaOp, mhlo::DynamicIotaOp>(
            &op)) {
      auto result_ty = op.getResult(0).getType().dyn_cast<RankedTensorType>();
      assert(result_ty && "unexpected non ranked type for ConstOp");
      auto elem_type = result_ty.getElementType();
      if (elem_type.isInteger(32)) {
        op.setAttr(kPlaceTyAttr, builder.getStringAttr(kTypeHost));
      }
      return;
    }

    std::string op_name = op.getName().getStringRef().str();
    bool place_on_host = false;
    // follow the rule of kPlaceRuleMap exist, or else follow
    // kShapeCalcOperandMap
    if (kPlaceRuleMap.find(op_name) != kPlaceRuleMap.end()) {
      auto dominant_idx = kPlaceRuleMap.at(op_name);
      auto operand_ty =
          op.getOperand(dominant_idx).getType().dyn_cast<RankedTensorType>();
      assert(operand_ty && "unexpected non unranked type of operand");
      if (operand_ty.getElementType().isInteger(32)) {
        place_on_host = true;
      }
    } else {
      std::set<int> shape_operand_indices;
      if (kShapeCalcOperandMap.find(op_name) != kShapeCalcOperandMap.end()) {
        shape_operand_indices = kShapeCalcOperandMap.at(op_name);
      }
      for (int idx = 0; idx < op.getNumOperands(); ++idx) {
        // if it is not "shape operand", then it must be "data operand"
        if (shape_operand_indices.find(idx) == shape_operand_indices.end()) {
          auto operand_ty =
              op.getOperand(idx).getType().dyn_cast<RankedTensorType>();
          if (!operand_ty) continue;
          auto elem_type = operand_ty.getElementType();
          if (elem_type.isInteger(32)) {
            place_on_host = true;
            break;
          }
        }
      }
    }

    if (!place_on_host) {
      // For most ops, we can safely omit to insert placement attribute since
      // currently we suppose ops without placement attribute are placed on
      // device. However, ops having tuple outputs should have explicit
      // placement attributes.
      if (auto tp = op.getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(),
                                        builder.getStringAttr(kTypeDevice));
        op.setAttr(kPlaceTyAttr, ArrayAttr::get(tp.getContext(), attrs));
      }
    } else {
      if (auto tp = op.getResult(0).getType().dyn_cast<TupleType>()) {
        SmallVector<Attribute, 4> attrs(tp.size(),
                                        builder.getStringAttr(kTypeHost));
        op.setAttr(kPlaceTyAttr, ArrayAttr::get(tp.getContext(), attrs));
      } else {
        op.setAttr(kPlaceTyAttr, builder.getStringAttr(kTypeHost));
      }
    }
    return;
  };
}

void PlaceShapeCalcOnHost::enforceOutputPlacement(
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
        operand_op ? getOpPlacement(operand_op) : getArgumentPlacement(operand);
    PlacementType dst_placement;
    if (output_placements[i] == kTypeHost) {
      dst_placement = PlacementType::kHost;
    } else {
      assert(output_placements[i] == kTypeDevice);
      dst_placement = PlacementType::kDevice;
    }
    if (dst_placement == PlacementType::kHost &&
        src_placement == PlacementType::kDevice) {
      d2h_worklist.push_back(std::make_pair(dst, i));
    } else if (dst_placement == PlacementType::kDevice &&
               src_placement == PlacementType::kHost) {
      h2d_worklist.push_back(std::make_pair(dst, i));
    }
  }
}

// Insert potential h2d and d2h for cross device edges
void PlaceShapeCalcOnHost::addMemcpyNodes() {
  ModuleOp module = getOperation();
  Builder builder(&getContext());

  SmallVector<std::pair<Operation*, size_t>, 4> d2h_worklist;
  SmallVector<std::pair<Operation*, size_t>, 4> h2d_worklist;

  for (Operation& dst : module.getOps()) {
    // Enforce output placement specified by the users using attrs.
    if (isa<mlir::ReturnOp>(&dst)) {
      auto parent = dst.getParentOp();
      if (!isa<mlir::FuncOp>(parent) ||
          (cast<mlir::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto main_func = dyn_cast<mlir::FuncOp>(parent);
      enforceOutputPlacement(&dst, main_func, d2h_worklist, h2d_worklist);
    }

    if (isa<tensor::ExtractOp>(&dst)) {
      auto operand = dst.getOperand(0);
      auto parent = operand.getParentRegion()->getParentOp();
      if (!isa<mlir::FuncOp>(parent) ||
          (cast<mlir::FuncOp>(parent).getName() != "main")) {
        return;
      }
      auto defining_op = operand.getDefiningOp();
      if (defining_op) return;
      if (getArgumentPlacement(operand) == PlacementType::kDevice) {
        d2h_worklist.push_back(std::make_pair(&dst, 0));
      }
    }
    if (!isTargetDialect(&dst) ||
        (isa<mhlo::GetTupleElementOp, mhlo::ReturnOp>(&dst)))
      return;

    // TODO(disc): output of the while's cond func should be placed on host.

    for (auto indexed_operand : llvm::enumerate(dst.getOperands())) {
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
      } else if ((operand_op->getDialect()->getNamespace() != "mhlo") ||
                 (isa<mhlo::TupleOp>(operand_op))) {
        continue;
      }

      auto dst_placement = getTensorPlacement(&dst, index);
      auto src_placement = operand_op ? getOpPlacement(operand_op)
                                      : getArgumentPlacement(operand);
      if (dst_placement == PlacementType::kHost &&
          src_placement == PlacementType::kDevice) {
        d2h_worklist.push_back(std::make_pair(&dst, index));
      } else if (dst_placement == PlacementType::kDevice &&
                 src_placement == PlacementType::kHost) {
        h2d_worklist.push_back(std::make_pair(&dst, index));
      }
    }
  };

  for (auto h2d : h2d_worklist) {
    insertMemcpy(h2d.first, h2d.second, 1);
  }
  for (auto d2h : d2h_worklist) {
    insertMemcpy(d2h.first, d2h.second, 0);
  }
}

PlacementType PlaceShapeCalcOnHost::getArgumentPlacement(Value arg) {
  auto parent = arg.getParentRegion()->getParentOp();
  assert(isa<mlir::FuncOp>(parent) && "invalid use of getArgumentPlacement");
  auto main_func = cast<mlir::FuncOp>(parent);
  assert(main_func.getName() == "main" &&
         "invalid use of getArgumentPlacement");
  auto dict_attr = parent->getAttrOfType<DictionaryAttr>("tf.entry_function");
  assert(dict_attr && "main_func must has tf.entry_function attr");
  auto input_placements_attr = dict_attr.get(kInputPlacementAttr);
  if (!input_placements_attr) return PlacementType::kDevice;

  SmallVector<StringRef, 4> input_placements;
  input_placements_attr.cast<mlir::StringAttr>().getValue().split(
      input_placements, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  assert(input_placements.size() == main_func.getNumArguments() &&
         "input_placements.size() is not equal to num of inputs");
  auto arg_index = hlo::getArgumentIndex(main_func, arg);
  if (input_placements[arg_index] == hlo::kTypeHost) {
    return PlacementType::kHost;
  } else {
    assert((input_placements[arg_index] == hlo::kTypeDevice) &&
           "invalid input_placements string");
    return PlacementType::kDevice;
  }
}

SmallVector<llvm::StringRef, 4> PlaceShapeCalcOnHost::getOutputPlacements(
    FuncOp main_func) {
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

void PlaceShapeCalcOnHost::markI64ReturnedCPUScalarOps(
    FuncOp func, llvm::DenseSet<Operation*>& marked_ops) {
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

    if (!isTargetDialect(op)) continue;

    if (auto type = op->getResult(0).getType().dyn_cast<RankedTensorType>()) {
      if ((output_placements[idx] == kTypeHost) &&
          type.getElementType().isInteger(64) && (type.getRank() == 0)) {
        marked_ops.insert(op);
      }
    }
  }
}

void PlaceShapeCalcOnHost::markShapeCalculationOps(
    FuncOp func, llvm::DenseSet<Operation*>& marked_ops) {
  auto& block = func.getBlocks().front();
  for (Operation& op : block) {
    if (!isTargetDialect(&op)) return;
    if (op.getParentOp() != func.getOperation()) return;

    // 1. If the op is already marked, mark all of its operands
    //    as shape calculation ops
    if (marked_ops.contains(&op)) {
      for (auto operand_value : op.getOperands()) {
        Operation* operand = operand_value.getDefiningOp();
        if (operand == nullptr) continue;
        if (!isTargetDialect(operand)) {
          continue;
        }
        marked_ops.insert(operand);
      }
    }
    // 2. If the op is not marked, mark the shape operands as
    //    shape calculation ops
    if (!marked_ops.contains(&op)) {
      std::string name_str = op.getName().getStringRef().str();
      if (kShapeCalcOperandMap.find(name_str) != kShapeCalcOperandMap.end()) {
        for (auto operand_idx : kShapeCalcOperandMap.at(name_str)) {
          auto operand = op.getOperand(operand_idx).getDefiningOp();
          if (operand == nullptr) continue;
          if (!isTargetDialect(operand)) {
            continue;
          }
          marked_ops.insert(operand);
        }
      }
    }
    // TODO(disc): 3. If the operand of the op is a nested FuncOp, mark the
    //    associated producer in the nested FuncOp
  };
}

PlacementType PlaceShapeCalcOnHost::getOpPlacement(Operation* op) {
  auto attr = op->getAttrOfType<StringAttr>(hlo::kPlaceTyAttr);
  if ((attr != nullptr) && (attr.getValue() == kTypeHost)) {
    return PlacementType::kHost;
  }
  return PlacementType::kDevice;
}

PlacementType PlaceShapeCalcOnHost::getTensorPlacement(Operation* dst,
                                                       size_t operand_idx) {
  // special case when dst is TupleOp
  if (isa<mhlo::TupleOp>(dst)) {
    auto array_attr = dst->getAttrOfType<ArrayAttr>(kPlaceTyAttr);
    assert(array_attr && "kPlaceTyAttr on Tuple not found");
    if (array_attr[operand_idx].cast<StringAttr>().getValue() == kTypeHost) {
      return PlacementType::kHost;
    } else {
      return PlacementType::kDevice;
    }
  }

  // when dst op placed on Host(CPU)
  if (getOpPlacement(dst) == PlacementType::kHost) return PlacementType::kHost;

  // when dst op placed on Device
  std::string name_str = dst->getName().getStringRef().str();
  if (kShapeCalcOperandMap.find(name_str) == kShapeCalcOperandMap.end())
    return PlacementType::kDevice;

  const auto& shape_operand_indices = kShapeCalcOperandMap.at(name_str);
  if (shape_operand_indices.find(operand_idx) != shape_operand_indices.end())
    return PlacementType::kHost;

  return PlacementType::kDevice;
}

void PlaceShapeCalcOnHost::insertMemcpy(Operation* dst, size_t operand_index,
                                        bool is_h2d) {
  OpBuilder b(dst);
  Location loc = dst->getLoc();
  auto orig_operand = dst->getOperand(operand_index);
  Value copy_result = nullptr;
  if (is_h2d) {
    copy_result = b.create<mhlo_disc::H2DOp>(loc, orig_operand).getResult();
  } else {
    auto new_copy = b.create<mhlo_disc::D2HOp>(loc, orig_operand);
    new_copy->setAttr(kPlaceTyAttr, b.getStringAttr(kTypeHost));
    copy_result = new_copy.getResult();
  }
  dst->setOperand(operand_index, copy_result);
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createPlaceShapeCalcOnHostPass() {
  return absl::make_unique<PlaceShapeCalcOnHost>();
}

}  // namespace mhlo
}  // namespace mlir
