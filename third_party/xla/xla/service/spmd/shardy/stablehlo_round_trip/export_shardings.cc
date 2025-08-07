/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace stablehlo = ::mlir::stablehlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayRef;
using ::mlir::DictionaryAttr;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::success;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingAttr;

// Check if all shardings in an array are unreduced. Hard fail if at least one
// but not all are unreduced.
bool allShardingsUnreduced(ArrayRef<TensorShardingAttr> shardings) {
  bool hasUnreduced = false;
  bool hasNonUnreduced = false;
  for (TensorShardingAttr sharding : shardings) {
    if (sharding.getUnreducedAxes().empty()) {
      hasNonUnreduced = true;
    } else {
      hasUnreduced = true;
    }
  }
  CHECK(!hasUnreduced || !hasNonUnreduced)
      << "Shardings have a mix of unreduced and non-unreduced.";
  return hasUnreduced;
}

// Convert the shardings from kShardingAttr into kXlaShardingAttr.
LogicalResult exportFunc(FuncOp funcOp, const SymbolTable& symbolTable,
                         OpBuilder& builder,
                         bool addMissingShardingToControlFlow) {
  std::function<StringAttr(const HloSharding&)> getStringAttr =
      [&](const HloSharding& hloSharding) {
        return builder.getStringAttr(hloSharding.ToString());
      };
  std::function<MeshAttr(TensorShardingAttr)> getMeshAttr =
      [&](TensorShardingAttr sharding) {
        return sharding.getMesh(symbolTable);
      };

  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    if (auto sdySharding = funcOp.getArgAttrOfType<TensorShardingAttr>(
            argNum, kShardingAttr)) {
      ArrayRef<StringAttr> manualAxes;
      if (ManualAxesAttr manualAxesAttr =
              funcOp.getArgAttrOfType<ManualAxesAttr>(argNum, kManualAxes)) {
        manualAxes = manualAxesAttr.getValue();
        funcOp.removeArgAttr(argNum, kManualAxes);
      }
      funcOp.setArgAttr(argNum, kXlaShardingAttr,
                        getStringAttr(convertToHloSharding(
                            sdySharding, getMeshAttr, manualAxes)));
      funcOp.removeArgAttr(argNum, kShardingAttr);
    }
  }

  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    if (auto sdySharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
            resNum, kShardingAttr)) {
      ArrayRef<StringAttr> manualAxes;
      if (ManualAxesAttr manualAxesAttr =
              funcOp.getResultAttrOfType<ManualAxesAttr>(resNum, kManualAxes)) {
        manualAxes = manualAxesAttr.getValue();
        funcOp.removeResultAttr(resNum, kManualAxes);
      }
      funcOp.setResultAttr(resNum, kXlaShardingAttr,
                           getStringAttr(convertToHloSharding(
                               sdySharding, getMeshAttr, manualAxes)));
      funcOp.removeResultAttr(
          resNum, StringAttr::get(funcOp.getContext(), kShardingAttr));
    }
  }

  funcOp.front().walk([&](Operation* op) {
    ArrayRef<StringAttr> manualAxes;
    if (ManualAxesAttr manualAxesAttr =
            op->getAttrOfType<ManualAxesAttr>(kManualAxes)) {
      manualAxes = manualAxesAttr.getValue();
      op->removeAttr(kManualAxes);
    }

    if (ArrayRef<TensorShardingAttr> shardings = mlir::sdy::getShardings(op);
        !shardings.empty()) {
      if (allShardingsUnreduced(shardings)) {
        setFrontendAttribute(op, kHasUnreducedAxes,
                             builder.getStringAttr("true"));
      }
      setHloShardingAttr(op, shardings, getMeshAttr, manualAxes);
      op->removeAttr(kShardingAttr);
    } else if (addMissingShardingToControlFlow &&
               mlir::isa<stablehlo::WhileOp, stablehlo::CaseOp,
                         stablehlo::IfOp>(op)) {
      // The shard map export pass assigns shardings to any operation with
      // manual axes. Since this operation lacks shardings
      // (`shardings.empty()`), it cannot have manual axes. This CHECK asserts
      // this invariant before we add a replicated sharding.
      CHECK(manualAxes.empty());
      op->setAttr(kXlaShardingAttr, getStringAttr(HloSharding::Replicate()));
    }
  });

  return success();
}

class ExportStablehloShardingsPass
    : public PassWrapper<ExportStablehloShardingsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportStablehloShardingsPass)

  explicit ExportStablehloShardingsPass(bool addMissingShardingToControlFlow)
      : addMissingShardingToControlFlow(addMissingShardingToControlFlow) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    auto builder = OpBuilder::atBlockBegin(&moduleOp.getBodyRegion().front());

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (mlir::failed(exportFunc(funcOp, symbolTable, builder,
                                  addMissingShardingToControlFlow))) {
        signalPassFailure();
      }
    }

    moduleOp.walk([&](stablehlo::CustomCallOp customCall) {
      // StableHLO doesn't have an equivalent of `erf` and `topk` ops.
      // If they have a sharding annotation, we need to move it into
      // `mhlo.attributes`, which StableHLO->MHLO conversion would lift back up.
      StringRef callTargetName = customCall.getCallTargetName();
      if (callTargetName != "mhlo.erf" && callTargetName != "mhlo.topk") {
        return;
      }
      // TODO(bartchr): refactor `addFrontendAttribute` to take a key for the
      // dictionary attribute. Then can re-use the logic instead of duplicating
      // it here for `kMhloAttributesAttr`.
      if (auto sdySharding =
              customCall->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
        customCall->removeAttr(kXlaShardingAttr);
        SmallVector<mlir::NamedAttribute> newAttributes(
            customCall->getAttrOfType<DictionaryAttr>(kMhloAttributesAttr)
                .getValue());
        newAttributes.push_back(
            builder.getNamedAttr(kXlaShardingAttr, sdySharding));
        customCall->setAttr(kMhloAttributesAttr,
                            builder.getDictionaryAttr(newAttributes));
      }
    });
    // Remove all mesh symbols
    for (MeshOp meshOp :
         llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      symbolTable.erase(meshOp);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-export-shardings";
  }

  StringRef getDescription() const override {
    return "Converts the shardings from kShardingAttr to kXlaShardingAttr and "
           "removes mesh symbols.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }

 private:
  bool addMissingShardingToControlFlow;
};

HloSharding getHloShardingForOp(
    Operation* op, ArrayRef<TensorShardingAttr> shardings,
    std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
    ArrayRef<StringAttr> manualAxes) {
  // TODO(bartchr): pass through a symbol table to `getMesh(...)` below.
  bool isNoResultMaximal = op->getNumResults() == 0 && shardings.size() == 1 &&
                           (shardings.front().getMesh(op).isMaximal() ||
                            shardings.front().isFullyReplicated());
  CHECK(shardings.size() == op->getNumResults() || isNoResultMaximal);
  if (op->getNumResults() == 1 || isNoResultMaximal) {
    return convertToHloSharding(shardings.front(), getMeshAttr, manualAxes);
  }

  SmallVector<HloSharding> newShardings;
  llvm::transform(shardings, std::back_inserter(newShardings),
                  [&](TensorShardingAttr sdySharding) {
                    return convertToHloSharding(sdySharding, getMeshAttr,
                                                manualAxes);
                  });

  std::vector<xla::Shape> shapes;
  llvm::transform(op->getResultTypes(), std::back_inserter(shapes),
                  [&](mlir::Type type) { return xla::TypeToShape(type); });

  return HloSharding::Tuple(xla::ShapeUtil::MakeTupleShape(shapes),
                            newShardings);
}

}  // namespace

HloSharding convertToHloSharding(
    TensorShardingAttr sdySharding,
    std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
    ArrayRef<StringAttr> manualAxes) {
  MeshAttr mesh = getMeshAttr(sdySharding);

  // If there are no axes, convert to:
  // - maximal sharding if the mesh has a device id
  // - else replicated sharding
  if (mesh.getAxes().empty()) {
    return mesh.getDeviceIds().empty()
               ? HloSharding::Replicate()
               : HloSharding::AssignDevice(mesh.getDeviceIds().front());
  }

  SmallVector<int64_t> tileAssignmentDims(sdySharding.getRank(), 1);
  llvm::SmallDenseMap<AxisRefAttr, int64_t> axisRefToShardedPos;
  SmallVector<OpSharding::Type> types;
  int64_t shardedPos = 0;

  if (mesh.getAxes().size() == manualAxes.size()) {
    return HloSharding::Manual();
  }

  // Iterate the dim shardings.
  for (auto [index, dimSharding] :
       llvm::enumerate(sdySharding.getDimShardings())) {
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      tileAssignmentDims[index] *= axisRef.getSize(mesh);
      axisRefToShardedPos[axisRef] = shardedPos++;
    }
  }

  // Iterate the manual axes.
  if (!manualAxes.empty()) {
    types.push_back(OpSharding::MANUAL);
    int64_t& manualDim = tileAssignmentDims.emplace_back(1);
    mlir::MLIRContext* context = sdySharding.getContext();
    for (StringRef manualAxis : manualAxes) {
      manualDim *= mesh.getAxisSize(manualAxis);
      axisRefToShardedPos[AxisRefAttr::get(context, manualAxis)] = shardedPos++;
    }
  }

  // We will add all axes and let canonicalization merge adjacent axes.
  SmallVector<AxisRefAttr> meshAxisRefs = getOrderedAxisRefs(sdySharding, mesh);
  SmallVector<int64_t> reshapeDims(meshAxisRefs.size());
  SmallVector<int> transposePerm(meshAxisRefs.size());

  int64_t totalReplicatedSize = 1;
  int64_t replicatedPos = shardedPos;
  for (auto [axisIndex, axisRef] : llvm::enumerate(meshAxisRefs)) {
    reshapeDims[axisIndex] = axisRef.getSize(mesh);

    auto shardedPosIt = axisRefToShardedPos.find(axisRef);
    if (shardedPosIt == axisRefToShardedPos.end()) {
      // Axis is replicated
      transposePerm[replicatedPos++] = axisIndex;
      totalReplicatedSize *= axisRef.getSize(mesh);
    } else {
      // Axis is sharded or manual
      transposePerm[shardedPosIt->second] = axisIndex;
    }
  }

  if (totalReplicatedSize > 1) {
    tileAssignmentDims.push_back(totalReplicatedSize);
    types.push_back(OpSharding::REPLICATED);
  }

  // Handle arbitrary device ID list.
  if (!mesh.getDeviceIds().empty()) {
    Array<int64_t> deviceIdsArray(reshapeDims);
    deviceIdsArray.SetValues(mesh.getDeviceIds());
    deviceIdsArray.TransposeDimensions(transposePerm);
    deviceIdsArray.Reshape(tileAssignmentDims);
    return HloSharding::Subgroup(
        TileAssignment(
            std::make_shared<const Array<int64_t>>(std::move(deviceIdsArray))),
        types);
  }

  return HloSharding::Subgroup(
      xla::TileAssignment(tileAssignmentDims, reshapeDims, transposePerm),
      types);
}

void setHloShardingAttr(Operation* op, ArrayRef<TensorShardingAttr> shardings,
                        std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
                        ArrayRef<StringAttr> manualAxes) {
  HloSharding hloSharding =
      getHloShardingForOp(op, shardings, getMeshAttr, manualAxes);
  op->setAttr(kXlaShardingAttr,
              StringAttr::get(op->getContext(), hloSharding.ToString()));
}

std::unique_ptr<Pass> createExportStablehloShardingsPass(
    bool addMissingShardingToControlFlow) {
  return std::make_unique<ExportStablehloShardingsPass>(
      addMissingShardingToControlFlow);
}

void registerStablehloExportShardingsPass() {
  mlir::registerPass(std::bind(createExportStablehloShardingsPass, false));
}

}  // namespace sdy
}  // namespace xla
