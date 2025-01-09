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

#include "xla/service/spmd/shardy/mhlo_round_trip/export_shardings.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

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

using ::mlir::stablehlo::CustomCallOp;

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::SubAxisInfoAttr;
using ::mlir::sdy::TensorShardingAttr;

// Return all axes or sub-axes in the `mesh`, such that sub-axes are derived
// from `dimShardings` and sorted by their order in the mesh. For example, given
// mesh <"x"=2, "y"=16, "z"=4> and dimShardings [{"x"}, {"y":2(2)}], we would
// return ["x", "y":1(2), "y":2(2), "y":4(4), "z"].
SmallVector<AxisRefAttr> getOrderedAxisRefs(
    ArrayRef<DimensionShardingAttr> dimShardings, MeshAttr mesh) {
  // We use a map vector to maintain the order of mesh axes.
  llvm::MapVector<StringRef, SmallVector<int64_t>> axisNameToPreSizes;
  axisNameToPreSizes.reserve(mesh.getAxes().size());
  for (MeshAxisAttr meshAxis : mesh.getAxes()) {
    SmallVector<int64_t>& preSizes = axisNameToPreSizes[meshAxis.getName()];
    preSizes.push_back(1);
    preSizes.push_back(meshAxis.getSize());
  }

  for (const DimensionShardingAttr dimSharding : dimShardings) {
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      // Add sub-axis pre-sizes to `axisNameToPreSizes`. We'll dedup later.
      if (axisRef.getSubAxisInfo()) {
        SmallVector<int64_t>& preSizes = axisNameToPreSizes[axisRef.getName()];
        preSizes.push_back(axisRef.getSubAxisInfo().getPreSize());
        preSizes.push_back(axisRef.getSubAxisInfo().getNextPreSize());
      }
    }
  }

  SmallVector<AxisRefAttr> axisRefs;
  mlir::MLIRContext* ctx = mesh.getContext();
  for (auto& [axisName, preSizes] : axisNameToPreSizes) {
    if (preSizes.size() == 2) {
      // Full axis
      axisRefs.push_back(AxisRefAttr::get(ctx, axisName));
      continue;
    }
    absl::c_sort(preSizes);
    preSizes.erase(std::unique(preSizes.begin(), preSizes.end()),
                   preSizes.end());
    for (int64_t i = 0; i < preSizes.size() - 1; ++i) {
      int64_t preSize = preSizes[i];
      int64_t size = preSizes[i + 1] / preSize;
      axisRefs.push_back(AxisRefAttr::get(
          ctx, axisName, SubAxisInfoAttr::get(ctx, preSize, size)));
    }
  }

  return axisRefs;
}

// Convert the shardings from kShardingAttr into kXlaShardingAttr.
LogicalResult exportFunc(FuncOp funcOp, const SymbolTable& symbolTable,
                         OpBuilder& builder) {
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
      funcOp.setArgAttr(
          argNum, kXlaShardingAttr,
          getStringAttr(convertToHloSharding(sdySharding, getMeshAttr)));
      funcOp.removeArgAttr(argNum, kShardingAttr);
    }
  }

  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    if (auto sdySharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
            resNum, kShardingAttr)) {
      funcOp.setResultAttr(
          resNum, kXlaShardingAttr,
          getStringAttr(convertToHloSharding(sdySharding, getMeshAttr)));
      funcOp.removeResultAttr(
          resNum, StringAttr::get(funcOp.getContext(), kShardingAttr));
    }
  }

  funcOp.front().walk([&](Operation* op) {
    if (ArrayRef<TensorShardingAttr> shardings = mlir::sdy::getShardings(op);
        !shardings.empty()) {
      op->setAttr(
          kXlaShardingAttr,
          convertToHloShardingAttr(op, shardings, getMeshAttr, getStringAttr));
      op->removeAttr(kShardingAttr);
    }
  });

  return success();
}

class ExportMhloShardingsPass
    : public PassWrapper<ExportMhloShardingsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportMhloShardingsPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    auto builder = OpBuilder::atBlockBegin(&moduleOp.getBodyRegion().front());

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      if (mlir::failed(exportFunc(funcOp, symbolTable, builder))) {
        signalPassFailure();
      }
    }

    moduleOp.walk([&](CustomCallOp customCall) {
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
    return "xla-sdy-mhlo-export-shardings";
  }

  StringRef getDescription() const override {
    return "Converts the shardings from kShardingAttr to kXlaShardingAttr and "
           "removes mesh symbols.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }
};

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
  SmallVector<AxisRefAttr> meshAxisRefs =
      getOrderedAxisRefs(sdySharding.getDimShardings(), mesh);
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

StringAttr convertToHloShardingAttr(
    Operation* op, ArrayRef<TensorShardingAttr> shardings,
    std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
    std::function<StringAttr(const HloSharding&)> getStringAttr,
    ArrayRef<StringAttr> manualAxes) {
  assert(shardings.size() == op->getNumResults());
  if (op->getNumResults() == 1) {
    return getStringAttr(
        convertToHloSharding(shardings.front(), getMeshAttr, manualAxes));
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

  return getStringAttr(
      HloSharding::Tuple(xla::ShapeUtil::MakeTupleShape(shapes), newShardings));
}

std::unique_ptr<Pass> createExportMhloShardingsPass() {
  return std::make_unique<ExportMhloShardingsPass>();
}

void registerMhloExportShardingsPass() {
  mlir::registerPass(createExportMhloShardingsPass);
}

}  // namespace sdy
}  // namespace xla
