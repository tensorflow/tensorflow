/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_manual_reduction_collectives.h"

#include <algorithm>
#include <cstdint>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/array.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

namespace sdy = ::mlir::sdy;
namespace stablehlo = ::mlir::stablehlo;

using ::mlir::ArrayRef;
using ::mlir::ModuleOp;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::SmallVector;
using ::mlir::StringRef;

using ::mlir::sdy::AxisRefAttr;

// Returns the channel ID after the maximum channel ID in the given `moduleOp`.
// TODO(b/419222666): remove dependency on `channel_handle` attribute name.
int64_t getNextChannelId(ModuleOp moduleOp) {
  int64_t maxChannelId = 0;
  moduleOp->walk([&](mlir::Operation* op) {
    if (auto channelHandle =
            op->getAttrOfType<stablehlo::ChannelHandleAttr>("channel_handle")) {
      maxChannelId = std::max(maxChannelId, channelHandle.getHandle());
    }
  });
  return maxChannelId + 1;
}

// Builds the replica groups for a `stablehlo::AllReduceOp`.
//
// For example, given:
//
// ```mlir
// sdy.mesh @mesh = <["x"=2, "y"=2]>
//
// sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x8xf32>
// ```
//
// Returns `[[0, 1], [2, 3]]`.
mlir::DenseIntElementsAttr getReplicaGroups(sdy::AllReduceOp op,
                                            mlir::OpBuilder& rewriter) {
  sdy::MeshAttr mesh = op.getOutSharding().getMesh(op);
  SmallVector<AxisRefAttr> meshAxisRefs =
      getOrderedAxisRefs(op.getReductionAxesAttr(), mesh);

  ArrayRef<AxisRefAttr> reductionAxes = op.getReductionAxes();
  int64_t groupSize = 1;
  llvm::SmallDenseMap<AxisRefAttr, int64_t> axisRefToReductionIndex;
  axisRefToReductionIndex.reserve(reductionAxes.size());
  for (auto [index, axis] : llvm::enumerate(reductionAxes)) {
    groupSize *= axis.getSize(mesh);
    axisRefToReductionIndex[axis] = index;
  }
  int64_t totalSize = mesh.getTotalSize();
  int64_t numGroups = totalSize / groupSize;

  SmallVector<int64_t> transposePerm(meshAxisRefs.size());
  SmallVector<int64_t> reshapeDims;
  reshapeDims.reserve(meshAxisRefs.size());

  int64_t nonReductionIndex = 0;
  int64_t nonReductionCount = meshAxisRefs.size() - reductionAxes.size();
  for (auto [meshIndex, axis] : llvm::enumerate(meshAxisRefs)) {
    reshapeDims.push_back(axis.getSize(mesh));
    auto reductionIndexIt = axisRefToReductionIndex.find(axis);
    if (reductionIndexIt == axisRefToReductionIndex.end()) {
      // Axis is not a reduction axis.
      transposePerm[nonReductionIndex++] = meshIndex;
    } else {
      transposePerm[nonReductionCount + reductionIndexIt->second] = meshIndex;
    }
  }

  // TODO(b/410040098): output V2 if possible, and maybe canonicalize.

  Array<int64_t> array(reshapeDims);
  if (mesh.getDeviceIds().empty()) {
    array.FillIota(0);
  } else {
    array.SetValues(mesh.getDeviceIds());
  }
  array.TransposeDimensions(transposePerm);
  array.Reshape({totalSize});
  auto replicaGroupsType = mlir::RankedTensorType::get(
      {numGroups, groupSize}, rewriter.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(replicaGroupsType,
                                         llvm::to_vector(array));
}

void convertAllReduce(sdy::AllReduceOp op, int64_t channelId,
                      mlir::IRRewriter& rewriter) {
  // TODO(tomnatan): Insert inside `sdy.manual_computation`.
  // TODO(tomnatan): Add a sharding to all-reduce if not fully manual.
  // TODO(tomnatan): Support sdy.reduce_scatter.
  // Channel type is DEVICE_TO_DEVICE.
  auto channelHandle = stablehlo::ChannelHandleAttr::get(
      op->getContext(), /*handle=*/channelId, /*type=*/1);
  // Setting `use_global_device_ids=true` as we are targeting the
  // `CollectiveOpGroupMode::kFlattenedID` mode.
  rewriter.setInsertionPoint(op);
  mlir::Type tensorType = op.getTensor().getType();
  auto newAllReduce = rewriter.replaceOpWithNewOp<stablehlo::AllReduceOp>(
      op, tensorType, op.getTensor(), getReplicaGroups(op, rewriter),
      channelHandle, /*use_global_device_ids=*/true);
  auto elementType = mlir::cast<mlir::ShapedType>(tensorType).getElementType();
  stablehlo::buildReduceBody<stablehlo::AddOp>(
      elementType, newAllReduce.getComputation(), rewriter);
}

class StablehloExportManualReductionCollectivesPass
    : public mlir::PassWrapper<StablehloExportManualReductionCollectivesPass,
                               OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      StablehloExportManualReductionCollectivesPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    mlir::IRRewriter rewriter(moduleOp.getContext());
    int64_t nextChannelId = getNextChannelId(moduleOp);
    moduleOp.walk([&](sdy::AllReduceOp allReduce) {
      // TODO(tomnatan): Only convert all-reduce ops that are marked by Shardy.
      convertAllReduce(allReduce, nextChannelId++, rewriter);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-export-manual-reduction-collectives";
  }

  StringRef getDescription() const override {
    return "Exports `sdy.all_reduce`, that originate from user defined "
           "shardings with unreduced axes, to `stablehlo.all_reduce` inside an "
           "`sdy.manual_computation`";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect, stablehlo::StablehloDialect>();
  }
};

}  // namespace

std::unique_ptr<Pass> createStablehloExportManualReductionCollectivesPass() {
  return std::make_unique<StablehloExportManualReductionCollectivesPass>();
}

void registerStablehloExportManualReductionCollectivesPass() {
  mlir::registerPass(createStablehloExportManualReductionCollectivesPass);
}

}  // namespace sdy
}  // namespace xla
