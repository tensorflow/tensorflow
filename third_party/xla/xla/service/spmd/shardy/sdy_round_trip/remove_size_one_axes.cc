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

#include "xla/service/spmd/shardy/sdy_round_trip/remove_size_one_axes.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string_view>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::getMeshAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::ManualComputationOp;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

bool hasSizeOneAxes(MeshOp meshOp) {
  return llvm::any_of(meshOp.getMesh().getAxes(),
                      [](MeshAxisAttr axis) { return axis.getSize() == 1; });
}

MeshAttr removeSizeOneAxes(MeshAttr mesh) {
  SmallVector<MeshAxisAttr> axes;
  llvm::copy_if(mesh.getAxes(), std::back_inserter(axes),
                [](MeshAxisAttr axis) { return axis.getSize() != 1; });
  return MeshAttr::get(mesh.getContext(), axes);
}

TensorShardingAttr removeSizeOneAxes(TensorShardingAttr sharding,
                                     const SymbolTable& symbolTable) {
  MeshAttr mesh = getMeshAttr(symbolTable, sharding.getMeshName());
  CHECK(mesh) << "unknown mesh: " << std::string_view(sharding.getMeshName());

  auto isNotSizeOne = [&](AxisRefAttr axis) { return axis.getSize(mesh) != 1; };

  // Remove from dimension shardings.
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(sharding.getRank());
  for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
    SmallVector<AxisRefAttr> newAxes;
    newAxes.reserve(dimSharding.getAxes().size());
    llvm::copy_if(dimSharding.getAxes(), std::back_inserter(newAxes),
                  isNotSizeOne);
    // Remove priority if there are no sharding axes and the dimension is
    // closed, since this isn't allowed by verification (would have no effect on
    // propagation).
    std::optional<int64_t> priority =
        newAxes.empty() && dimSharding.getIsClosed()
            ? std::nullopt
            : dimSharding.getPriority();
    dimShardings.push_back(
        DimensionShardingAttr::get(dimSharding.getContext(), newAxes,
                                   dimSharding.getIsClosed(), priority));
  }

  // Remove from replicated axes.
  SmallVector<AxisRefAttr> replicatedAxes;
  llvm::copy_if(sharding.getReplicatedAxes(),
                std::back_inserter(replicatedAxes), isNotSizeOne);

  return TensorShardingAttr::get(sharding.getContext(), sharding.getMeshName(),
                                 dimShardings, replicatedAxes);
}

TensorShardingPerValueAttr removeSizeOneAxes(
    TensorShardingPerValueAttr shardings, const SymbolTable& symbolTable) {
  SmallVector<TensorShardingAttr> newShardings;
  newShardings.reserve(shardings.size());
  for (TensorShardingAttr sharding : shardings.getShardings()) {
    newShardings.push_back(removeSizeOneAxes(sharding, symbolTable));
  }
  return TensorShardingPerValueAttr::get(shardings.getContext(), newShardings);
}

ManualAxesAttr removeSizeOneAxes(ManualAxesAttr manualAxes, MeshAttr mesh) {
  SmallVector<StringAttr> newAxes;
  llvm::copy_if(
      manualAxes.getValue(), std::back_inserter(newAxes),
      [&](StringAttr axisName) { return mesh.getAxisSize(axisName) != 1; });
  return ManualAxesAttr::get(manualAxes.getContext(), newAxes);
}

void removeSizeOneAxes(ManualComputationOp manualComputationOp,
                       const SymbolTable& symbolTable) {
  CHECK(!manualComputationOp->getOperands().empty() &&
        !manualComputationOp->getResults().empty())
      << "ManualComputationOp must have at least one operand or one result";
  std::optional<StringRef> meshName = mlir::sdy::getCommonMeshName(
      manualComputationOp.getInShardings().getShardings(),
      manualComputationOp.getOutShardings().getShardings());
  CHECK(meshName) << "all in/out shardings must have the same mesh";
  MeshAttr mesh = getMeshAttr(symbolTable, *meshName);
  CHECK(mesh) << "unknown mesh: " << std::string_view(*meshName);

  manualComputationOp.setInShardingsAttr(
      removeSizeOneAxes(manualComputationOp.getInShardingsAttr(), symbolTable));
  manualComputationOp.setOutShardingsAttr(removeSizeOneAxes(
      manualComputationOp.getOutShardingsAttr(), symbolTable));
  manualComputationOp.setManualAxesAttr(
      removeSizeOneAxes(manualComputationOp.getManualAxesAttr(), mesh));
}

void removeSizeOneAxes(FuncOp funcOp, const SymbolTable& symbolTable) {
  for (mlir::BlockArgument arg : funcOp.getArguments()) {
    if (auto sharding = mlir::sdy::getSharding(arg)) {
      mlir::sdy::setSharding(arg, removeSizeOneAxes(sharding, symbolTable));
    }
  }

  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    if (auto sharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
            resNum, kShardingAttr)) {
      funcOp.setResultAttr(resNum, kShardingAttr,
                           removeSizeOneAxes(sharding, symbolTable));
    }
  }

  funcOp.front().walk([&](Operation* op) {
    return mlir::TypeSwitch<Operation*, void>(op)
        .Case<ManualComputationOp>(
            [&](ManualComputationOp manualComputationOp) {
              removeSizeOneAxes(manualComputationOp, symbolTable);
            })
        .Default([&](Operation* op) {
          if (auto sharding = op->getAttrOfType<TensorShardingPerValueAttr>(
                  kShardingAttr)) {
            op->setAttr(kShardingAttr,
                        removeSizeOneAxes(sharding, symbolTable));
          }
        });
  });
}

class SdyRoundTripRemoveSizeOneAxesPass
    : public mlir::PassWrapper<SdyRoundTripRemoveSizeOneAxesPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripRemoveSizeOneAxesPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    if (llvm::none_of(moduleOp.getOps<MeshOp>(), hasSizeOneAxes)) {
      // Nothing to do.
      return;
    }

    LOG(INFO) << "[Shardy] removing axes of size one.";

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      removeSizeOneAxes(funcOp, symbolTable);
    }

    for (auto meshOp : moduleOp.getOps<MeshOp>()) {
      meshOp.setMeshAttr(removeSizeOneAxes(meshOp.getMesh()));
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-remove-size-one-axes";
  }

  StringRef getDescription() const override {
    return "Removes axes of size one from all meshes, shardings, and manual "
           "computation ops, to avoid conflict during propagation that are due "
           "to such axes.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripRemoveSizeOneAxesPass() {
  return std::make_unique<SdyRoundTripRemoveSizeOneAxesPass>();
}

void registerSdyRoundTripRemoveSizeOneAxesPass() {
  mlir::registerPass(createSdyRoundTripRemoveSizeOneAxesPass);
}

}  // namespace sdy
}  // namespace xla
