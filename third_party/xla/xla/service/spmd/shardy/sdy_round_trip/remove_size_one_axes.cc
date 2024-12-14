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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::ManualComputationOp;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::TensorShardingAttr;

bool hasSizeOneAxes(MeshOp meshOp) {
  return llvm::any_of(meshOp.getMesh().getAxes(),
                      [](MeshAxisAttr axis) { return axis.getSize() == 1; });
}

MeshAttr removeSizeOneAxes(MeshAttr mesh) {
  SmallVector<MeshAxisAttr> axes;
  llvm::copy_if(mesh.getAxes(), std::back_inserter(axes),
                [](MeshAxisAttr axis) { return axis.getSize() != 1; });
  return MeshAttr::get(mesh.getContext(), axes, mesh.getDeviceIds());
}

TensorShardingAttr removeSizeOneAxes(TensorShardingAttr sharding,
                                     const SymbolTable& symbolTable) {
  MeshAttr mesh = sharding.getMesh(symbolTable);
  CHECK(mesh) << "unknown mesh: " << absl::string_view(sharding.getMeshName());

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

  // Remove for inlined mesh.
  mlir::Attribute meshOrRef = sharding.getMeshOrRef();
  if (auto mesh = mlir::dyn_cast<MeshAttr>(meshOrRef)) {
    meshOrRef = removeSizeOneAxes(mesh);
  }

  return TensorShardingAttr::get(sharding.getContext(), meshOrRef, dimShardings,
                                 replicatedAxes);
}

void removeSizeOneManualAxes(ManualComputationOp manualComputationOp,
                             const SymbolTable& symbolTable) {
  MeshAttr mesh = mlir::sdy::getCommonMesh(
      manualComputationOp.getInShardings().getShardings(),
      manualComputationOp.getOutShardings().getShardings(), symbolTable);
  CHECK(mesh) << "no common mesh found for ManualComputationOp";

  SmallVector<StringAttr> newManualAxes;
  llvm::copy_if(
      manualComputationOp.getManualAxes(), std::back_inserter(newManualAxes),
      [&](StringAttr axisName) { return mesh.getAxisSize(axisName) != 1; });
  manualComputationOp.setManualAxesAttr(
      ManualAxesAttr::get(manualComputationOp.getContext(), newManualAxes));
}

class SdyRoundTripRemoveSizeOneAxesPass
    : public mlir::PassWrapper<SdyRoundTripRemoveSizeOneAxesPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      SdyRoundTripRemoveSizeOneAxesPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);

    if (llvm::none_of(moduleOp.getOps<MeshOp>(), hasSizeOneAxes)) {
      // Nothing to do.
      return;
    }

    LOG(INFO) << "[Shardy] removing axes of size one.";

    mlir::sdy::transformShardings(
        moduleOp,
        [&](TensorShardingAttr sharding) {
          return removeSizeOneAxes(sharding, symbolTable);
        },
        [&](Operation* op) {
          if (auto manualComputationOp =
                  mlir::dyn_cast<ManualComputationOp>(op)) {
            removeSizeOneManualAxes(manualComputationOp, symbolTable);
          }
        });

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
