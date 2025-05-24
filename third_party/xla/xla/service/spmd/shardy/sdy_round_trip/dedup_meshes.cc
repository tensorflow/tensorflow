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

#include "xla/service/spmd/shardy/sdy_round_trip/dedup_meshes.h"

#include <cstdint>
#include <iterator>
#include <memory>  // IWYU pragma: keep
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"

namespace xla {
namespace sdy {

namespace {

using ::llvm::SmallDenseMap;
using ::mlir::ArrayRef;
using ::mlir::DenseMap;
using ::mlir::DenseSet;
using ::mlir::ModuleOp;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::SymbolTable;

namespace sdy = ::mlir::sdy;

using AxisRefMap = SmallDenseMap<sdy::AxisRefAttr, sdy::AxisRefAttr>;

// Try to map the axes of `targetMesh` to the (sub)axes of `mainMesh`. If
// successful, `targetToMainAxisMap` will be populated.
bool mapTargetAxesToMainAxes(sdy::MeshOp targetMesh, sdy::MeshOp mainMesh,
                             AxisRefMap& targetToMainAxisMap,
                             mlir::MLIRContext* context) {
  ArrayRef<sdy::MeshAxisAttr> mainAxes = mainMesh.getMeshAttr().getAxes();
  ArrayRef<sdy::MeshAxisAttr> targetAxes = targetMesh.getMeshAttr().getAxes();
  int mainAxisIndex = 0;
  int targetAxisIndex = 0;
  int64_t preSize = 1;
  int64_t remainingMainAxisSize = 1;
  bool renewRemainingMainAxisSize = true;
  while (mainAxisIndex < mainAxes.size() &&
         targetAxisIndex < targetAxes.size()) {
    const sdy::MeshAxisAttr& mainAxis = mainAxes[mainAxisIndex];
    const sdy::MeshAxisAttr& targetAxis = targetAxes[targetAxisIndex];
    if (renewRemainingMainAxisSize) {
      remainingMainAxisSize = mainAxis.getSize();
      renewRemainingMainAxisSize = false;
    }
    if (remainingMainAxisSize % targetAxis.getSize() != 0) {
      return false;
    }
    const sdy::AxisRefAttr targetAxisRef =
        sdy::AxisRefAttr::get(context, targetAxis.getName(), {});
    // Set SubAxisInfoAttr to null when target axis maps to an entire main axis.
    const sdy::AxisRefAttr mainAxisRef = sdy::AxisRefAttr::get(
        context, mainAxis.getName(),
        mainAxis.getSize() == targetAxis.getSize()
            ? nullptr
            : sdy::SubAxisInfoAttr::get(context, preSize,
                                        targetAxis.getSize()));
    targetToMainAxisMap[targetAxisRef] = mainAxisRef;
    preSize *= targetAxis.getSize();
    targetAxisIndex++;
    remainingMainAxisSize /= targetAxis.getSize();
    if (remainingMainAxisSize == 1) {
      preSize = 1;
      mainAxisIndex++;
      renewRemainingMainAxisSize = true;
    }
  }
  return true;
}

// Maps a pair of <total number of devices, device_ids> to a list of meshes
// that share those properties. Total number of devices is used to distinguish
// default meshes.
using DeviceIdsToMeshesMap =
    SmallDenseMap<std::pair<int64_t, ArrayRef<int64_t>>,
                  SmallVector<sdy::MeshOp>>;

// Maps a mesh name to the main mesh name that will be used to replace it, and a
// map of old axes to the (sub)axes in the main mesh.
using MeshToAxisMap = SmallDenseMap<
    StringRef,
    std::pair<StringRef, SmallDenseMap<sdy::AxisRefAttr, sdy::AxisRefAttr>>>;

// Builds a map of meshes to the main mesh name that will be used (which has the
// same total number of devices and device_ids) and a map of old axis name to
// the axis names in the main mesh.
//
// NOTE: the main mesh will not be saved as a key in the map, since it won't
// need to be replaced.
MeshToAxisMap buildDuplicateMeshesToAxisMap(ModuleOp moduleOp) {
  DeviceIdsToMeshesMap deviceIdsToMeshesMap;
  for (sdy::MeshOp meshOp : moduleOp.getOps<sdy::MeshOp>()) {
    deviceIdsToMeshesMap[{meshOp.getMesh().getTotalSize(),
                          meshOp.getMesh().getDeviceIds()}]
        .push_back(meshOp);
  }
  // Process axis mapping from target meshes to the main mesh for each group.
  MeshToAxisMap duplicateMeshesToAxisMap;
  for (const auto& [_, meshes] : deviceIdsToMeshesMap) {
    if (meshes.size() < 2) {
      continue;
    }
    // Use the first mesh with real axes as the main mesh. Use the first mesh if
    // all meshes have fake axes.
    sdy::MeshOp mainMesh = meshes[0];
    for (sdy::MeshOp meshOp : meshes) {
      // All meshes in this group are empty, use the first one.
      if (meshOp.getMeshAttr().getAxes().empty()) {
        break;
      }
      if (!meshOp.getMeshAttr().getAxes()[0].getName().starts_with("_")) {
        break;
      }
    }

    for (sdy::MeshOp targetMesh : meshes) {
      if (targetMesh == mainMesh) {
        continue;
      }
      SmallDenseMap<sdy::AxisRefAttr, sdy::AxisRefAttr> targetToMainAxisMap;
      if (mapTargetAxesToMainAxes(targetMesh, mainMesh, targetToMainAxisMap,
                                  moduleOp.getContext())) {
        duplicateMeshesToAxisMap.try_emplace(targetMesh.getSymName(),
                                             mainMesh.getSymName(),
                                             std::move(targetToMainAxisMap));
      }
    }
  }
  return duplicateMeshesToAxisMap;
}

// Replaces the shardings attrs that refer to some mesh that isn't the main
// mesh saved in the pair of `MeshToAxisMap` with the main mesh. All shardings
// which before referred to different meshes with the same axis sizes will now
// refer to one meshes. So there can still be multiple meshes, but they will all
// all have different axis sizes.
void dedupMeshes(ModuleOp moduleOp,
                 const MeshToAxisMap& duplicateMeshesToAxisMap) {
  mlir::MLIRContext* context = moduleOp.getContext();
  sdy::transformShardings(
      moduleOp,
      [&](sdy::TensorShardingAttr oldSharding) -> sdy::TensorShardingAttr {
        if (mlir::isa<sdy::MeshAttr>(oldSharding.getMeshOrRef())) {
          // Skip shardings with inlined meshes.
          return oldSharding;
        }
        auto meshNameAndAxisMap =
            duplicateMeshesToAxisMap.find(oldSharding.getMeshName());
        // Exit early since this is a sharding with the main mesh that will be
        // used.
        if (meshNameAndAxisMap == duplicateMeshesToAxisMap.end()) {
          return oldSharding;
        }
        auto [mainMeshName, axisMap] = meshNameAndAxisMap->getSecond();
        auto buildNewAxisRef = [&, &axisMap =
                                       axisMap](sdy::AxisRefAttr oldAxisRef) {
          // If the old axis had a sub-axis info, we need to look up the
          // corresponding main axis without sub-axis info, and build the new
          // axis info.
          if (oldAxisRef.getSubAxisInfo()) {
            sdy::AxisRefAttr mappedAxisRef;
            mappedAxisRef = axisMap.at(
                sdy::AxisRefAttr::get(context, oldAxisRef.getName(), {}));
            sdy::SubAxisInfoAttr newSubAxisInfo;
            if (mappedAxisRef.getSubAxisInfo()) {
              newSubAxisInfo = sdy::SubAxisInfoAttr::get(
                  context,
                  mappedAxisRef.getSubAxisInfo().getPreSize() *
                      oldAxisRef.getSubAxisInfo().getPreSize(),
                  oldAxisRef.getSubAxisInfo().getSize());
            } else {
              newSubAxisInfo = oldAxisRef.getSubAxisInfo();
            }
            return sdy::AxisRefAttr::get(context, mappedAxisRef.getName(),
                                         newSubAxisInfo);
          }
          return sdy::AxisRefAttr::get(context,
                                       axisMap.at(oldAxisRef).getName(),
                                       axisMap.at(oldAxisRef).getSubAxisInfo());
        };
        SmallVector<sdy::DimensionShardingAttr> newDimShardings;
        newDimShardings.reserve(oldSharding.getDimShardings().size());
        for (sdy::DimensionShardingAttr oldDimSharding :
             oldSharding.getDimShardings()) {
          SmallVector<sdy::AxisRefAttr> newAxisRefs;
          newAxisRefs.reserve(oldDimSharding.getAxes().size());
          llvm::transform(oldDimSharding.getAxes(),
                          std::back_inserter(newAxisRefs), buildNewAxisRef);
          newDimShardings.push_back(sdy::DimensionShardingAttr::get(
              context, newAxisRefs, oldDimSharding.getIsClosed(),
              oldDimSharding.getPriority()));
        }
        auto buildNewAxisRefList =
            [buildNewAxisRef](ArrayRef<sdy::AxisRefAttr> oldAxisRefs) {
              SmallVector<sdy::AxisRefAttr> newAxisRefs;
              newAxisRefs.reserve(oldAxisRefs.size());
              llvm::transform(oldAxisRefs, std::back_inserter(newAxisRefs),
                              buildNewAxisRef);
              return newAxisRefs;
            };
        SmallVector<sdy::AxisRefAttr> newReplicatedAxes =
            buildNewAxisRefList(oldSharding.getReplicatedAxes());
        SmallVector<sdy::AxisRefAttr> newUnreducedAxes =
            buildNewAxisRefList(oldSharding.getUnreducedAxes());
        return sdy::TensorShardingAttr::get(context, mainMeshName,
                                            newDimShardings, newReplicatedAxes,
                                            newUnreducedAxes);
      });
}

void eraseMeshes(ModuleOp moduleOp,
                 const MeshToAxisMap& duplicateMeshesToAxisMap) {
  SymbolTable symbolTable(moduleOp);
  for (const auto& [meshName, _] : duplicateMeshesToAxisMap) {
    symbolTable.erase(symbolTable.lookup(meshName));
  }
}

class SdyRoundTripDedupMeshesPass
    : public mlir::PassWrapper<SdyRoundTripDedupMeshesPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripDedupMeshesPass)

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Exit early if there are no meshes or just one mesh.
    auto meshIter = moduleOp.getOps<sdy::MeshOp>();
    if (meshIter.empty()) {
      return;
    }
    auto meshIterBegin = meshIter.begin();
    std::advance(meshIterBegin, 1);
    if (meshIterBegin == meshIter.end()) {
      return;
    }

    MeshToAxisMap duplicateMeshesToAxisMap =
        buildDuplicateMeshesToAxisMap(moduleOp);
    dedupMeshes(moduleOp, duplicateMeshesToAxisMap);
    eraseMeshes(moduleOp, duplicateMeshesToAxisMap);
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-dedup-meshes";
  }

  StringRef getDescription() const override {
    return "Creates the pass that deduplicates any meshes with the same axis "
           "sizes (in the same order) but different names into a single mesh.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createSdyRoundTripDedupMeshesPass() {
  return std::make_unique<SdyRoundTripDedupMeshesPass>();
}

void registerSdyRoundTripDedupMeshesPass() {
  mlir::registerPass(createSdyRoundTripDedupMeshesPass);
}

}  // namespace sdy
}  // namespace xla
