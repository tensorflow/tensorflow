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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <memory>  // IWYU pragma: keep
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/common/sharding_walker.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla {
namespace sdy {

namespace {

using ::llvm::DenseMapInfo;
using ::llvm::SmallDenseMap;
using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SubAxisInfoAttr;
using ::mlir::sdy::TensorShardingAttr;

namespace sdy = ::mlir::sdy;

using AxisRefVector = SmallVector<AxisRefAttr>;
// Maps a target axis name to a list of (sub-)axes in the main mesh.
using AxisMap = SmallDenseMap<StringRef, AxisRefVector>;
using TotalDeviceCountMapInfo = DenseMapInfo<int64_t>;
using DeviceIdsMapInfo = DenseMapInfo<ArrayRef<int64_t>>;

bool hasFakeAxis(MeshOp mesh) {
  ArrayRef<MeshAxisAttr> axes = mesh.getMesh().getAxes();
  if (axes.empty()) {
    return false;
  }
  return axes[0].getName().starts_with("_");
}

struct MeshDeviceIdentifier {
  int64_t totalDeviceCount;
  ArrayRef<int64_t> deviceIds;

  MeshDeviceIdentifier(int64_t totalDeviceCount, ArrayRef<int64_t> deviceIds)
      : totalDeviceCount(totalDeviceCount), deviceIds(deviceIds) {}

  // Use -1 as the total device count for empty meshes, to distinguish them
  // from meshes with all axes of size 1.
  explicit MeshDeviceIdentifier(MeshAttr mesh)
      : MeshDeviceIdentifier(mesh.getAxes().empty() ? -1 : mesh.getTotalSize(),
                             mesh.getDeviceIds()) {}
};

struct MeshDeviceIdentifierInfo : public DenseMapInfo<MeshDeviceIdentifier> {
  static unsigned getHashValue(const MeshDeviceIdentifier& inputs) {
    return llvm::hash_combine(
        TotalDeviceCountMapInfo::getHashValue(inputs.totalDeviceCount),
        DeviceIdsMapInfo::getHashValue(inputs.deviceIds));
  }

  static bool isEqual(const MeshDeviceIdentifier& lhs,
                      const MeshDeviceIdentifier& rhs) {
    return lhs.totalDeviceCount == rhs.totalDeviceCount &&
           lhs.deviceIds == rhs.deviceIds;
  }
};

// Try to map the axes of `targetMesh` to the (sub)axes of `mainMesh`. If
// successful, `targetToMainAxisMap` will be populated.
bool mapTargetAxesToMainAxes(MeshOp targetMesh, MeshOp mainMesh,
                             AxisMap& targetToMainAxisMap,
                             MLIRContext* context) {
  ArrayRef<MeshAxisAttr> mainAxes = mainMesh.getMesh().getAxes();
  ArrayRef<MeshAxisAttr> targetAxes = targetMesh.getMesh().getAxes();
  struct AxisMappingState {
    int64_t axisIndex = -1;
    int64_t preSize = 1;
    int64_t remainingSize = 1;
  };

  AxisMappingState mainAxisState, targetAxisState;
  auto advanceToNextNonSizeOneAxis = [](ArrayRef<MeshAxisAttr> axes,
                                        AxisMappingState& state) {
    state.axisIndex++;
    while (state.axisIndex < axes.size() &&
           axes[state.axisIndex].getSize() == 1) {
      state.axisIndex++;
    }
    state.preSize = 1;
    if (state.axisIndex < axes.size()) {
      state.remainingSize = axes[state.axisIndex].getSize();
    }
  };
  advanceToNextNonSizeOneAxis(mainAxes, mainAxisState);
  advanceToNextNonSizeOneAxis(targetAxes, targetAxisState);
  while (mainAxisState.axisIndex < mainAxes.size() &&
         targetAxisState.axisIndex < targetAxes.size()) {
    MeshAxisAttr mainAxis = mainAxes[mainAxisState.axisIndex];
    MeshAxisAttr targetAxis = targetAxes[targetAxisState.axisIndex];

    const auto [sizeToConsume, largerRemainingSize] = std::minmax(
        {mainAxisState.remainingSize, targetAxisState.remainingSize});
    if (largerRemainingSize % sizeToConsume != 0) {
      return false;
    }
    AxisRefAttr mainAxisRef = AxisRefAttr::get(
        context, mainAxis.getName(),
        mainAxis.getSize() == sizeToConsume
            ? nullptr
            : SubAxisInfoAttr::get(context, mainAxisState.preSize,
                                   sizeToConsume));
    targetToMainAxisMap[targetAxis.getName()].push_back(mainAxisRef);

    auto updateMappingState =
        [sizeToConsume = sizeToConsume, advanceToNextNonSizeOneAxis](
            ArrayRef<MeshAxisAttr> axes, AxisMappingState& state) {
          state.remainingSize /= sizeToConsume;
          if (state.remainingSize == 1) {
            advanceToNextNonSizeOneAxis(axes, state);
          } else {
            state.preSize *= sizeToConsume;
          }
        };
    updateMappingState(mainAxes, mainAxisState);
    updateMappingState(targetAxes, targetAxisState);
  }
  CHECK_EQ(mainAxisState.remainingSize, 1);
  CHECK_EQ(targetAxisState.remainingSize, 1);
  CHECK_EQ(mainAxisState.axisIndex, mainAxes.size());
  CHECK_EQ(targetAxisState.axisIndex, targetAxes.size());
  return true;
}

// Maps a pair of <total number of devices, device_ids> to a list of meshes
// that share those properties. Total number of devices is used to distinguish
// default meshes.
using MeshIdentifierToMainMeshesMap =
    SmallDenseMap<MeshDeviceIdentifier, MeshOp, 4, MeshDeviceIdentifierInfo>;

// Maps a mesh name to the main mesh information that will be used to replace
// it: the main mesh symbol name, the updated MeshAttr of the main mesh, and a
// map of old axes to the (sub)axes in the main mesh.
struct MainMeshInfo {
  StringRef mainMeshName;
  MeshAttr mainMesh;
  AxisMap axisMap;
};
using MeshToAxisMap = SmallDenseMap<StringRef, MainMeshInfo>;

void addOrMergeFullAxis(StringRef oldAxisName, AxisRefVector& newAxisRefs,
                        const MeshAttr& mainMesh, const AxisMap& axisMap) {
  if (auto it = axisMap.find(oldAxisName); it != axisMap.end()) {
    for (AxisRefAttr mainAxisRef : it->second) {
      sdy::addAxisOrMerge(newAxisRefs, mainAxisRef, mainMesh);
    }
  }
}

// Takes an old axis ref, find the mapped new (sub-)axes, add them to the new
// axis ref list and merge consecutive sub-axes.
void addOrMergeNewAxisRefAttr(AxisRefAttr oldAxisRef,
                              AxisRefVector& newAxisRefs, MLIRContext* context,
                              const MeshAttr& mainMesh,
                              const AxisMap& axisMap) {
  if (!oldAxisRef.getSubAxisInfo()) {
    addOrMergeFullAxis(oldAxisRef.getName(), newAxisRefs, mainMesh, axisMap);
    return;
  }
  auto it = axisMap.find(oldAxisRef.getName());
  if (it == axisMap.end()) {
    // The old axis is of size 1, skip it.
    return;
  }
  ArrayRef<AxisRefAttr> mainAxisRefs = it->second;
  // TODO(zenong): We can iterate the list once for sub-axes of a target axis.
  int64_t remainingOldPreSize = oldAxisRef.getSubAxisInfo().getPreSize();
  int64_t remainingOldSize = oldAxisRef.getSubAxisInfo().getSize();
  for (AxisRefAttr mainAxisRef : mainAxisRefs) {
    // Consume presize from the old axis ref.
    const int64_t mainAxisSize = mainAxisRef.getSize(mainMesh);
    // Find the correct (sub-)axis to start from.
    if (remainingOldPreSize >= mainAxisSize) {
      // Cases like "[a":6] -> ["x":2, "y":3] then "a":(3)2 are not supported.
      CHECK_EQ(remainingOldPreSize % mainAxisSize, 0);
      remainingOldPreSize /= mainAxisSize;
      continue;
    }
    // Cases like "[a":6] -> ["x":3, "y":2] then "a":(2)3 are not supported.
    CHECK_EQ(mainAxisSize % remainingOldPreSize, 0);
    const int64_t consumableMainAxisSize = mainAxisSize / remainingOldPreSize;
    const int64_t consumedSize =
        std::min(remainingOldSize, consumableMainAxisSize);
    AxisRefAttr newAxisRef =
        consumedSize == mainAxisSize
            ? mainAxisRef
            : AxisRefAttr::get(
                  context, mainAxisRef.getName(),
                  remainingOldPreSize * mainAxisRef.getSubAxisPreSize(),
                  consumedSize);
    sdy::addAxisOrMerge(newAxisRefs, newAxisRef, mainMesh);
    // Cases like "[a":6] -> ["x":2, "y":3] then "a":(1)3 are not supported.
    CHECK_EQ(remainingOldSize % consumedSize, 0);
    remainingOldPreSize = 1;
    remainingOldSize /= consumedSize;
    if (remainingOldSize == 1) {
      break;
    }
  }
}

int64_t countNonSizeOneAxes(MeshOp mesh) {
  return llvm::count_if(mesh.getMesh().getAxes(),
                        [](MeshAxisAttr axis) { return axis.getSize() > 1; });
}

// Returns true if `candidate` is preferred over `current` as a main mesh:
// 1. The mesh contains real axes rather than fake axes.
// 2. The mesh has more non-size-one axes.
bool isBetterMainMesh(MeshOp candidate, MeshOp current) {
  bool candidateIsFake = hasFakeAxis(candidate);
  bool currentIsFake = hasFakeAxis(current);
  if (candidateIsFake != currentIsFake) {
    return !candidateIsFake;
  }
  return countNonSizeOneAxes(candidate) > countNonSizeOneAxes(current);
}

// Builds a map of meshes to the main mesh that will be used (which has the
// same total number of devices and device_ids), the final MeshAttr of that main
// mesh (aligned to the canonical mesh for that total device count when
// compatible), and a map of old axis name to the axis names in the aligned main
// mesh.
//
// A main mesh whose axes are already canonical will not be saved as a key in
// the map, since it won't need to be updated or replaced.
MeshToAxisMap buildDuplicateMeshesToAxisMap(ModuleOp moduleOp) {
  MeshIdentifierToMainMeshesMap meshIdentifierToMainMeshesMap;
  SmallDenseMap<int64_t, MeshOp, 4> totalDeviceCountToMainMeshMap;
  for (MeshOp meshOp : moduleOp.getOps<MeshOp>()) {
    MeshDeviceIdentifier id{meshOp.getMesh()};
    if (auto [it, inserted] =
            meshIdentifierToMainMeshesMap.try_emplace(id, meshOp);
        !inserted && isBetterMainMesh(meshOp, it->getSecond())) {
      it->getSecond() = meshOp;
    }
    if (auto [it, inserted] = totalDeviceCountToMainMeshMap.try_emplace(
            id.totalDeviceCount, meshOp);
        !inserted && isBetterMainMesh(meshOp, it->getSecond())) {
      it->getSecond() = meshOp;
    }
  }

  MLIRContext* context = moduleOp.getContext();
  MeshToAxisMap duplicateMeshesToAxisMap;
  for (auto [id, deviceMainMesh] : meshIdentifierToMainMeshesMap) {
    MeshOp sizeMainMesh =
        totalDeviceCountToMainMeshMap.lookup(id.totalDeviceCount);
    if (deviceMainMesh.getMesh().getAxes() ==
        sizeMainMesh.getMesh().getAxes()) {
      continue;
    }
    AxisMap targetToMainAxisMap;
    if (mapTargetAxesToMainAxes(deviceMainMesh, sizeMainMesh,
                                targetToMainAxisMap, context)) {
      MeshAttr newMeshAttr =
          MeshAttr::get(context, sizeMainMesh.getMesh().getAxes(),
                        deviceMainMesh.getMesh().getDeviceIds());
      duplicateMeshesToAxisMap.try_emplace(
          deviceMainMesh.getSymName(),
          MainMeshInfo{deviceMainMesh.getSymName(), newMeshAttr,
                       std::move(targetToMainAxisMap)});
    }
  }

  for (MeshOp targetMesh : moduleOp.getOps<MeshOp>()) {
    MeshDeviceIdentifier id{targetMesh.getMesh()};
    MeshOp deviceMainMesh = meshIdentifierToMainMeshesMap.lookup(id);
    if (targetMesh == deviceMainMesh) {
      continue;
    }
    MeshOp sizeMainMesh =
        totalDeviceCountToMainMeshMap.lookup(id.totalDeviceCount);
    auto deviceMainIt =
        duplicateMeshesToAxisMap.find(deviceMainMesh.getSymName());
    bool canAlignToSizeMain =
        deviceMainIt != duplicateMeshesToAxisMap.end() ||
        deviceMainMesh.getMesh().getAxes() == sizeMainMesh.getMesh().getAxes();
    MeshOp axisMainMesh = canAlignToSizeMain ? sizeMainMesh : deviceMainMesh;
    MeshAttr mainMeshAttr = deviceMainIt != duplicateMeshesToAxisMap.end()
                                ? deviceMainIt->getSecond().mainMesh
                                : deviceMainMesh.getMesh();

    AxisMap targetToMainAxisMap;
    if (mapTargetAxesToMainAxes(targetMesh, axisMainMesh, targetToMainAxisMap,
                                context)) {
      duplicateMeshesToAxisMap.try_emplace(
          targetMesh.getSymName(),
          MainMeshInfo{deviceMainMesh.getSymName(), mainMeshAttr,
                       std::move(targetToMainAxisMap)});
    }
  }
  return duplicateMeshesToAxisMap;
}

// Replaces `oldSharding`, if it refers to some mesh that isn't the main
// mesh saved in `MeshToAxisMap`, with the main mesh.
TensorShardingAttr replaceAxesInSharding(
    TensorShardingAttr oldSharding,
    const MeshToAxisMap& duplicateMeshesToAxisMap) {
  MLIRContext* context = oldSharding.getContext();
  if (mlir::isa<MeshAttr>(oldSharding.getMeshOrRef())) {
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
  const auto& [mainMeshName, mainMesh, axisMap] =
      meshNameAndAxisMap->getSecond();
  auto buildNewAxisRefList =
      [&, &axisMap = axisMap,
       &mainMesh = mainMesh](ArrayRef<AxisRefAttr> oldAxisRefs) {
        AxisRefVector newAxisRefs;
        newAxisRefs.reserve(oldAxisRefs.size());
        for (AxisRefAttr oldAxisRef : oldAxisRefs) {
          addOrMergeNewAxisRefAttr(oldAxisRef, newAxisRefs, context, mainMesh,
                                   axisMap);
        }
        return newAxisRefs;
      };
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(oldSharding.getDimShardings().size());
  for (DimensionShardingAttr oldDimSharding : oldSharding.getDimShardings()) {
    newDimShardings.push_back(DimensionShardingAttr::get(
        context, buildNewAxisRefList(oldDimSharding.getAxes()),
        oldDimSharding.getIsClosed(), oldDimSharding.getPriority()));
  }
  AxisRefVector newReplicatedAxes =
      buildNewAxisRefList(oldSharding.getReplicatedAxes());
  AxisRefVector newUnreducedAxes =
      buildNewAxisRefList(oldSharding.getUnreducedAxes());
  return TensorShardingAttr::get(context, mainMeshName, newDimShardings,
                                 newReplicatedAxes, newUnreducedAxes,
                                 newUnreducedAxes.empty()
                                     ? sdy::ReductionOp::SUM
                                     : oldSharding.getReductionOp());
}

// Replaces the manual axes in `manualComputation`, if it refers to some mesh
// that isn't the main mesh saved in the pair of `MeshToAxisMap`, with the main
// mesh.
void replaceManualAxes(sdy::ManualComputationOp manualComputation,
                       const SymbolTable& symbolTable,
                       const MeshToAxisMap& duplicateMeshesToAxisMap) {
  if (manualComputation.getManualAxes().empty() ||
      (manualComputation.getInShardings().empty() &&
       manualComputation.getOutShardings().empty())) {
    return;
  }
  Attribute meshOrRef = getCommonMeshOrRef(
      manualComputation.getInShardings().getShardings(),
      manualComputation.getOutShardings().getShardings(), symbolTable);
  CHECK(meshOrRef) << "no common mesh found for ManualComputationOp";
  if (mlir::isa<MeshAttr>(meshOrRef)) {
    // Skip manual computation with inlined meshes.
    return;
  }
  auto meshNameAndAxisMap = duplicateMeshesToAxisMap.find(
      mlir::cast<mlir::FlatSymbolRefAttr>(meshOrRef).getValue());
  // Exit early since this is the main mesh that will be used.
  if (meshNameAndAxisMap == duplicateMeshesToAxisMap.end()) {
    return;
  }
  const auto& [mainMeshName, mainMesh, axisMap] =
      meshNameAndAxisMap->getSecond();
  MLIRContext* context = manualComputation.getContext();
  AxisRefVector newAxisRefs;
  newAxisRefs.reserve(manualComputation.getManualAxes().size());
  for (AxisRefAttr manualAxis : manualComputation.getManualAxes()) {
    addOrMergeNewAxisRefAttr(manualAxis, newAxisRefs, context, mainMesh,
                             axisMap);
  }
  manualComputation.setManualAxesAttr(
      sdy::ManualAxesAttr::get(context, newAxisRefs));
}

Attribute rewriteReplicaGroupsAttr(
    Attribute attr, MLIRContext* context,
    const MeshToAxisMap& duplicateMeshesToAxisMap) {
  auto rgv3 = mlir::dyn_cast<mlir::stablehlo::ReplicaGroupMeshAxesAttr>(attr);
  if (!rgv3) {
    return attr;
  }
  Attribute meshAttr = rgv3.getMesh();
  StringRef oldMeshName;
  if (auto symbolRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(meshAttr)) {
    oldMeshName = symbolRef.getValue();
  } else if (auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(meshAttr)) {
    oldMeshName = stringAttr.getValue();
  }
  if (oldMeshName.empty()) {
    return attr;
  }
  auto meshNameAndAxisMap = duplicateMeshesToAxisMap.find(oldMeshName);
  if (meshNameAndAxisMap == duplicateMeshesToAxisMap.end()) {
    return attr;
  }
  const auto& [mainMeshName, mainMesh, axisMap] =
      meshNameAndAxisMap->getSecond();

  AxisRefVector newSdyAxes;
  newSdyAxes.reserve(rgv3.getAxes().size());
  for (Attribute axis : rgv3.getAxes()) {
    auto axisRef = mlir::cast<mlir::stablehlo::AxisRefAttr>(axis);
    SubAxisInfoAttr sdySubAxisInfo;
    if (auto subAxisInfo = axisRef.getSubAxisInfo()) {
      sdySubAxisInfo = SubAxisInfoAttr::get(context, subAxisInfo.getPreSize(),
                                            subAxisInfo.getSize());
    }
    auto oldSdyAxisRef =
        AxisRefAttr::get(context, axisRef.getName(), sdySubAxisInfo);
    addOrMergeNewAxisRefAttr(oldSdyAxisRef, newSdyAxes, context, mainMesh,
                             axisMap);
  }

  SmallVector<Attribute> newAxes;
  newAxes.reserve(newSdyAxes.size());
  llvm::transform(newSdyAxes, std::back_inserter(newAxes),
                  sdy::convertAxisRefAttr);

  return mlir::stablehlo::ReplicaGroupMeshAxesAttr::get(
      context, mlir::FlatSymbolRefAttr::get(context, mainMeshName),
      mlir::ArrayAttr::get(context, newAxes));
}

void replaceReplicaGroups(ModuleOp moduleOp,
                          const MeshToAxisMap& duplicateMeshesToAxisMap) {
  MLIRContext* context = moduleOp.getContext();
  moduleOp.walk([&](Operation* op) {
    if (auto attr = op->getAttr("replica_groups")) {
      Attribute newAttr =
          rewriteReplicaGroupsAttr(attr, context, duplicateMeshesToAxisMap);
      if (newAttr != attr) {
        op->setAttr("replica_groups", newAttr);
      }
    }
  });
}

// Maintains the following meshes and remove all the other meshes.
// 1. For each unique combination of total size and device id order, keep one
//    main mesh.
// 2. inlined meshes.
//
// If a sharding or manual axes refer to a removed mesh, update them accordingly
// to use the respective main mesh.
void dedupMeshes(ModuleOp moduleOp, const SymbolTable& symbolTable,
                 const MeshToAxisMap& duplicateMeshesToAxisMap) {
  sdy::transformShardings(
      moduleOp,
      [&](TensorShardingAttr oldSharding) -> TensorShardingAttr {
        return replaceAxesInSharding(oldSharding, duplicateMeshesToAxisMap);
      },
      [&](Operation* op) {
        if (auto manualComputation =
                mlir::dyn_cast<sdy::ManualComputationOp>(op)) {
          replaceManualAxes(manualComputation, symbolTable,
                            duplicateMeshesToAxisMap);
        }
      });
  replaceReplicaGroups(moduleOp, duplicateMeshesToAxisMap);
  for (const auto& [targetMesh, mainMeshInfo] : duplicateMeshesToAxisMap) {
    if (targetMesh == mainMeshInfo.mainMeshName) {
      sdy::getMeshOp(symbolTable, targetMesh)
          .setMeshAttr(mainMeshInfo.mainMesh);
      continue;
    }
    if (failed(SymbolTable::replaceAllSymbolUses(
            symbolTable.lookup(targetMesh),
            StringAttr::get(moduleOp.getContext(), mainMeshInfo.mainMeshName),
            moduleOp))) {
      assert(false && "failed to rename mesh symbol in replica_groups");
    }
  }
}

void eraseMeshes(SymbolTable& symbolTable,
                 const MeshToAxisMap& duplicateMeshesToAxisMap) {
  for (const auto& [meshName, mainMeshInfo] : duplicateMeshesToAxisMap) {
    if (meshName != mainMeshInfo.mainMeshName) {
      symbolTable.erase(symbolTable.lookup(meshName));
    }
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
    auto meshOps = moduleOp.getOps<MeshOp>();
    if (meshOps.empty() || llvm::hasSingleElement(meshOps)) {
      return;
    }

    MeshToAxisMap duplicateMeshesToAxisMap =
        buildDuplicateMeshesToAxisMap(moduleOp);
    if (duplicateMeshesToAxisMap.empty()) {
      // No meshes to dedup.
      return;
    }
    SymbolTable symbolTable(moduleOp);
    dedupMeshes(moduleOp, symbolTable, duplicateMeshesToAxisMap);
    eraseMeshes(symbolTable, duplicateMeshesToAxisMap);
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-dedup-meshes";
  }

  StringRef getDescription() const override {
    return "Creates the pass that deduplicates any meshes with the same axis "
           "sizes (in the same order) but different names into a single mesh.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<sdy::SdyDialect>();
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
