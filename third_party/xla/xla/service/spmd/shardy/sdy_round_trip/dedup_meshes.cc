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

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
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

using ::llvm::DenseMapInfo;
using ::llvm::SmallDenseMap;
using ::mlir::ArrayRef;
using ::mlir::DenseMap;
using ::mlir::DenseSet;
using ::mlir::ModuleOp;
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

// A list of <target (sub-)axis refs, main (sub-)axis refs> pairs.
using AxisMappingList = SmallVector<std::pair<AxisRefAttr, AxisRefAttr>>;
// Maps a target axis name to a list of axis mapping pairs.
using AxisMap = SmallDenseMap<StringRef, AxisMappingList>;
using TotalDeviceCountMapInfo = DenseMapInfo<int64_t>;
using DeviceIdsMapInfo = DenseMapInfo<ArrayRef<int64_t>>;
using AxisRefVector = SmallVector<AxisRefAttr>;

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
  static inline MeshDeviceIdentifier getEmptyKey() {
    return {TotalDeviceCountMapInfo::getEmptyKey(),
            DeviceIdsMapInfo::getEmptyKey()};
  }

  static inline MeshDeviceIdentifier getTombstoneKey() {
    return {TotalDeviceCountMapInfo::getTombstoneKey(),
            DeviceIdsMapInfo::getTombstoneKey()};
  }
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

class AddAxisOrMergeInserter {
 public:
  using iterator_category = std::output_iterator_tag;
  using value_type = void;
  using difference_type = void;
  using pointer = void;
  using reference = void;

  explicit AddAxisOrMergeInserter(AxisRefVector* newAxisRefs,
                                  const MeshAttr* mesh)
      : axisRefs(newAxisRefs), mesh(mesh) {}

  AddAxisOrMergeInserter& operator=(const AxisRefAttr& value) {
    sdy::addAxisOrMerge(*axisRefs, value, *mesh);
    return *this;
  }

  AddAxisOrMergeInserter& operator=(const AxisRefVector& values) {
    for (const AxisRefAttr& value : values) {
      sdy::addAxisOrMerge(*axisRefs, value, *mesh);
    }
    return *this;
  }

  AddAxisOrMergeInserter& operator*() { return *this; }
  AddAxisOrMergeInserter& operator++() { return *this; }
  AddAxisOrMergeInserter& operator++(int) { return *this; }

 private:
  AxisRefVector* axisRefs;
  const MeshAttr* mesh;
};

// Try to map the axes of `targetMesh` to the (sub)axes of `mainMesh`. If
// successful, `targetToMainAxisMap` will be populated.
bool mapTargetAxesToMainAxes(MeshOp targetMesh, MeshOp mainMesh,
                             AxisMap& targetToMainAxisMap,
                             mlir::MLIRContext* context) {
  ArrayRef<MeshAxisAttr> mainAxes = mainMesh.getMesh().getAxes();
  ArrayRef<MeshAxisAttr> targetAxes = targetMesh.getMesh().getAxes();
  struct AxisMappingState {
    int64_t axisIndex = 0;
    int64_t preSize = 1;
    int64_t remainingSize = 1;
  };
  AxisMappingState mainAxisState, targetAxisState;
  while (mainAxisState.axisIndex < mainAxes.size() &&
         targetAxisState.axisIndex < targetAxes.size()) {
    const MeshAxisAttr mainAxis = mainAxes[mainAxisState.axisIndex];
    const MeshAxisAttr targetAxis = targetAxes[targetAxisState.axisIndex];
    // Skip mapping target axes of size 1 and remove the uses later.
    if (targetAxis.getSize() == 1) {
      targetAxisState.axisIndex++;
      continue;
    }
    // Bump to the next axis if the the current axis has been fully consumed
    // during the last iteration of mapping.
    if (mainAxisState.remainingSize == 1) {
      mainAxisState.remainingSize = mainAxis.getSize();
    }
    if (targetAxisState.remainingSize == 1) {
      targetAxisState.remainingSize = targetAxis.getSize();
    }
    // Map the from/smaller (sub-)axis to the to/larger (sub-)axis.
    // Return true if the mapping is successful.
    auto tryMapAxis = [&](const MeshAxisAttr& fromAxis,
                          AxisMappingState& fromState,
                          const MeshAxisAttr& toAxis, AxisMappingState& toState,
                          bool isTargetToMain) -> bool {
      if (toState.remainingSize % fromState.remainingSize != 0) {
        return false;
      }
      // Construct the sub axis info even if it's a full axis, as the axis size
      // is needed in deduping uses.
      const AxisRefAttr fromAxisRef =
          AxisRefAttr::get(context, fromAxis.getName(),
                           SubAxisInfoAttr::get(context, fromState.preSize,
                                                fromState.remainingSize));
      const AxisRefAttr toAxisRef =
          AxisRefAttr::get(context, toAxis.getName(),
                           SubAxisInfoAttr::get(context, toState.preSize,
                                                fromState.remainingSize));
      if (isTargetToMain) {
        targetToMainAxisMap[fromAxisRef.getName()].push_back(
            {fromAxisRef, toAxisRef});
      } else {
        targetToMainAxisMap[toAxisRef.getName()].push_back(
            {toAxisRef, fromAxisRef});
      }
      toState.remainingSize /= fromState.remainingSize;
      if (toState.remainingSize == 1) {
        toState.axisIndex++;
        toState.preSize = 1;
      } else {
        toState.preSize *= fromState.remainingSize;
      }
      fromState.preSize = 1;
      fromState.remainingSize = 1;
      fromState.axisIndex++;
      return true;
    };

    // Always try to map the smaller (sub-)axis to the larger (sub-)axis.
    if (mainAxisState.remainingSize > targetAxisState.remainingSize &&
        !tryMapAxis(targetAxis, targetAxisState, mainAxis, mainAxisState,
                    /*isTargetToMain=*/true)) {
      return false;
    }
    if (mainAxisState.remainingSize <= targetAxisState.remainingSize &&
        !tryMapAxis(mainAxis, mainAxisState, targetAxis, targetAxisState,
                    /*isTargetToMain=*/false)) {
      return false;
    }
  }
  // Consume the trailing axes of size 1.
  while (mainAxisState.axisIndex < mainAxes.size()) {
    if (mainAxes[mainAxisState.axisIndex].getSize() != 1) {
      break;
    }
    mainAxisState.axisIndex++;
  }
  while (targetAxisState.axisIndex < targetAxes.size()) {
    if (targetAxes[targetAxisState.axisIndex].getSize() != 1) {
      break;
    }
    targetAxisState.axisIndex++;
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

// Maps a mesh name to the main mesh name that will be used to replace it, and a
// map of old axes to the (sub)axes in the main mesh.
using MeshToAxisMap = SmallDenseMap<StringRef, std::pair<StringRef, AxisMap>>;

// Builds a map of meshes to the main mesh name that will be used (which has the
// same total number of devices and device_ids) and a map of old axis name to
// the axis names in the main mesh.
//
// NOTE: the main mesh will not be saved as a key in the map, since it won't
// need to be replaced.
MeshToAxisMap buildDuplicateMeshesToAxisMap(ModuleOp moduleOp) {
  MeshIdentifierToMainMeshesMap meshIdentifierToMainMeshesMap;
  MeshToAxisMap duplicateMeshesToAxisMap;
  // Use the first mesh with real axes as the main mesh. Use the first mesh if
  // all meshes have fake axes.
  for (MeshOp meshOp : moduleOp.getOps<MeshOp>()) {
    auto [entries, inserted] = meshIdentifierToMainMeshesMap.try_emplace(
        MeshDeviceIdentifier{meshOp.getMesh()}, meshOp);
    if (inserted) {
      continue;
    }
    MeshOp& mainMesh = entries->getSecond();
    // Update the main mesh if current mesh is real and main mesh is fake.
    if (hasFakeAxis(meshOp) || !hasFakeAxis(mainMesh)) {
      continue;
    }
    mainMesh = meshOp;
  }
  for (MeshOp targetMesh : moduleOp.getOps<MeshOp>()) {
    MeshOp mainMesh = meshIdentifierToMainMeshesMap.lookup(
        MeshDeviceIdentifier{targetMesh.getMesh()});
    if (targetMesh == mainMesh) {
      continue;
    }
    AxisMap targetToMainAxisMap;
    if (mapTargetAxesToMainAxes(targetMesh, mainMesh, targetToMainAxisMap,
                                moduleOp.getContext())) {
      duplicateMeshesToAxisMap.try_emplace(targetMesh.getSymName(),
                                           mainMesh.getSymName(),
                                           std::move(targetToMainAxisMap));
    }
  }
  return duplicateMeshesToAxisMap;
}

// For a `AxisMappingList`, extract the second (target) axis refs and remove sub
// axis info for full axes.
AxisRefVector extractTargetAxisRefs(AxisMappingList aixMappingList,
                                    const MeshAttr& mainMesh) {
  AxisRefVector toAxisRefs;
  toAxisRefs.reserve(aixMappingList.size());
  llvm::transform(aixMappingList, std::back_inserter(toAxisRefs),
                  [&mainMesh](const std::pair<AxisRefAttr, AxisRefAttr>& pair) {
                    AxisRefAttr mainAxisRef = pair.second;
                    CHECK(mainAxisRef.getSubAxisInfo());
                    AxisRefAttr mainAxisRefFull = AxisRefAttr::get(
                        mainAxisRef.getContext(), mainAxisRef.getName());
                    if (mainAxisRef.getSubAxisInfo().getSize() ==
                        mainAxisRefFull.getSize(mainMesh)) {
                      CHECK_EQ(mainAxisRef.getSubAxisPreSize(), 1);
                      return mainAxisRefFull;
                    }
                    return pair.second;
                  });
  return toAxisRefs;
}

// Replaces `oldSharding`, if it refers to some mesh that isn't the main
// mesh saved in the pair of `MeshToAxisMap`, with the main mesh.
TensorShardingAttr replaceAxesInSharding(
    TensorShardingAttr oldSharding, const SymbolTable& symbolTable,
    const MeshToAxisMap& duplicateMeshesToAxisMap) {
  mlir::MLIRContext* context = oldSharding.getContext();
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
  auto [mainMeshName, axisMap] = meshNameAndAxisMap->getSecond();
  MeshAttr mainMesh = sdy::getMeshOp(symbolTable, mainMeshName).getMesh();
  auto buildNewAxisRefs = [&](AxisRefAttr oldAxisRef) -> AxisRefVector {
    if (!axisMap.contains(oldAxisRef.getName())) {
      // The old axis is of size 1, skip it.
      return {};
    }
    AxisMappingList aixMappingList = axisMap.at(oldAxisRef.getName());
    if (!oldAxisRef.getSubAxisInfo()) {
      return extractTargetAxisRefs(aixMappingList, mainMesh);
    }
    AxisRefVector newAxisRefs;
    newAxisRefs.reserve(aixMappingList.size());
    // TODO(zenong): We can iterate the list once.
    int64_t oldPreSize = oldAxisRef.getSubAxisInfo().getPreSize();
    int64_t remainingOldSize = oldAxisRef.getSubAxisInfo().getSize();
    for (const auto& [fromAxisRef, toAxisRef] : aixMappingList) {
      // Consume presize from the old axis ref.
      SubAxisInfoAttr fromSubAxisInfo = fromAxisRef.getSubAxisInfo();
      CHECK(fromSubAxisInfo);
      SubAxisInfoAttr toSubAxisInfo = toAxisRef.getSubAxisInfo();
      CHECK(toSubAxisInfo);
      const int64_t fromSize = fromSubAxisInfo.getSize();
      // Find the correct to (sub-)axis to start from.
      if (oldPreSize >= fromSize) {
        oldPreSize /= fromSize;
        continue;
      }
      const int64_t consumedSize =
          remainingOldSize <= fromSize ? remainingOldSize : fromSize;
      SubAxisInfoAttr newSubAxisInfo = SubAxisInfoAttr::get(
          context, oldPreSize * toSubAxisInfo.getPreSize(), consumedSize);
      newAxisRefs.push_back(
          AxisRefAttr::get(context, toAxisRef.getName(), newSubAxisInfo));
      remainingOldSize /= consumedSize;
      if (remainingOldSize == 1) {
        break;
      }
    }
    return newAxisRefs;
  };
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(oldSharding.getDimShardings().size());
  for (DimensionShardingAttr oldDimSharding : oldSharding.getDimShardings()) {
    AxisRefVector newAxisRefs;
    newAxisRefs.reserve(oldDimSharding.getAxes().size());
    llvm::transform(oldDimSharding.getAxes(),
                    AddAxisOrMergeInserter(&newAxisRefs, &mainMesh),
                    buildNewAxisRefs);
    newDimShardings.push_back(DimensionShardingAttr::get(
        context, newAxisRefs, oldDimSharding.getIsClosed(),
        oldDimSharding.getPriority()));
  }
  auto buildNewAxisRefList = [buildNewAxisRefs,
                              mainMesh](ArrayRef<AxisRefAttr> oldAxisRefs) {
    AxisRefVector newAxisRefs;
    newAxisRefs.reserve(oldAxisRefs.size());
    llvm::transform(oldAxisRefs,
                    AddAxisOrMergeInserter(&newAxisRefs, &mainMesh),
                    buildNewAxisRefs);
    return newAxisRefs;
  };
  AxisRefVector newReplicatedAxes =
      buildNewAxisRefList(oldSharding.getReplicatedAxes());
  AxisRefVector newUnreducedAxes =
      buildNewAxisRefList(oldSharding.getUnreducedAxes());
  return TensorShardingAttr::get(context, mainMeshName, newDimShardings,
                                 newReplicatedAxes, newUnreducedAxes);
}

// Replaces the manual axes in `manualComputation`, if it refers to some mesh
// that isn't the main mesh saved in the pair of `MeshToAxisMap`, with the main
// mesh.
void replaceManualAxes(sdy::ManualComputationOp manualComputation,
                       const SymbolTable& symbolTable,
                       const MeshToAxisMap& duplicateMeshesToAxisMap) {
  if (manualComputation.getInShardings().empty() &&
      manualComputation.getOutShardings().empty()) {
    return;
  }
  mlir::Attribute meshOrRef = getCommonMeshOrRef(
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
  auto [mainMeshName, axisMap] = meshNameAndAxisMap->getSecond();
  MeshAttr mainMesh = sdy::getMeshOp(symbolTable, mainMeshName).getMesh();
  AxisRefVector newAxisRefs;
  newAxisRefs.reserve(manualComputation.getManualAxes().size());
  llvm::transform(manualComputation.getManualAxes(),
                  AddAxisOrMergeInserter(&newAxisRefs, &mainMesh),
                  [&](StringAttr manualAxis) {
                    return extractTargetAxisRefs(
                        axisMap.at(manualAxis.getValue()), mainMesh);
                  });
  SmallVector<StringAttr> newManualAxes;
  newManualAxes.reserve(newAxisRefs.size());
  llvm::transform(
      newAxisRefs, std::back_inserter(newManualAxes), [](AxisRefAttr axisRef) {
        CHECK(!axisRef.getSubAxisInfo())
            << "Manual sub-axis isn't supported. Please "
               "file a bug with a reproducer.";
        return StringAttr::get(axisRef.getContext(), axisRef.getName());
      });
  manualComputation.setManualAxesAttr(
      sdy::ManualAxesAttr::get(manualComputation.getContext(), newManualAxes));
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
        return replaceAxesInSharding(oldSharding, symbolTable,
                                     duplicateMeshesToAxisMap);
      },
      [&](mlir::Operation* op) {
        if (auto manualComputation =
                mlir::dyn_cast<sdy::ManualComputationOp>(op)) {
          replaceManualAxes(manualComputation, symbolTable,
                            duplicateMeshesToAxisMap);
        }
      });
}

void eraseMeshes(SymbolTable& symbolTable,
                 const MeshToAxisMap& duplicateMeshesToAxisMap) {
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
    SymbolTable symbolTable(moduleOp);

    // Exit early if there are no meshes or just one mesh.
    auto meshIter = moduleOp.getOps<MeshOp>();
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
