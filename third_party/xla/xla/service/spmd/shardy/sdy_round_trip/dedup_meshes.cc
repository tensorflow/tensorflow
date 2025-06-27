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

  AddAxisOrMergeInserter& operator=(ArrayRef<AxisRefAttr> values) {
    for (AxisRefAttr value : values) {
      sdy::addAxisOrMerge(*axisRefs, value, *mesh);
    }
    return *this;
  }

  AddAxisOrMergeInserter& operator*() { return *this; }
  AddAxisOrMergeInserter& operator++() { return *this; }
  AddAxisOrMergeInserter& operator++(int) { return *this; }

 private:
  // Use a pointer so that callers like `llvm::transform` can copy the inserter.
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

// Maps a mesh name to the main mesh name that will be used to replace it, and a
// map of old axes to the (sub)axes in the main mesh.
using MeshToAxisMap = SmallDenseMap<StringRef, std::pair<StringRef, AxisMap>>;

// Takes an old axis ref, find the mapped new (sub-)axes, add them to the new
// axis ref list and merge consecutive sub-axes.
void addOrMergeNewAxisRefAttr(AxisRefAttr oldAxisRef,
                              AxisRefVector& newAxisRefs,
                              mlir::MLIRContext* context,
                              const MeshAttr& mainMesh,
                              const AxisMap& axisMap) {
  if (!axisMap.contains(oldAxisRef.getName())) {
    // The old axis is of size 1, skip it.
    return;
  }
  ArrayRef<AxisRefAttr> mainAxisRefs = axisMap.at(oldAxisRef.getName());
  if (!oldAxisRef.getSubAxisInfo()) {
    for (AxisRefAttr mainAxisRef : mainAxisRefs) {
      sdy::addAxisOrMerge(newAxisRefs, mainAxisRef, mainMesh);
    }
    return;
  }
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
  SmallVector<DimensionShardingAttr> newDimShardings;
  newDimShardings.reserve(oldSharding.getDimShardings().size());
  for (DimensionShardingAttr oldDimSharding : oldSharding.getDimShardings()) {
    AxisRefVector newAxisRefs;
    newAxisRefs.reserve(oldDimSharding.getAxes().size());
    for (AxisRefAttr oldAxisRef : oldDimSharding.getAxes()) {
      addOrMergeNewAxisRefAttr(oldAxisRef, newAxisRefs, context, mainMesh,
                               axisMap);
    }
    newDimShardings.push_back(DimensionShardingAttr::get(
        context, newAxisRefs, oldDimSharding.getIsClosed(),
        oldDimSharding.getPriority()));
  }
  auto buildNewAxisRefList =
      [&, &axisMap = axisMap](ArrayRef<AxisRefAttr> oldAxisRefs) {
        AxisRefVector newAxisRefs;
        newAxisRefs.reserve(oldAxisRefs.size());
        for (AxisRefAttr oldAxisRef : oldAxisRefs) {
          addOrMergeNewAxisRefAttr(oldAxisRef, newAxisRefs, context, mainMesh,
                                   axisMap);
        }
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
                  [&axisMap = axisMap](StringAttr manualAxis) -> AxisRefVector {
                    if (!axisMap.contains(manualAxis.getValue())) {
                      return {};
                    }
                    return axisMap.at(manualAxis.getValue());
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
