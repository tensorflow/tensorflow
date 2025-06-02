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

using AxisRefMap = SmallDenseMap<AxisRefAttr, AxisRefAttr>;
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

  // The core logic: assignment calls push_back
  AddAxisOrMergeInserter& operator=(const AxisRefAttr& value) {
    sdy::addAxisOrMerge(*axisRefs, value, *mesh);
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
                             AxisRefMap& targetToMainAxisMap,
                             mlir::MLIRContext* context) {
  ArrayRef<MeshAxisAttr> mainAxes = mainMesh.getMesh().getAxes();
  ArrayRef<MeshAxisAttr> targetAxes = targetMesh.getMesh().getAxes();
  int mainAxisIndex = 0, targetAxisIndex = 0;
  int64_t preSize = 1, remainingMainAxisSize = 1;
  while (mainAxisIndex < mainAxes.size() &&
         targetAxisIndex < targetAxes.size()) {
    const MeshAxisAttr mainAxis = mainAxes[mainAxisIndex];
    const MeshAxisAttr targetAxis = targetAxes[targetAxisIndex];
    if (mainAxis.getSize() == 1 && targetAxis.getSize() != 1) {
      mainAxisIndex++;
      continue;
    }
    // We can leave mainAxis of size 1 without being mapped.
    if (targetAxis.getSize() == 1 && mainAxis.getSize() != 1) {
      return false;
    }
    // The targetAxis of size 1 can only be mapped to a mainAxis of size 1.
    if (remainingMainAxisSize == 1) {
      remainingMainAxisSize = mainAxis.getSize();
    }
    if (remainingMainAxisSize % targetAxis.getSize() != 0) {
      return false;
    }
    const AxisRefAttr targetAxisRef =
        AxisRefAttr::get(context, targetAxis.getName());
    // Set SubAxisInfoAttr to null when target axis maps to an entire main axis.
    const AxisRefAttr mainAxisRef = AxisRefAttr::get(
        context, mainAxis.getName(),
        mainAxis.getSize() == targetAxis.getSize()
            ? nullptr
            : SubAxisInfoAttr::get(context, preSize, targetAxis.getSize()));
    targetToMainAxisMap[targetAxisRef] = mainAxisRef;
    preSize *= targetAxis.getSize();
    targetAxisIndex++;
    remainingMainAxisSize /= targetAxis.getSize();
    if (remainingMainAxisSize == 1) {
      preSize = 1;
      mainAxisIndex++;
    }
  }
  CHECK_EQ(remainingMainAxisSize, 1);
  return mainAxisIndex == mainAxes.size() &&
         targetAxisIndex == targetAxes.size();
}

// Maps a pair of <total number of devices, device_ids> to a list of meshes
// that share those properties. Total number of devices is used to distinguish
// default meshes.
using MeshIdentifierToMainMeshesMap =
    SmallDenseMap<MeshDeviceIdentifier, MeshOp, 4, MeshDeviceIdentifierInfo>;

// Maps a mesh name to the main mesh name that will be used to replace it, and a
// map of old axes to the (sub)axes in the main mesh.
using MeshToAxisMap =
    SmallDenseMap<StringRef, std::pair<StringRef, AxisRefMap>>;

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
    SmallDenseMap<AxisRefAttr, AxisRefAttr> targetToMainAxisMap;
    if (mapTargetAxesToMainAxes(targetMesh, mainMesh, targetToMainAxisMap,
                                moduleOp.getContext())) {
      duplicateMeshesToAxisMap.try_emplace(targetMesh.getSymName(),
                                           mainMesh.getSymName(),
                                           std::move(targetToMainAxisMap));
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
      moduleOp, [&](TensorShardingAttr oldSharding) -> TensorShardingAttr {
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
        MeshAttr newMesh =
            sdy::getMeshOp(SymbolTable(moduleOp), mainMeshName).getMesh();
        auto buildNewAxisRef = [&, &axisMap = axisMap](AxisRefAttr oldAxisRef) {
          // If the old axis had a sub-axis info, we need to look up the
          // corresponding main axis without sub-axis info, and build the new
          // axis info.
          if (oldAxisRef.getSubAxisInfo()) {
            AxisRefAttr mappedAxisRef =
                axisMap.at(AxisRefAttr::get(context, oldAxisRef.getName()));
            SubAxisInfoAttr newSubAxisInfo = SubAxisInfoAttr::get(
                context,
                mappedAxisRef.getSubAxisPreSize() *
                    oldAxisRef.getSubAxisInfo().getPreSize(),
                oldAxisRef.getSubAxisInfo().getSize());
            return AxisRefAttr::get(context, mappedAxisRef.getName(),
                                    newSubAxisInfo);
          }
          return axisMap.at(oldAxisRef);
        };
        SmallVector<DimensionShardingAttr> newDimShardings;
        newDimShardings.reserve(oldSharding.getDimShardings().size());
        for (DimensionShardingAttr oldDimSharding :
             oldSharding.getDimShardings()) {
          AxisRefVector newAxisRefs;
          newAxisRefs.reserve(oldDimSharding.getAxes().size());
          llvm::transform(oldDimSharding.getAxes(),
                          AddAxisOrMergeInserter(&newAxisRefs, &newMesh),
                          buildNewAxisRef);
          newDimShardings.push_back(DimensionShardingAttr::get(
              context, newAxisRefs, oldDimSharding.getIsClosed(),
              oldDimSharding.getPriority()));
        }
        auto buildNewAxisRefList =
            [buildNewAxisRef, newMesh](ArrayRef<AxisRefAttr> oldAxisRefs) {
              AxisRefVector newAxisRefs;
              newAxisRefs.reserve(oldAxisRefs.size());
              llvm::transform(oldAxisRefs,
                              AddAxisOrMergeInserter(&newAxisRefs, &newMesh),
                              buildNewAxisRef);
              return newAxisRefs;
            };
        AxisRefVector newReplicatedAxes =
            buildNewAxisRefList(oldSharding.getReplicatedAxes());
        AxisRefVector newUnreducedAxes =
            buildNewAxisRefList(oldSharding.getUnreducedAxes());
        return TensorShardingAttr::get(context, mainMeshName, newDimShardings,
                                       newReplicatedAxes, newUnreducedAxes);
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
