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
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
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

using ::llvm::DenseMapInfo;
using ::llvm::SmallDenseMap;
using ::mlir::ArrayRef;
using ::mlir::DenseMap;
using ::mlir::DenseSet;
using ::mlir::ModuleOp;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::SymbolTable;

namespace sdy = ::mlir::sdy;

using DeviceIdsMapInfo = DenseMapInfo<ArrayRef<int64_t>>;

// A mesh without the axis names. This is used as a key to find meshes with the
// same `axisSizes` and `deviceIds`, since a mesh with the same axis sizes and
// device IDs is equivalent to another mesh with the same axis names even if
// they have different axis names.
struct MeshWithUnamedAxes {
  SmallVector<int64_t> axisSizes;
  ArrayRef<int64_t> deviceIds;
};

// SmallVector is not hashable by default, so we need to define a custom
// hasher. This is safe here since each element isn't mutable, so the key
// is stable.
struct MeshWithUnamedAxesInfo : public DenseMapInfo<MeshWithUnamedAxes> {
  static inline MeshWithUnamedAxes getEmptyKey() {
    return {{}, DeviceIdsMapInfo::getEmptyKey()};
  }

  static inline MeshWithUnamedAxes getTombstoneKey() {
    return {{}, DeviceIdsMapInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const MeshWithUnamedAxes& inputs) {
    return llvm::hash_combine(llvm::hash_combine_range(inputs.axisSizes.begin(),
                                                       inputs.axisSizes.end()),
                              DeviceIdsMapInfo::getHashValue(inputs.deviceIds));
  }

  static bool isEqual(const MeshWithUnamedAxes& lhs,
                      const MeshWithUnamedAxes& rhs) {
    return lhs.axisSizes == rhs.axisSizes && lhs.deviceIds == rhs.deviceIds;
  }
};

// Maps a set vector of axis sizes to the main mesh with those matching axis
// sizes.
using MeshWithUnamedAxesToFirstMeshMap =
    SmallDenseMap<MeshWithUnamedAxes, sdy::MeshOp, 4, MeshWithUnamedAxesInfo>;
// Maps a mesh to the main mesh name that will be used (which has the same axis
// sizes that were in `AxisSizesToFirstMeshMap`) and a map of old axis name to
// the axis names in the main mesh.
using MeshToAxisMap =
    SmallDenseMap<StringRef, std::pair<StringRef, llvm::StringMap<StringRef>>>;

// Builds a map of meshes to the main mesh name that will be used (which has the
// same axis sizes that were in `AxisSizesToFirstMeshMap`) and a map of old axis
// name to the axis names in the main mesh.
//
// NOTE: the main mesh will not be saved as a key in the map, since it won't
// need to be replaced.
MeshToAxisMap buildDuplicateMeshesToAxisMap(ModuleOp moduleOp) {
  MeshToAxisMap duplicateMeshesToAxisMap;
  MeshWithUnamedAxesToFirstMeshMap meshWithUnamedAxesToFirstMeshMap;
  for (sdy::MeshOp meshOp : moduleOp.getOps<sdy::MeshOp>()) {
    SmallVector<int64_t> meshSizes;
    sdy::MeshAttr meshAttr = meshOp.getMeshAttr();
    meshSizes.reserve(meshAttr.getAxes().size());
    for (sdy::MeshAxisAttr axis : meshAttr.getAxes()) {
      meshSizes.push_back(axis.getSize());
    }
    if (meshSizes.empty() && !meshOp.getMesh().isMaximal()) {
      // This can happen for empty maximal meshes. Use {-1} for it since an
      // empty/tombstone value can't be used as a key in the map.
      meshSizes = {-1};
    }
    // NOTE: we don't allow an explicit iota list of device IDs as part of
    // verification. So we don't need to worry about an empty list of device IDs
    // and an iota list of device IDs being equivalent but different keys in the
    // map.
    auto [entries, inserted] = meshWithUnamedAxesToFirstMeshMap.try_emplace(
        MeshWithUnamedAxes{meshSizes, meshOp.getMesh().getDeviceIds()}, meshOp);
    if (inserted) {
      continue;
    }
    llvm::StringMap<StringRef> oldToNewAxis;
    sdy::MeshOp mainMesh = entries->getSecond();
    for (auto [oldAxis, newAxisName] :
         llvm::zip_equal(meshOp.getMeshAttr().getAxes(),
                         mainMesh.getMeshAttr().getAxes())) {
      oldToNewAxis[oldAxis.getName()] = newAxisName.getName();
    }
    duplicateMeshesToAxisMap.try_emplace(
        meshOp.getSymName(), mainMesh.getSymName(), std::move(oldToNewAxis));
  };
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
          return sdy::AxisRefAttr::get(context,
                                       axisMap.at(oldAxisRef.getName()),
                                       oldAxisRef.getSubAxisInfo());
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
