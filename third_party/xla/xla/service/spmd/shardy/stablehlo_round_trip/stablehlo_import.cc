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

#include "xla/service/spmd/shardy/stablehlo_round_trip/stablehlo_import.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
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
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/round_trip_common/pipeline_passes.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/shard_map_import.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

namespace {

using ::llvm::SmallDenseMap;
using ::llvm::SmallDenseSet;
using ::mlir::ArrayRef;
using ::mlir::LogicalResult;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::ShapedType;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::func::FuncOp;

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingAttr;

// The information of a sub-dimension in IotaTileAssignment. One tile dimension
// in tile assignment may correspond to multiple sub-dimensions. See
// getOrderedSubDimsFromIotaTileAssignment() for an example.
struct SubDimInfo {
  int64_t tileDimIndex;     // The tile assignment dimension that this
                            // sub-dimension belongs to.
  int64_t tileSubDimIndex;  // The sub-dimension index, whose order is minor to
                            // major.
  int64_t reshapeDimIndex;  // The reshape dimension that this sub-dimension
                            // belongs to.
  int64_t size;             // The size of this sub-dimension.
};

struct AnalyzeTileAssignmentResult {
  SmallVector<SubDimInfo> subDims;
  SmallVector<int64_t> localMesh;
};

}  // namespace

// Parse the string `sharding` to obtain a `xla::HloSharding`.
xla::HloSharding parseShardingFromString(const StringAttr& sharding) {
  const std::optional<xla::OpSharding> shardingProto =
      xla::ConvertSharding(sharding.getValue());
  CHECK(shardingProto) << sharding.getValue().str();
  absl::StatusOr<HloSharding> hloSharding =
      xla::HloSharding::FromProto(*shardingProto);
  CHECK_OK(hloSharding) << shardingProto->DebugString();
  return *hloSharding;
}

namespace {

// Given a vector of integers, we can factorize its elements into a product of
// smaller factors. For example, with the input vector of [4, 8], we can
// decompose the element 4 into 2x2, and decompose the element 8 into 4x2. Thus,
// we have a factorization of [2, 2, 4, 2]. Other valid factorizations include
// [4, 4, 2], [2, 2, 8], [2, 2, 2, 2, 2], etc. We ignore the element 1 in the
// input vectors.
//
// This function finds the shortest common factorization of two input vectors.
// For example,
// 1. f([2, 3], [3, 2]) = empty vector since they have no common factorizations.
// 2. f([4, 8], [8, 4]) = [4, 2, 4]
// 3. f([4, 8], [4, 2, 4]) = [4, 2, 4]
// 4. f([4, 8], [4, 4, 2]) = [4, 4, 2]
// 5. f([2, 2, 2, 2, 2], [4, 4, 2]) = [2, 2, 2, 2, 2]
// 6. f([6, 1, 4], [2, 6, 2]) = [2, 3, 2, 2]. We skip the element 1.
SmallVector<int64_t> shortestCommonFactorization(ArrayRef<int64_t> array1,
                                                 ArrayRef<int64_t> array2) {
  SmallVector<int64_t> result;
  result.reserve(std::max(array1.size(), array2.size()));

  auto nextIndexWithNonOneElement = [](ArrayRef<int64_t> array,
                                       int64_t index) -> int64_t {
    while (index < array.size() && array[index] == 1) {
      index++;
    }
    return index;
  };

  int64_t index1 = nextIndexWithNonOneElement(array1, 0);
  int64_t index2 = nextIndexWithNonOneElement(array2, 0);
  int64_t nextStride1 = 1;
  int64_t nextStride2 = 1;
  int64_t accumulatedFactor = 1;

  while (index1 < array1.size() || index2 < array2.size()) {
    if (index1 < array1.size() && nextStride1 == accumulatedFactor) {
      nextStride1 *= array1[index1++];
    }
    if (index2 < array2.size() && nextStride2 == accumulatedFactor) {
      nextStride2 *= array2[index2++];
    }

    const auto [smallFactor, largeFactor] = std::minmax(
        {nextStride1 / accumulatedFactor, nextStride2 / accumulatedFactor});

    if (largeFactor % smallFactor != 0 || smallFactor == 1) {
      return {};
    }

    result.push_back(smallFactor);
    accumulatedFactor *= smallFactor;
    CHECK_EQ(accumulatedFactor, Product(result));

    index1 = nextIndexWithNonOneElement(array1, index1);
    index2 = nextIndexWithNonOneElement(array2, index2);
  }

  return result;
}

// Create ordered SubDimInfo based on IotaTileAssignment (dims, reshape_dims,
// and transpose_perm) in three steps.
// 1. Find shortest common factorization of (1) dims and (2) reshape_dims
// reordered by transpose_perm.
// 2. Construct the SubDimInfo based on the common factorization, which
// represents the common axes.
// 3. Sort the vector of SubDimInfo based on tuple (reshapeDimIndex,
// tileDimIndex).
//
// Take {devices=[6,35]<=[7,10,3]T(2,1,0)} as an example. It has
// * dims = [6,35]
// * reshape_dims = [7,10,3]
// * transpose_perm = [2,1,0].
// We find the shortest common factorization between (1) [6,35], and (2)
// [7,10,3] reordered by [2,1,0], which is [3,10,7]. The common factorization
// [3,2,5,7] is then used to create a sorted vector of SubDimInfo.
//
// index tileDimIndex tileSubDimIndex reshapeDimIndex size
//   0        1              0               0         7
//   1        0              0               1         2
//   2        1              1               1         5
//   3        0              1               2         3
//
// The 0-th sub-dimension with size 7 corresponds to dims[1] (which is 35), and
// reshape_dims[0] (which is 7). The dims[1] (which is 35) is decomposed into
// two sub-dimensions 7 and 5. Thus the sub-dimensions with size 7 and 5 share
// the same tileDimIndex of 1 and have different tileSubDimIndex 0 and 1. 7 is
// on the right of 5 in the common factorization [3,2,5,7] and the
// tileSubDimIndex is ordered minor to major.
//
// If the input is not compatible with the mesh-based sharding, the function
// returns an empty vector, such as {devices=[2,3]<=[2,3]T(1,0)}.
SmallVector<SubDimInfo> getOrderedSubDimsFromIotaTileAssignment(
    const xla::IotaTileAssignment& iota) {
  SmallVector<int64_t> deviceShape(iota.transpose_perm().size());
  for (auto [index, perm_i] : llvm::enumerate(iota.transpose_perm())) {
    deviceShape[index] = iota.reshape_dims()[perm_i];
  }

  const SmallVector<int64_t> axisSizes = shortestCommonFactorization(
      ArrayRef<int64_t>(iota.dims().begin(), iota.dims().end()), deviceShape);
  if (axisSizes.empty()) {
    return {};
  }

  SmallVector<SubDimInfo> subDims;
  subDims.reserve(axisSizes.size());

  int64_t tileDimIndex = iota.ndims() - 1;
  int64_t transPermIndex = iota.transpose_perm().size() - 1;
  int64_t accTileSize = 1;
  int64_t accDeviceSize = 1;
  int64_t subDim = 0;

  for (const int64_t axisSize : llvm::reverse(axisSizes)) {
    while (iota.dim(tileDimIndex) == 1) {
      tileDimIndex--;
    }
    subDims.push_back(SubDimInfo{
        /* .tileDimIndex = */ tileDimIndex,
        /* .tileSubDimIndex = */ subDim++,
        /* .reshapeDimIndex = */ iota.transpose_perm()[transPermIndex],
        /* .size = */ axisSize,
    });
    accTileSize *= axisSize;
    accDeviceSize *= axisSize;
    if (iota.dim(tileDimIndex) == accTileSize) {
      tileDimIndex--;
      accTileSize = 1;
      subDim = 0;
    }
    if (deviceShape[transPermIndex] == accDeviceSize) {
      accDeviceSize = 1;
      transPermIndex--;
    }
  }

  // We use sort instead of stable_sort, since the sub dimensions have different
  // tuple (reshapeDimIndex, tileDimIndex).
  absl::c_sort(subDims, [](const SubDimInfo& a, const SubDimInfo& b) {
    return std::forward_as_tuple(a.reshapeDimIndex, a.tileDimIndex) <
           std::forward_as_tuple(b.reshapeDimIndex, b.tileDimIndex);
  });
  return subDims;
}

// Analyze the input tile assignment to obtain the information on the mesh and
// sub dimensions.
AnalyzeTileAssignmentResult analyzeTileAssignment(
    const xla::TileAssignment& tileAssignment) {
  // If the input has iota tile assignment (the corresponding HloSharding is in
  // V2 format), we use getOrderedSubDimsFromIotaTileAssignment.
  // TODO(zixuanjiang). We may handle HloShardingV1 in the future.
  const std::optional<IotaTileAssignment>& iota = tileAssignment.iota();
  CHECK(iota.has_value()) << "tile assignment: " << tileAssignment.ToString();
  const SmallVector<SubDimInfo> subDims =
      getOrderedSubDimsFromIotaTileAssignment(*iota);

  // TODO(zixuanjiang). We cannot handle the sharding that needs to specify the
  // device list. For example, we cannot handle the V2 sharding
  // {devices=[2,3]<=[2,3]T(1,0)}, which is equivalent to
  // {devices=[2,3]0,3,1,4,2,5}.
  CHECK(!subDims.empty()) << "tile assignment: " << tileAssignment.ToString();

  SmallVector<int64_t> mesh;
  mesh.reserve(subDims.size());
  for (SubDimInfo subDimInfo : subDims) {
    mesh.push_back(subDimInfo.size);
  }
  return AnalyzeTileAssignmentResult{
      /* .subDims = */ std::move(subDims),
      /* .localMesh = */ std::move(mesh),
  };
}

// Collect shardings with the attr name kXlaShardingAttr in the `moduleOp`.
absl::flat_hash_set<xla::HloSharding> collectXlaHloShardings(
    ModuleOp moduleOp) {
  absl::flat_hash_set<xla::HloSharding> oldShardings;
  for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
    for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
      if (auto oldSharding =
              funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
        oldShardings.insert(parseShardingFromString(oldSharding));
      }
    }

    for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
      if (auto oldSharding = funcOp.getResultAttrOfType<StringAttr>(
              resNum, kXlaShardingAttr)) {
        oldShardings.insert(parseShardingFromString(oldSharding));
      }
    }

    funcOp.front().walk([&](mlir::Operation* op) {
      if (auto oldSharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
        const xla::HloSharding hloSharding =
            parseShardingFromString(oldSharding);
        if (hloSharding.IsTuple()) {
          for (const xla::HloSharding& element : hloSharding.tuple_elements()) {
            oldShardings.insert(element);
          }
        } else {
          oldShardings.insert(hloSharding);
        }
      }
    });
  }
  return oldShardings;
}

struct MeshAxesAndIds {
  SmallVector<MeshAxisAttr> namedAxes;
  // The device ids for maximal shardings sorted in ascending order
  // without duplicates.
  SmallVector<int64_t> maximalDeviceIds;
};

// Collect shardings with the attr name kXlaShardingAttr. Find common axes for
// these shardings and device ids for maximal shardings.
MeshAxesAndIds findMeshAxesAndIds(ModuleOp moduleOp) {
  MeshAxesAndIds result;
  auto& [namedAxes, maximalDeviceIds] = result;
  // 1. Collect old shardings in the format of xla::HloSharding.
  const absl::flat_hash_set<xla::HloSharding> oldShardings =
      collectXlaHloShardings(moduleOp);

  // 2. Find common axes of old shardings.
  SmallVector<int64_t> axes;
  llvm::SmallDenseSet<int64_t> maximalDeviceIdSet;
  for (const xla::HloSharding& hloSharding : oldShardings) {
    // If the sharding is a maximal sharding, we do not need to find common
    // axes but add the device id to `deviceIdsForMaximalMesh`.
    if (hloSharding.HasUniqueDevice()) {
      maximalDeviceIdSet.insert(hloSharding.GetUniqueDevice());
      continue;
    }

    CHECK(!hloSharding.IsTuple());
    if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
        hloSharding.IsUnknown()) {
      continue;
    }

    CHECK(hloSharding.IsTiled());
    const AnalyzeTileAssignmentResult result =
        analyzeTileAssignment(hloSharding.tile_assignment());

    axes = (axes.empty()) ? result.localMesh
                          : shortestCommonFactorization(result.localMesh, axes);
    CHECK(!axes.empty());
    // TODO(zixuanjiang). Support cases without common factorizations.
  }

  // 3. Create a mesh.
  namedAxes.reserve(axes.size());
  for (auto [axisIndex, axisSize] : llvm::enumerate(axes)) {
    auto name = StringAttr::get(moduleOp->getContext(),
                                absl::StrCat("axis_", axisIndex));
    namedAxes.push_back(
        MeshAxisAttr::get(moduleOp->getContext(), name, axisSize));
  }

  maximalDeviceIds = llvm::to_vector(maximalDeviceIdSet);
  llvm::sort(maximalDeviceIds);
  return result;
}

}  // namespace

// Convert the `hloSharding` into a `TensorShardingAttr` based on the
// `globalMesh`.
TensorShardingAttr convertToSdySharding(
    const xla::HloSharding& hloSharding, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToMaximalMeshName,
    int64_t rank, bool openDims) {
  mlir::MLIRContext* ctx = globalMesh.getContext();

  // If the sharding is a maximal sharding, return a fully closed sharding.
  // The exact sharding does not matter since the tensor can only exist on one
  // device.
  if (hloSharding.HasUniqueDevice()) {
    return TensorShardingAttr::getFullyClosed(
        ctx, /*rank=*/0,
        deviceIdToMaximalMeshName.lookup(hloSharding.GetUniqueDevice()));
  }
  CHECK(!hloSharding.IsTuple());

  if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
      hloSharding.IsUnknown()) {
    return hloSharding.IsUnknown() || openDims
               ? TensorShardingAttr::getFullyOpen(ctx, rank, kGlobalMeshName)
               : TensorShardingAttr::getFullyClosed(ctx, rank, kGlobalMeshName);
  }

  CHECK(hloSharding.IsTiled());
  const AnalyzeTileAssignmentResult result =
      analyzeTileAssignment(hloSharding.tile_assignment());

  // 1. Create a mapping from local axis to global axis refs.
  CHECK_EQ(Product(result.localMesh), globalMesh.getTotalSize());
  SmallVector<SmallVector<AxisRefAttr>> localAxisIndexToGlobalAxes;
  localAxisIndexToGlobalAxes.reserve(result.localMesh.size());
  ArrayRef<MeshAxisAttr> remainingGlobalAxes = globalMesh.getAxes();
  int64_t globalAxisPreSize = 1;
  for (int64_t localAxisRemainingSize : result.localMesh) {
    SmallVector<AxisRefAttr>& globalAxes =
        localAxisIndexToGlobalAxes.emplace_back();
    // The local axis size can correspond to multiple global axes or sub-axes.
    while (localAxisRemainingSize > 1) {
      CHECK(!remainingGlobalAxes.empty());
      int64_t globalAxisRemainingSize =
          remainingGlobalAxes.front().getSize() / globalAxisPreSize;
      if (globalAxisRemainingSize == 1) {
        remainingGlobalAxes = remainingGlobalAxes.drop_front();
        globalAxisPreSize = 1;
        continue;
      }
      int64_t gcd = std::gcd(localAxisRemainingSize, globalAxisRemainingSize);
      CHECK_NE(gcd, 1) << "Incompatible local and global axis sizes: "
                       << localAxisRemainingSize << " vs "
                       << globalAxisRemainingSize;
      StringRef globalAxisName = remainingGlobalAxes.front().getName();
      if (gcd == globalAxisRemainingSize && globalAxisPreSize == 1) {
        // The full global axis is used.
        globalAxes.push_back(AxisRefAttr::get(ctx, globalAxisName));
      } else {
        // We use a sub-axis of the global axis.
        globalAxes.push_back(
            AxisRefAttr::get(ctx, globalAxisName, globalAxisPreSize, gcd));
      }
      globalAxisPreSize *= gcd;
      localAxisRemainingSize /= gcd;
    }
  }

  // 2. Create a mapping from dim and nested sub-dim to local axis index.
  SmallVector<SmallVector<int64_t>> dimToSubDimToLocalAxisIndex(rank);
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(result.subDims)) {
    if (subDimInfo.tileDimIndex >= rank) {
      // This is the last tile dimension that is replicated.
      continue;
    }
    SmallVector<int64_t>& subDimToLocalAxisIndex =
        dimToSubDimToLocalAxisIndex[subDimInfo.tileDimIndex];
    if (subDimInfo.tileSubDimIndex >= subDimToLocalAxisIndex.size()) {
      subDimToLocalAxisIndex.resize(subDimInfo.tileSubDimIndex + 1);
    }
    subDimToLocalAxisIndex[subDimInfo.tileSubDimIndex] = localAxisIndex;
  }

  // 3. Finally, create the new sharding by flattening
  // `dimToSubDimToLocalAxisIndex` and replacing each local axis index with the
  // corresponding global axes in `localAxisIndexToGlobalAxes`.
  SmallVector<DimensionShardingAttr> dimShardings;
  dimShardings.reserve(rank);
  for (ArrayRef<int64_t> subDimToLocalAxisIndex : dimToSubDimToLocalAxisIndex) {
    SmallVector<AxisRefAttr> axes;
    // We iterate the sub dims in the reverse order since they are minor to
    // major.
    for (int64_t localAxisIndex : llvm::reverse(subDimToLocalAxisIndex)) {
      absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                   std::back_inserter(axes));
    }
    dimShardings.push_back(
        DimensionShardingAttr::get(ctx, axes, /*is_closed=*/!openDims));
  }
  return TensorShardingAttr::get(ctx, StringAttr::get(ctx, kGlobalMeshName),
                                 dimShardings, /*replicated_axes=*/{},
                                 /*unreduced_axes=*/{});
}

namespace {

bool shouldOpenDims(ArrayRef<bool> allowPropagationToTensors, int64_t index) {
  if (allowPropagationToTensors.empty()) {
    // Default to false for all tensors.
    return false;
  }
  if (allowPropagationToTensors.size() == 1) {
    // This means the only element in `allowPropagationToTensors` applies for
    // all tensors, i.e., any index.
    return allowPropagationToTensors.front();
  }
  // Otherwise, we assume that that `allowPropagationToTensors` has the same
  // size as the number of tensors.
  CHECK_LT(index, allowPropagationToTensors.size());
  return allowPropagationToTensors[index];
}

// Convert the shardings in `funcOp` from kXlaShardingAttr into kShardingAttr.
LogicalResult importShardings(
    FuncOp funcOp, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToMaximalMeshName,
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults) {
  for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
    if (auto oldSharding =
            funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
      funcOp.setArgAttr(
          argNum, kShardingAttr,
          convertToSdySharding(parseShardingFromString(oldSharding), globalMesh,
                               deviceIdToMaximalMeshName,
                               mlir::cast<ShapedType>(argType).getRank(),
                               shouldOpenDims(allowPropagationToArgs, argNum)));
      funcOp.removeArgAttr(argNum, kXlaShardingAttr);
    }
  }

  for (auto [resNum, resType] : llvm::enumerate(funcOp.getResultTypes())) {
    if (auto oldSharding =
            funcOp.getResultAttrOfType<StringAttr>(resNum, kXlaShardingAttr)) {
      funcOp.setResultAttr(
          resNum, kShardingAttr,
          convertToSdySharding(
              parseShardingFromString(oldSharding), globalMesh,
              deviceIdToMaximalMeshName,
              mlir::cast<ShapedType>(resType).getRank(),
              shouldOpenDims(allowPropagationToResults, resNum)));
      funcOp.removeResultAttr(
          resNum, StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
    }
  }

  funcOp.front().walk([&](mlir::Operation* op) {
    if (auto oldSharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
      const xla::HloSharding hloSharding = parseShardingFromString(oldSharding);
      ArrayRef<xla::HloSharding> flatHloSharding = hloSharding;
      if (hloSharding.IsTuple()) {
        flatHloSharding = hloSharding.tuple_elements();
      }
      SmallVector<TensorShardingAttr> newShardings;
      newShardings.reserve(op->getNumResults());
      for (const auto& [resHloSharding, resType] :
           llvm::zip_equal(flatHloSharding, op->getResultTypes())) {
        newShardings.push_back(convertToSdySharding(
            resHloSharding, globalMesh, deviceIdToMaximalMeshName,
            mlir::cast<ShapedType>(resType).getRank(),
            /*openDims=*/false));
      }
      mlir::sdy::setShardings(op, newShardings);
      op->removeAttr(kXlaShardingAttr);
    }
  });

  return mlir::success();
}

class ImportShardingsPass
    : public PassWrapper<ImportShardingsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportShardingsPass)

  ImportShardingsPass(ArrayRef<bool> allowPropagationToArgs,
                      ArrayRef<bool> allowPropagationToResults)
      : allowPropagationToArgs(allowPropagationToArgs),
        allowPropagationToResults(allowPropagationToResults) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    auto [namedAxes, deviceIdsForMaximalMesh] = findMeshAxesAndIds(moduleOp);
    if (namedAxes.empty() && deviceIdsForMaximalMesh.empty()) {
      // There are no shardings in the `moduleOp`.
      return;
    }

    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);

    OpBuilder opBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
    // Create a global mesh containing all the axes.
    symbolTable.insert(opBuilder.create<MeshOp>(
        moduleOp.getLoc(), kGlobalMeshName,
        MeshAttr::get(moduleOp.getContext(), namedAxes)));

    SmallDenseMap<int64_t, StringRef> deviceIdToMaximalMeshName;
    for (int64_t deviceId : deviceIdsForMaximalMesh) {
      // Create a mesh name with its deviceId as a suffix for each maximal mesh.
      std::string meshName = absl::StrCat("maximal_mesh_", deviceId);
      auto meshOp = opBuilder.create<MeshOp>(
          moduleOp.getLoc(), meshName,
          MeshAttr::get(moduleOp.getContext(), deviceId));
      symbolTable.insert(meshOp);
      deviceIdToMaximalMeshName[deviceId] = meshOp.getSymName();
    }

    for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
      // `allowPropagationToArgs` and `allowPropagationToResults` apply
      // only to the main/entry function.
      bool isMain = funcOp.getSymName() == "main";
      MeshAttr globalMesh = MeshAttr::get(moduleOp.getContext(), namedAxes);
      if (mlir::failed(importShardings(
              funcOp, globalMesh, deviceIdToMaximalMeshName,
              isMain ? allowPropagationToArgs : ArrayRef<bool>(),
              isMain ? allowPropagationToResults : ArrayRef<bool>()))) {
        signalPassFailure();
      }
    }
  }

  StringRef getArgument() const override { return "xla-sdy-import-shardings"; }

  StringRef getDescription() const override {
    return "Builds the mesh and converts the shardings from kXlaShardingAttr "
           "to kShardingAttr.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }

 private:
  ArrayRef<bool> allowPropagationToArgs;
  ArrayRef<bool> allowPropagationToResults;
};

std::unique_ptr<mlir::Pass> createImportShardingsPass(
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults) {
  return std::make_unique<ImportShardingsPass>(allowPropagationToArgs,
                                               allowPropagationToResults);
}

}  // namespace

void registerStablehloImportShardingsPass() {
  mlir::registerPass(
      std::bind(createImportShardingsPass, ArrayRef<bool>(), ArrayRef<bool>()));
}

void addStablehloImportPipeline(mlir::OpPassManager& pm,
                                ArrayRef<bool> allowPropagationToArgs,
                                ArrayRef<bool> allowPropagationToResults) {
  addCommonPreImportPasses(pm);
  pm.addPass(createImportShardingsPass(allowPropagationToArgs,
                                       allowPropagationToResults));
  pm.addPass(createStablehloRoundTripShardMapImportPass());
  addCommonPostImportPasses(pm);
}

void registerStablehloImportPipeline() {
  mlir::PassPipelineRegistration<> importPipeline(
      "xla-sdy-stablehlo-import-pipeline",
      "Run passes to import a StableHLO module with `mhlo.shardings` into the "
      "SDY (Shardy) dialect.",
      std::bind(addStablehloImportPipeline, std::placeholders::_1,
                ArrayRef<bool>(), ArrayRef<bool>()));
}

}  // namespace sdy
}  // namespace xla
