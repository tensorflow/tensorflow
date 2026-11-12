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

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/AsmParser/AsmParser.h"
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
#include "shardy/dialect/sdy/transforms/import/passes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/sdy_round_trip/pipelines.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/sharding_op_util.h"
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

using ::mlir::sdy::attributeToString;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

struct AnalyzedTileAssignment {
  xla::AnalyzeTileAssignmentResult info;
  std::vector<int64_t> deviceIds;
};

AnalyzedTileAssignment analyzeTileAssignmentWithDeviceIds(
    const TileAssignment& tileAssignment) {
  if (!tileAssignment.iota().has_value() ||
      xla::GetOrderedSubDimsFromIotaTileAssignment(*tileAssignment.iota())
          .has_value()) {
    if (std::optional<xla::AnalyzeTileAssignmentResult> res =
            xla::AnalyzeTileAssignment(tileAssignment);
        res.has_value()) {
      return AnalyzedTileAssignment{std::move(*res), {}};
    }
  }
  xla::AnalyzeTileAssignmentResult fallback;
  for (int64_t i = 0; i < tileAssignment.num_dimensions(); ++i) {
    int64_t dimSize = tileAssignment.dim(i);
    if (dimSize <= 1) {
      continue;
    }
    int64_t reshapeIdx = fallback.local_mesh.size();
    fallback.local_mesh.push_back(dimSize);
    fallback.sub_dims.push_back(xla::SubDimInfo{
        /*tile_dim_index=*/i,
        /*tile_sub_dim_index=*/0,
        /*reshape_dim_index=*/reshapeIdx,
        /*size=*/dimSize,
    });
  }
  std::vector<int64_t> deviceIds(tileAssignment.array().begin(),
                                 tileAssignment.array().end());
  if (llvm::is_sorted(deviceIds)) {
    deviceIds.clear();
  }
  return AnalyzedTileAssignment{std::move(fallback), std::move(deviceIds)};
}

bool opHasSdyFrontendSharding(mlir::Operation* op) {
  const StringRef sdyShardingAttrName(
      xla::HloSharding::kShardingFrontendAttrName.data(),
      xla::HloSharding::kShardingFrontendAttrName.size());
  mlir::DictionaryAttr dictAttr = getFrontendAttrs(op);
  if (hasKey(dictAttr, sdyShardingAttrName) || hasKey(dictAttr, kInShardings) ||
      hasKey(dictAttr, kOutShardings) || hasKey(dictAttr, kManualAxes)) {
    return true;
  }
  if (auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
    StringRef target = customCall.getCallTargetName();
    if (target == kFuncResultShardingTargetName ||
        target == kGlobalToLocalShapeCallTargetName ||
        target == kLocalToGlobalShapeCallTargetName) {
      return true;
    }
  }
  if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op);
      callOp && isManualComputation(callOp)) {
    return true;
  }
  return false;
}

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

// Collect shardings with the attr name kXlaShardingAttr in the `moduleOp`.
absl::flat_hash_set<xla::HloSharding> collectXlaHloShardings(
    ModuleOp moduleOp) {
  absl::flat_hash_set<xla::HloSharding> oldShardings;
  mlir::DictionaryAttr moduleDict = getFrontendAttrs(moduleOp);
  const bool hasInTupleSdy = hasKey(moduleDict, kInTupleShardings);
  const bool hasOutTupleSdy = hasKey(moduleDict, kOutTupleShardings);
  const StringRef sdyShardingAttrName(
      xla::HloSharding::kShardingFrontendAttrName.data(),
      xla::HloSharding::kShardingFrontendAttrName.size());
  for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
    if (funcOp.getSymName().contains(kManualComputationFuncName)) {
      continue;
    }
    const bool isMain = funcOp.getSymName() == "main";
    if (!(isMain && hasInTupleSdy)) {
      for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
        if (hasKey(getFuncArgFrontendAttrs(funcOp, argNum),
                   sdyShardingAttrName)) {
          continue;
        }
        if (auto oldSharding =
                funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
          oldShardings.insert(parseShardingFromString(oldSharding));
        }
      }
    }

    bool hasFuncResultSdy = isMain && hasOutTupleSdy;
    if (!funcOp.isExternal()) {
      funcOp.front().walk([&](mlir::stablehlo::CustomCallOp customCall) {
        if (customCall.getCallTargetName() == kFuncResultShardingTargetName) {
          hasFuncResultSdy = true;
        }
      });
    }
    if (!hasFuncResultSdy) {
      for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
        if (auto oldSharding = funcOp.getResultAttrOfType<StringAttr>(
                resNum, kXlaShardingAttr)) {
          oldShardings.insert(parseShardingFromString(oldSharding));
        }
      }
    }

    if (funcOp.isExternal()) {
      continue;
    }
    funcOp.front().walk([&](mlir::Operation* op) {
      if (opHasSdyFrontendSharding(op) ||
          mlir::isa<mlir::mhlo::AsyncStartOp, mlir::mhlo::AsyncUpdateOp,
                    mlir::mhlo::AsyncDoneOp>(op) ||
          llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
            return mlir::isa<mlir::TupleType>(t);
          })) {
        return;
      }
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
  // The device ids for single-device shardings sorted in ascending order
  // without duplicates.
  SmallVector<int64_t> singleDeviceIds;
  std::vector<int64_t> meshDeviceIds;
};

MeshAxesAndIds findMeshAxesAndIds(
    const absl::flat_hash_set<xla::HloSharding>& oldShardings,
    mlir::MLIRContext* context, int64_t numPartitions = 1) {
  MeshAxesAndIds result;
  auto& [namedAxes, singleDeviceIds, meshDeviceIds] = result;

  const xla::NamedSharding* firstNamedSharding = nullptr;
  for (const xla::HloSharding& hloSharding : oldShardings) {
    if (hloSharding.UseNamedShardingLeaf()) {
      firstNamedSharding = &hloSharding.named_sharding();
      break;
    }
  }
  if (firstNamedSharding != nullptr) {
    const xla::Mesh& mesh = firstNamedSharding->mesh();
    namedAxes.reserve(mesh.num_axes());
    meshDeviceIds.reserve(mesh.device_assignment().array().num_elements());
    if (mesh.num_axes() > 0) {
      for (int64_t i = 0; i < mesh.num_axes(); ++i) {
        auto name = StringAttr::get(context, mesh.axis_names()[i]);
        namedAxes.push_back(
            MeshAxisAttr::get(context, name, mesh.axis_size(i)));
      }
      for (int64_t device_id : mesh.device_assignment().array()) {
        meshDeviceIds.push_back(device_id);
      }
      if (llvm::is_sorted(meshDeviceIds)) {
        meshDeviceIds.clear();
      }
    }
  }

  // Find common axes of old shardings.
  SmallVector<int64_t> axes;
  llvm::SmallDenseSet<int64_t> singleDeviceIdSet;
  bool hasNonSingleDeviceSharding = false;
  for (const xla::HloSharding& hloSharding : oldShardings) {
    // If the sharding is a single-device sharding, we do not need to find
    // common axes but add the device id to `deviceIdsForSingleDeviceMesh`.
    if (hloSharding.IsSingleDevice()) {
      singleDeviceIdSet.insert(hloSharding.GetUniqueDevice());
      continue;
    }
    hasNonSingleDeviceSharding = true;

    if (hloSharding.UseNamedShardingLeaf()) {
      continue;
    }

    CHECK(!hloSharding.IsTuple());
    if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
        hloSharding.IsUnknown()) {
      continue;
    }

    CHECK(hloSharding.IsTiled());
    AnalyzedTileAssignment analyzed =
        analyzeTileAssignmentWithDeviceIds(hloSharding.tile_assignment());

    if (axes.empty()) {
      axes.assign(analyzed.info.local_mesh.begin(),
                  analyzed.info.local_mesh.end());
      meshDeviceIds = std::move(analyzed.deviceIds);
    } else if (analyzed.deviceIds == meshDeviceIds) {
      std::vector<int64_t> commonFactors = xla::ExtractCommonFactorSequence(
          absl::MakeConstSpan(analyzed.info.local_mesh),
          absl::MakeConstSpan(axes));
      if (!commonFactors.empty()) {
        axes.assign(commonFactors.begin(), commonFactors.end());
      }
    }
  }

  if (namedAxes.empty() && axes.empty() && hasNonSingleDeviceSharding &&
      numPartitions > 1) {
    axes.push_back(numPartitions);
  }

  // Create a mesh with fake axis names that starts with and underscore so that
  // we can replace the fake axis names with real axis names later.
  if (namedAxes.empty()) {
    namedAxes.reserve(axes.size());
    for (auto [axisIndex, axisSize] : llvm::enumerate(axes)) {
      auto name = StringAttr::get(context, absl::StrCat("_axis_", axisIndex));
      namedAxes.push_back(MeshAxisAttr::get(context, name, axisSize));
    }
  }

  singleDeviceIds = llvm::to_vector(singleDeviceIdSet);
  llvm::sort(singleDeviceIds);
  return result;
}

// Collect shardings with the attr name kXlaShardingAttr. Find common axes for
// these shardings and device ids for single-device shardings.
MeshAxesAndIds findMeshAxesAndIds(ModuleOp moduleOp, int64_t numPartitions) {
  // Collect old shardings in the format of xla::HloSharding.
  const absl::flat_hash_set<xla::HloSharding> oldShardings =
      collectXlaHloShardings(moduleOp);
  return findMeshAxesAndIds(oldShardings, moduleOp->getContext(),
                            numPartitions);
}

struct ResolvedTargetMesh {
  MeshAttr targetMesh;
  SmallVector<SmallVector<AxisRefAttr>> localAxisIndexToGlobalAxes;
  bool forceInlineMesh;
};

ResolvedTargetMesh resolveTargetMeshAndAxes(
    const AnalyzedTileAssignment& analyzed, MeshAttr globalMesh,
    bool inlineMesh) {
  mlir::MLIRContext* ctx = globalMesh.getContext();
  const xla::AnalyzeTileAssignmentResult& result = analyzed.info;

  if (ArrayRef<int64_t>(analyzed.deviceIds) == globalMesh.getDeviceIds() &&
      Product(result.local_mesh) == globalMesh.getTotalSize()) {
    SmallVector<SmallVector<AxisRefAttr>> localAxisIndexToGlobalAxes;
    localAxisIndexToGlobalAxes.reserve(result.local_mesh.size());
    ArrayRef<MeshAxisAttr> remainingGlobalAxes = globalMesh.getAxes();
    int64_t globalAxisPreSize = 1;
    bool compatible = true;
    for (int64_t localAxisRemainingSize : result.local_mesh) {
      SmallVector<AxisRefAttr>& globalAxes =
          localAxisIndexToGlobalAxes.emplace_back();
      while (localAxisRemainingSize > 1) {
        if (remainingGlobalAxes.empty()) {
          compatible = false;
          break;
        }
        int64_t globalAxisRemainingSize =
            remainingGlobalAxes.front().getSize() / globalAxisPreSize;
        if (globalAxisRemainingSize == 1) {
          remainingGlobalAxes = remainingGlobalAxes.drop_front();
          globalAxisPreSize = 1;
          continue;
        }
        int64_t gcd = std::gcd(localAxisRemainingSize, globalAxisRemainingSize);
        if (gcd == 1) {
          compatible = false;
          break;
        }
        StringRef globalAxisName = remainingGlobalAxes.front().getName();
        if (gcd == globalAxisRemainingSize && globalAxisPreSize == 1) {
          globalAxes.push_back(AxisRefAttr::get(ctx, globalAxisName));
        } else {
          globalAxes.push_back(
              AxisRefAttr::get(ctx, globalAxisName, globalAxisPreSize, gcd));
        }
        globalAxisPreSize *= gcd;
        localAxisRemainingSize /= gcd;
      }
      if (!compatible) {
        break;
      }
    }
    if (compatible) {
      return ResolvedTargetMesh{
          globalMesh, std::move(localAxisIndexToGlobalAxes), inlineMesh};
    }
  }

  SmallVector<MeshAxisAttr> localAxes;
  localAxes.reserve(result.local_mesh.size());
  SmallVector<SmallVector<AxisRefAttr>> localAxisIndexToGlobalAxes;
  localAxisIndexToGlobalAxes.reserve(result.local_mesh.size());
  for (auto [axisIdx, axisSize] : llvm::enumerate(result.local_mesh)) {
    auto axisName = StringAttr::get(ctx, absl::StrCat("_axis_", axisIdx));
    localAxes.push_back(MeshAxisAttr::get(ctx, axisName, axisSize));
    if (axisSize > 1) {
      localAxisIndexToGlobalAxes.push_back({AxisRefAttr::get(ctx, axisName)});
    } else {
      localAxisIndexToGlobalAxes.emplace_back();
    }
  }
  MeshAttr targetMesh = analyzed.deviceIds.empty()
                            ? MeshAttr::get(ctx, localAxes)
                            : MeshAttr::get(ctx, localAxes, analyzed.deviceIds);
  return ResolvedTargetMesh{targetMesh, std::move(localAxisIndexToGlobalAxes),
                            /*forceInlineMesh=*/true};
}

}  // namespace

SmallVector<int64_t> getAxisSizes(const TileAssignment& tileAssignment) {
  AnalyzedTileAssignment analyzed =
      analyzeTileAssignmentWithDeviceIds(tileAssignment);
  return SmallVector<int64_t>(analyzed.info.local_mesh.begin(),
                              analyzed.info.local_mesh.end());
}

std::string convertToSdySharding(const xla::OpSharding& opSharding,
                                 xla::Shape shape, bool openDims,
                                 bool inlineMesh, bool isSingleArg) {
  mlir::MLIRContext context;
  context.loadDialect<SdyDialect>();

  absl::StatusOr<xla::HloSharding> hloSharding =
      xla::HloSharding::FromProto(opSharding);
  CHECK_OK(hloSharding) << "Failed to parse sharding: "
                        << opSharding.DebugString();

  if (hloSharding->IsTuple()) {
    CHECK(shape.IsTuple());
    SmallVector<TensorShardingAttr> sdyShardings;
    auto [namedAxes, _, meshDeviceIds] =
        findMeshAxesAndIds(absl::flat_hash_set<xla::HloSharding>(
                               hloSharding->tuple_elements().begin(),
                               hloSharding->tuple_elements().end()),
                           &context);
    MeshAttr meshAttr = meshDeviceIds.empty()
                            ? MeshAttr::get(&context, namedAxes)
                            : MeshAttr::get(&context, namedAxes, meshDeviceIds);
    for (auto [indexedShape, elementSharding] :
         llvm::zip_equal(xla::ShapeUtil::GetLeafShapes(shape),
                         hloSharding->tuple_elements())) {
      sdyShardings.push_back(convertToSdySharding(
          elementSharding, meshAttr,
          llvm::SmallDenseMap<int64_t, mlir::StringRef>(),
          indexedShape.shape.dimensions().size(), openDims, inlineMesh));
    }

    return mlir::sdy::attributeToString(
        TensorShardingPerValueAttr::get(&context, sdyShardings));
  }

  auto [namedAxes, _, meshDeviceIds] =
      findMeshAxesAndIds({*hloSharding}, &context);
  MeshAttr meshAttr = meshDeviceIds.empty()
                          ? MeshAttr::get(&context, namedAxes)
                          : MeshAttr::get(&context, namedAxes, meshDeviceIds);
  mlir::sdy::TensorShardingAttr sdySharding = convertToSdySharding(
      *hloSharding, meshAttr, llvm::SmallDenseMap<int64_t, mlir::StringRef>(),
      shape.dimensions().size(), openDims, inlineMesh);
  return mlir::sdy::attributeToString(
      isSingleArg ? static_cast<mlir::Attribute>(sdySharding)
                  : TensorShardingPerValueAttr::get(&context, sdySharding));
}

// Convert the `hloSharding` into a `TensorShardingAttr` based on the
// `globalMesh`.
TensorShardingAttr convertToSdySharding(
    const xla::HloSharding& hloSharding, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToSingleDeviceMeshName,
    int64_t rank, bool openDims, bool inlineMesh) {
  mlir::MLIRContext* ctx = globalMesh.getContext();

  if (hloSharding.UseNamedShardingLeaf()) {
    const xla::NamedSharding& namedSharding = hloSharding.named_sharding();
    SmallVector<DimensionShardingAttr, 4> dimShardings;
    if (namedSharding.dim_shardings().empty() &&
        (namedSharding.IsReplicated() || namedSharding.IsManual() ||
         namedSharding.IsUnreduced())) {
      dimShardings.assign(
          rank, DimensionShardingAttr::get(ctx, {}, /*isClosed=*/!openDims));
    } else {
      for (const auto& dimSharding : namedSharding.dim_shardings()) {
        SmallVector<AxisRefAttr, 4> axes;
        for (const auto& axis : dimSharding.axes()) {
          int64_t index = axis.mesh_axis_index();
          if (index < globalMesh.getAxes().size()) {
            StringRef axisName = globalMesh.getAxes()[index].getName();
            if (axis.sub_axis_info().has_value()) {
              axes.push_back(AxisRefAttr::get(ctx, axisName, axis.pre_size(),
                                              axis.size(namedSharding.mesh())));
            } else {
              axes.push_back(AxisRefAttr::get(ctx, axisName));
            }
          }
        }
        dimShardings.push_back(DimensionShardingAttr::get(
            ctx, axes, dimSharding.is_closed() || !openDims));
      }
    }

    SmallVector<AxisRefAttr, 4> replicatedAxes;
    for (const xla::AxisRef& axis : namedSharding.replicated_axes()) {
      int64_t index = axis.mesh_axis_index();
      if (index < globalMesh.getAxes().size()) {
        StringRef axisName = globalMesh.getAxes()[index].getName();
        if (axis.sub_axis_info().has_value()) {
          replicatedAxes.push_back(AxisRefAttr::get(
              ctx, axisName, axis.pre_size(), axis.size(namedSharding.mesh())));
        } else {
          replicatedAxes.push_back(AxisRefAttr::get(ctx, axisName));
        }
      }
    }

    SmallVector<AxisRefAttr, 4> unreducedAxes;
    for (const xla::AxisRef& axis : namedSharding.unreduced_axes()) {
      int64_t index = axis.mesh_axis_index();
      if (index < globalMesh.getAxes().size()) {
        StringRef axisName = globalMesh.getAxes()[index].getName();
        if (axis.sub_axis_info().has_value()) {
          unreducedAxes.push_back(AxisRefAttr::get(
              ctx, axisName, axis.pre_size(), axis.size(namedSharding.mesh())));
        } else {
          unreducedAxes.push_back(AxisRefAttr::get(ctx, axisName));
        }
      }
    }

    mlir::sdy::ReductionOp reductionOp = mlir::sdy::ReductionOp::SUM;
    switch (namedSharding.reduction_op()) {
      case xla::ReductionOp::kSum:
        reductionOp = mlir::sdy::ReductionOp::SUM;
        break;
      case xla::ReductionOp::kMax:
        reductionOp = mlir::sdy::ReductionOp::MAX;
        break;
      case xla::ReductionOp::kMin:
        reductionOp = mlir::sdy::ReductionOp::MIN;
        break;
    }

    if (inlineMesh) {
      return TensorShardingAttr::get(ctx, globalMesh, dimShardings,
                                     replicatedAxes, unreducedAxes,
                                     reductionOp);
    }
    return TensorShardingAttr::get(ctx, StringAttr::get(ctx, kGlobalMeshName),
                                   dimShardings, replicatedAxes, unreducedAxes,
                                   reductionOp);
  }

  // If the sharding is a single-device sharding, return a fully closed
  // sharding. The exact sharding does not matter since the tensor can only
  // exist on one device.
  if (hloSharding.IsSingleDevice()) {
    if (inlineMesh) {
      return TensorShardingAttr::getFullyClosed(
          ctx, /*rank=*/0,
          MeshAttr::getSingleDevice(ctx, hloSharding.GetUniqueDevice()));
    }
    return TensorShardingAttr::getFullyClosed(
        ctx, /*rank=*/0,
        deviceIdToSingleDeviceMeshName.lookup(hloSharding.GetUniqueDevice()));
  }
  CHECK(!hloSharding.IsTuple());

  if (hloSharding.IsReplicated() || hloSharding.IsManual() ||
      hloSharding.IsUnknown()) {
    mlir::sdy::ReductionOp reductionOp = mlir::sdy::ReductionOp::SUM;
    switch (hloSharding.reduction_op()) {
      case xla::ReductionOp::kSum:
        reductionOp = mlir::sdy::ReductionOp::SUM;
        break;
      case xla::ReductionOp::kMax:
        reductionOp = mlir::sdy::ReductionOp::MAX;
        break;
      case xla::ReductionOp::kMin:
        reductionOp = mlir::sdy::ReductionOp::MIN;
        break;
    }
    SmallVector<DimensionShardingAttr> dimShardings(
        rank,
        DimensionShardingAttr::get(
            ctx, {},
            !hloSharding.IsUnknown() && (!openDims || hloSharding.IsManual())));
    if (inlineMesh) {
      return TensorShardingAttr::get(ctx, globalMesh, dimShardings,
                                     /*replicatedAxes=*/{},
                                     /*unreducedAxes=*/{}, reductionOp);
    }
    return TensorShardingAttr::get(ctx, StringAttr::get(ctx, kGlobalMeshName),
                                   dimShardings, /*replicatedAxes=*/{},
                                   /*unreducedAxes=*/{}, reductionOp);
  }

  CHECK(hloSharding.IsTiled());
  AnalyzedTileAssignment analyzed =
      analyzeTileAssignmentWithDeviceIds(hloSharding.tile_assignment());
  const xla::AnalyzeTileAssignmentResult* result = &analyzed.info;

  auto [targetMesh, localAxisIndexToGlobalAxes, forceInlineMesh] =
      resolveTargetMeshAndAxes(analyzed, globalMesh, inlineMesh);

  // 2. Create a mapping from dim and nested sub-dim to local axis index.
  SmallVector<SmallVector<int64_t>> dimToSubDimToLocalAxisIndex(rank);
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(result->sub_dims)) {
    if (subDimInfo.tile_dim_index >= rank) {
      // This is the last tile dimension that is replicated.
      continue;
    }
    SmallVector<int64_t>& subDimToLocalAxisIndex =
        dimToSubDimToLocalAxisIndex[subDimInfo.tile_dim_index];
    if (subDimInfo.tile_sub_dim_index >= subDimToLocalAxisIndex.size()) {
      subDimToLocalAxisIndex.resize(subDimInfo.tile_sub_dim_index + 1);
    }
    subDimToLocalAxisIndex[subDimInfo.tile_sub_dim_index] = localAxisIndex;
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

  SmallVector<AxisRefAttr> replicatedAxes;
  SmallVector<AxisRefAttr> unreducedAxes;
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(result->sub_dims)) {
    if (subDimInfo.tile_dim_index >= rank) {
      int64_t subgroupIndex = subDimInfo.tile_dim_index - rank;
      if (subgroupIndex <
          static_cast<int64_t>(hloSharding.subgroup_types().size())) {
        auto subgroupType = hloSharding.subgroup_types()[subgroupIndex];
        if (subgroupType == xla::OpSharding::REPLICATED) {
          absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                       std::back_inserter(replicatedAxes));
        } else if (subgroupType == xla::OpSharding::UNREDUCED) {
          absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                       std::back_inserter(unreducedAxes));
        }
      }
    }
  }
  if (!replicatedAxes.empty()) {
    mlir::sdy::sortAndMergeAxes(replicatedAxes, targetMesh);
  }
  if (!unreducedAxes.empty()) {
    mlir::sdy::sortAndMergeAxes(unreducedAxes, targetMesh);
  }

  mlir::sdy::ReductionOp reductionOp = mlir::sdy::ReductionOp::SUM;
  if (!unreducedAxes.empty()) {
    switch (hloSharding.reduction_op()) {
      case xla::ReductionOp::kSum:
        reductionOp = mlir::sdy::ReductionOp::SUM;
        break;
      case xla::ReductionOp::kMax:
        reductionOp = mlir::sdy::ReductionOp::MAX;
        break;
      case xla::ReductionOp::kMin:
        reductionOp = mlir::sdy::ReductionOp::MIN;
        break;
    }
  }

  if (forceInlineMesh) {
    return TensorShardingAttr::get(ctx, targetMesh, dimShardings,
                                   replicatedAxes, unreducedAxes, reductionOp);
  }
  return TensorShardingAttr::get(ctx, StringAttr::get(ctx, kGlobalMeshName),
                                 dimShardings, replicatedAxes, unreducedAxes,
                                 reductionOp);
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

struct ExtractedManualAxes {
  ManualAxesAttr manualAxes;
  MeshAttr mesh;
};

ExtractedManualAxes extractManualAxesFromHloSharding(
    const xla::HloSharding& hloSharding, MeshAttr globalMesh) {
  mlir::MLIRContext* ctx = globalMesh.getContext();
  if (hloSharding.IsTuple()) {
    ExtractedManualAxes common;
    for (const xla::HloSharding& elem : hloSharding.tuple_elements()) {
      if (elem.IsSingleDevice()) {
        continue;
      }
      ExtractedManualAxes extracted =
          extractManualAxesFromHloSharding(elem, globalMesh);
      if (!extracted.manualAxes) {
        return {};
      }
      if (!common.manualAxes) {
        common = extracted;
      } else if (common.manualAxes != extracted.manualAxes ||
                 common.mesh != extracted.mesh) {
        return {};
      }
    }
    return common;
  }
  if (hloSharding.UseNamedShardingLeaf()) {
    const xla::NamedSharding& namedSharding = hloSharding.named_sharding();
    if (namedSharding.manual_axes().empty()) {
      return {};
    }
    absl::Span<const std::string> axisNames = namedSharding.mesh().axis_names();
    SmallVector<AxisRefAttr> manualAxes;
    manualAxes.reserve(namedSharding.manual_axes().size());
    for (const xla::AxisRef& axisRef : namedSharding.manual_axes()) {
      StringRef name = axisNames[axisRef.mesh_axis_index()];
      if (axisRef.sub_axis_info().has_value()) {
        manualAxes.push_back(AxisRefAttr::get(ctx, name,
                                              axisRef.sub_axis_info()->pre_size,
                                              axisRef.sub_axis_info()->size));
      } else {
        manualAxes.push_back(AxisRefAttr::get(ctx, name));
      }
    }
    return {ManualAxesAttr::get(ctx, manualAxes),
            toSdyMeshAttr(namedSharding.mesh(), ctx)};
  }
  if (hloSharding.IsManual()) {
    if (globalMesh.getAxes().empty()) {
      return {};
    }
    SmallVector<AxisRefAttr> manualAxes;
    manualAxes.reserve(globalMesh.getAxes().size());
    for (MeshAxisAttr axis : globalMesh.getAxes()) {
      manualAxes.push_back(AxisRefAttr::get(ctx, axis.getName()));
    }
    return {ManualAxesAttr::get(ctx, manualAxes), globalMesh};
  }
  if (!hloSharding.IsTiled() || !hloSharding.IsManualSubgroup()) {
    return {};
  }

  AnalyzedTileAssignment analyzed =
      analyzeTileAssignmentWithDeviceIds(hloSharding.tile_assignment());
  const xla::AnalyzeTileAssignmentResult* result = &analyzed.info;
  auto [targetMesh, localAxisIndexToGlobalAxes, _] =
      resolveTargetMeshAndAxes(analyzed, globalMesh, /*inlineMesh=*/false);

  const int64_t rank = hloSharding.TiledDataRank();
  SmallVector<AxisRefAttr> manualAxes;
  for (auto [localAxisIndex, subDimInfo] : llvm::enumerate(result->sub_dims)) {
    if (subDimInfo.tile_dim_index >= rank) {
      int64_t subgroupIndex = subDimInfo.tile_dim_index - rank;
      if (subgroupIndex <
              static_cast<int64_t>(hloSharding.subgroup_types().size()) &&
          hloSharding.subgroup_types()[subgroupIndex] ==
              xla::OpSharding::MANUAL) {
        absl::c_copy(localAxisIndexToGlobalAxes[localAxisIndex],
                     std::back_inserter(manualAxes));
      }
    }
  }
  if (manualAxes.empty()) {
    return {};
  }
  mlir::sdy::sortAndMergeAxes(manualAxes, targetMesh);
  return {ManualAxesAttr::get(ctx, manualAxes), targetMesh};
}

void propagateInlineManualShardings(ModuleOp moduleOp, FuncOp funcOp,
                                    MeshAttr globalMesh) {
  if (funcOp.isExternal() || globalMesh.getAxes().empty()) {
    return;
  }
  mlir::MLIRContext* ctx = funcOp.getContext();

  struct ManualInfo {
    ManualAxesAttr manualAxes;
    mlir::Attribute meshOrRef;
    MeshAttr mesh;

    explicit operator bool() const {
      return manualAxes && !manualAxes.empty() && meshOrRef && mesh;
    }
    bool operator==(const ManualInfo& other) const {
      return manualAxes == other.manualAxes && meshOrRef == other.meshOrRef;
    }
  };

  auto resolveManualInfo = [&](ManualAxesAttr manualAxes,
                               TensorShardingAttr sharding) -> ManualInfo {
    if (!manualAxes || manualAxes.empty() || !sharding) {
      return {};
    }
    mlir::Attribute meshOrRef = sharding.getMeshOrRef();
    MeshAttr mesh = mlir::isa<MeshAttr>(meshOrRef)
                        ? mlir::cast<MeshAttr>(meshOrRef)
                        : sharding.getMesh(moduleOp);
    if (!mesh || mesh.getAxes().empty()) {
      return {};
    }
    return ManualInfo{manualAxes, meshOrRef, mesh};
  };

  auto getOpManualInfo = [&](mlir::Operation* op) -> ManualInfo {
    ManualAxesAttr manualAxes = op->getAttrOfType<ManualAxesAttr>(kManualAxes);
    ArrayRef<TensorShardingAttr> shardings = mlir::sdy::getShardings(op);
    if (!manualAxes || shardings.empty()) {
      return {};
    }
    return resolveManualInfo(manualAxes, shardings.front());
  };

  auto getValueManualInfo = [&](mlir::Value v) -> ManualInfo {
    llvm::SmallPtrSet<mlir::Value, 8> visited;
    auto impl = [&](auto& self, mlir::Value val) -> ManualInfo {
      if (!val || !visited.insert(val).second) {
        return {};
      }
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
        if (blockArg.getOwner() == &funcOp.front()) {
          unsigned idx = blockArg.getArgNumber();
          return resolveManualInfo(
              funcOp.getArgAttrOfType<ManualAxesAttr>(idx, kManualAxes),
              funcOp.getArgAttrOfType<TensorShardingAttr>(idx, kShardingAttr));
        }
        if (auto whileOp = mlir::dyn_cast_or_null<mlir::stablehlo::WhileOp>(
                blockArg.getOwner()->getParentOp())) {
          unsigned idx = blockArg.getArgNumber();
          if (idx < whileOp->getNumOperands()) {
            if (ManualInfo info = self(self, whileOp->getOperand(idx))) {
              return info;
            }
          }
          mlir::Operation* bodyTerm = whileOp.getBody().front().getTerminator();
          if (idx < bodyTerm->getNumOperands()) {
            if (ManualInfo info = self(self, bodyTerm->getOperand(idx))) {
              return info;
            }
          }
        }
        return {};
      }
      mlir::Operation* defOp = val.getDefiningOp();
      if (!defOp) {
        return {};
      }
      if (auto customCall =
              mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(defOp)) {
        StringRef target = customCall.getCallTargetName();
        if (target == kSPMDShardToFullShapeCallTargetName ||
            target == kLocalToGlobalShapeCallTargetName) {
          return {};
        }
      }
      if (ManualInfo info = getOpManualInfo(defOp)) {
        return info;
      }
      unsigned resIdx = mlir::cast<mlir::OpResult>(val).getResultNumber();
      if (auto whileOp = mlir::dyn_cast<mlir::stablehlo::WhileOp>(defOp)) {
        if (resIdx < whileOp->getNumOperands()) {
          if (ManualInfo info = self(self, whileOp->getOperand(resIdx))) {
            return info;
          }
        }
        mlir::Operation* bodyTerm = whileOp.getBody().front().getTerminator();
        if (resIdx < bodyTerm->getNumOperands()) {
          if (ManualInfo info = self(self, bodyTerm->getOperand(resIdx))) {
            return info;
          }
        }
        return {};
      }
      if (auto barrierOp =
              mlir::dyn_cast<mlir::stablehlo::OptimizationBarrierOp>(defOp)) {
        if (resIdx < barrierOp->getNumOperands()) {
          return self(self, barrierOp->getOperand(resIdx));
        }
        return {};
      }
      if (mlir::isa<mlir::mhlo::CopyOp, mlir::sdy::ReshardOp,
                    mlir::sdy::ShardingConstraintOp>(defOp) &&
          defOp->getNumOperands() == 1) {
        return self(self, defOp->getOperand(0));
      }
      return {};
    };
    return impl(impl, v);
  };

  auto isBoundaryOp = [](mlir::Operation* op) {
    if (op->hasTrait<mlir::OpTrait::IsTerminator>() ||
        mlir::isa<mlir::stablehlo::WhileOp, mlir::stablehlo::CaseOp,
                  mlir::stablehlo::IfOp, mlir::stablehlo::OptimizationBarrierOp,
                  mlir::stablehlo::InfeedOp, mlir::stablehlo::OutfeedOp,
                  mlir::stablehlo::SendOp, mlir::stablehlo::RecvOp,
                  mlir::stablehlo::AfterAllOp, mlir::stablehlo::CreateTokenOp,
                  mlir::func::CallOp, mlir::sdy::ManualComputationOp,
                  mlir::sdy::ShardingConstraintOp>(op)) {
      return true;
    }
    if (auto customCall = mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
      StringRef target = customCall.getCallTargetName();
      if (target == kSPMDFullToShardShapeCallTargetName ||
          target == kSPMDShardToFullShapeCallTargetName ||
          target == kGlobalToLocalShapeCallTargetName ||
          target == kLocalToGlobalShapeCallTargetName ||
          target == kFuncResultShardingTargetName ||
          target == kShardingCustomCallTargetName) {
        return true;
      }
    }
    return false;
  };

  auto assignManualToOp = [&](mlir::Operation* op, const ManualInfo& info) {
    if (!info || llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
          return mlir::isa<mlir::TupleType>(t);
        })) {
      return false;
    }
    if (isFullyManual(info.mesh, info.manualAxes.getValue())) {
      SmallVector<TensorShardingAttr> shardings;
      if (op->getNumResults() == 0) {
        shardings.push_back(TensorShardingAttr::get(
            ctx, info.meshOrRef, /*dimShardings=*/{}, /*replicatedAxes=*/{},
            /*unreducedAxes=*/{}));
      } else {
        shardings.reserve(op->getNumResults());
        for (mlir::Type resType : op->getResultTypes()) {
          int64_t rank = mlir::sdy::getTensorRank(resType);
          SmallVector<DimensionShardingAttr> closedDims(
              rank, DimensionShardingAttr::get(ctx, {}, /*is_closed=*/true));
          shardings.push_back(TensorShardingAttr::get(
              ctx, info.meshOrRef, closedDims, /*replicatedAxes=*/{},
              /*unreducedAxes=*/{}));
        }
      }
      mlir::sdy::setShardings(op, shardings);
    } else if (op->getNumResults() > 0) {
      SmallVector<AxisRefAttr> manualAxisRefs(
          info.manualAxes.getValue().begin(), info.manualAxes.getValue().end());
      mlir::sdy::sortAndMergeAxes(manualAxisRefs, info.mesh);
      SmallVector<TensorShardingAttr> shardings;
      shardings.reserve(op->getNumResults());
      for (mlir::Type resType : op->getResultTypes()) {
        int64_t rank = mlir::sdy::getTensorRank(resType);
        SmallVector<DimensionShardingAttr> openDims(
            rank, DimensionShardingAttr::get(ctx, {}, /*is_closed=*/false));
        shardings.push_back(TensorShardingAttr::get(ctx, info.meshOrRef,
                                                    openDims, manualAxisRefs,
                                                    /*unreducedAxes=*/{}));
      }
      mlir::sdy::setShardings(op, shardings);
    }
    op->setAttr(kManualAxes, info.manualAxes);
    return true;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    funcOp.front().walk([&](mlir::Operation* op) {
      if (op->hasAttr(kManualAxes) || !mlir::sdy::getShardings(op).empty() ||
          op->hasAttr(kXlaShardingAttr) || opHasSdyFrontendSharding(op) ||
          isBoundaryOp(op)) {
        return;
      }
      for (mlir::Value operand : op->getOperands()) {
        if (ManualInfo info = getValueManualInfo(operand)) {
          changed |= assignManualToOp(op, info);
          return;
        }
      }
      if (op->getNumResults() > 0 && !op->use_empty()) {
        ManualInfo commonUserManual;
        bool allUsesManual =
            llvm::all_of(op->getUses(), [&](const mlir::OpOperand& use) {
              mlir::Operation* user = use.getOwner();
              ManualInfo userManual;
              if (mlir::isa<mlir::func::ReturnOp>(user)) {
                unsigned idx = use.getOperandNumber();
                userManual = resolveManualInfo(
                    funcOp.getResultAttrOfType<ManualAxesAttr>(idx,
                                                               kManualAxes),
                    funcOp.getResultAttrOfType<TensorShardingAttr>(
                        idx, kShardingAttr));
              } else if (auto whileOp =
                             mlir::dyn_cast<mlir::stablehlo::WhileOp>(user)) {
                unsigned idx = use.getOperandNumber();
                if (idx < whileOp->getNumResults()) {
                  userManual = getValueManualInfo(whileOp->getResult(idx));
                }
              } else if (mlir::isa<mlir::stablehlo::ReturnOp>(user)) {
                if (auto whileOp =
                        mlir::dyn_cast_or_null<mlir::stablehlo::WhileOp>(
                            user->getParentOp());
                    whileOp && user->getParentRegion() == &whileOp.getBody()) {
                  unsigned idx = use.getOperandNumber();
                  if (idx < whileOp->getNumOperands()) {
                    userManual = getValueManualInfo(whileOp->getOperand(idx));
                  }
                }
              } else if (auto barrierOp = mlir::dyn_cast<
                             mlir::stablehlo::OptimizationBarrierOp>(user)) {
                unsigned idx = use.getOperandNumber();
                if (idx < barrierOp->getNumResults()) {
                  userManual = getValueManualInfo(barrierOp->getResult(idx));
                }
              } else if (auto customCall =
                             mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(
                                 user);
                         customCall &&
                         (customCall.getCallTargetName() ==
                              kSPMDFullToShardShapeCallTargetName ||
                          customCall.getCallTargetName() ==
                              kGlobalToLocalShapeCallTargetName)) {
                return false;
              } else {
                userManual = getOpManualInfo(user);
              }
              if (!userManual) {
                return false;
              }
              if (!commonUserManual) {
                commonUserManual = userManual;
              }
              return userManual == commonUserManual;
            });
        if (allUsesManual && commonUserManual) {
          changed |= assignManualToOp(op, commonUserManual);
        }
      }
    });
  }
}

// Convert the shardings in `funcOp` from kXlaShardingAttr into kShardingAttr.
LogicalResult importShardings(
    ModuleOp moduleOp, FuncOp funcOp, MeshAttr globalMesh,
    const SmallDenseMap<int64_t, StringRef>& deviceIdToSingleDeviceMeshName,
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults, bool inlineMesh) {
  if (funcOp.getSymName().contains(kManualComputationFuncName)) {
    return mlir::success();
  }
  const StringRef sdyShardingAttrName(
      xla::HloSharding::kShardingFrontendAttrName.data(),
      xla::HloSharding::kShardingFrontendAttrName.size());
  mlir::DictionaryAttr moduleDict = getFrontendAttrs(moduleOp);
  const bool isMain = funcOp.getSymName() == "main";
  const bool skipArgs = isMain && hasKey(moduleDict, kInTupleShardings);
  bool skipResults = isMain && hasKey(moduleDict, kOutTupleShardings);
  if (!funcOp.isExternal()) {
    funcOp.front().walk([&](mlir::stablehlo::CustomCallOp customCall) {
      if (customCall.getCallTargetName() == kFuncResultShardingTargetName) {
        skipResults = true;
      }
    });
  }

  if (!skipArgs) {
    for (auto [argNum, argType] : llvm::enumerate(funcOp.getArgumentTypes())) {
      if (hasKey(getFuncArgFrontendAttrs(funcOp, argNum),
                 sdyShardingAttrName)) {
        continue;
      }
      if (auto oldSharding =
              funcOp.getArgAttrOfType<StringAttr>(argNum, kXlaShardingAttr)) {
        const xla::HloSharding hloSharding =
            parseShardingFromString(oldSharding);
        ExtractedManualAxes extracted =
            extractManualAxesFromHloSharding(hloSharding, globalMesh);
        if (extracted.manualAxes) {
          funcOp.setArgAttr(argNum, kManualAxes, extracted.manualAxes);
        }
        funcOp.setArgAttr(
            argNum, kShardingAttr,
            convertToSdySharding(
                hloSharding, globalMesh, deviceIdToSingleDeviceMeshName,
                mlir::sdy::getTensorRank(argType),
                !extracted.manualAxes &&
                    shouldOpenDims(allowPropagationToArgs, argNum),
                inlineMesh));
        funcOp.removeArgAttr(argNum, kXlaShardingAttr);
      }
    }
  }

  if (!skipResults) {
    for (auto [resNum, resType] : llvm::enumerate(funcOp.getResultTypes())) {
      if (auto oldSharding = funcOp.getResultAttrOfType<StringAttr>(
              resNum, kXlaShardingAttr)) {
        const xla::HloSharding hloSharding =
            parseShardingFromString(oldSharding);
        ExtractedManualAxes extracted =
            extractManualAxesFromHloSharding(hloSharding, globalMesh);
        if (extracted.manualAxes) {
          funcOp.setResultAttr(resNum, kManualAxes, extracted.manualAxes);
        }
        funcOp.setResultAttr(
            resNum, kShardingAttr,
            convertToSdySharding(
                hloSharding, globalMesh, deviceIdToSingleDeviceMeshName,
                mlir::sdy::getTensorRank(resType),
                !extracted.manualAxes &&
                    shouldOpenDims(allowPropagationToResults, resNum),
                inlineMesh));
        funcOp.removeResultAttr(
            resNum, StringAttr::get(funcOp.getContext(), kXlaShardingAttr));
      }
    }
  }

  if (funcOp.isExternal()) {
    return mlir::success();
  }

  funcOp.front().walk([&](mlir::Operation* op) {
    const bool isTokenOnlyRecv =
        mlir::isa<mlir::stablehlo::RecvOp>(op) && op->getNumResults() == 1;
    if (opHasSdyFrontendSharding(op) || isTokenOnlyRecv ||
        mlir::isa<mlir::mhlo::AsyncStartOp, mlir::mhlo::AsyncUpdateOp,
                  mlir::mhlo::AsyncDoneOp, mlir::stablehlo::InfeedOp,
                  mlir::stablehlo::OutfeedOp, mlir::stablehlo::SendOp>(op) ||
        llvm::any_of(op->getResultTypes(), [](mlir::Type t) {
          return mlir::isa<mlir::TupleType>(t);
        })) {
      return;
    }
    if (auto oldSharding = op->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
      const xla::HloSharding hloSharding = parseShardingFromString(oldSharding);
      ArrayRef<xla::HloSharding> flatHloSharding = hloSharding;
      if (hloSharding.IsTuple()) {
        flatHloSharding = hloSharding.tuple_elements();
      }
      ExtractedManualAxes extracted =
          extractManualAxesFromHloSharding(hloSharding, globalMesh);
      SmallVector<AxisRefAttr> manualAxisRefs;
      if (extracted.manualAxes) {
        op->setAttr(kManualAxes, extracted.manualAxes);
        manualAxisRefs.assign(extracted.manualAxes.getValue().begin(),
                              extracted.manualAxes.getValue().end());
        mlir::sdy::sortAndMergeAxes(manualAxisRefs, extracted.mesh);
      }
      SmallVector<TensorShardingAttr> newShardings;
      if (op->getNumResults() == 0) {
        const xla::HloSharding& leafSharding = flatHloSharding.front();
        newShardings.push_back(convertToSdySharding(
            leafSharding, globalMesh, deviceIdToSingleDeviceMeshName,
            /*rank=*/0, /*openDims=*/false, inlineMesh));
      } else if (flatHloSharding.size() != op->getNumResults()) {
        return;
      } else {
        newShardings.reserve(op->getNumResults());
        for (const auto& [resHloSharding, resType] :
             llvm::zip_equal(flatHloSharding, op->getResultTypes())) {
          TensorShardingAttr shardingAttr = convertToSdySharding(
              resHloSharding, globalMesh, deviceIdToSingleDeviceMeshName,
              mlir::sdy::getTensorRank(resType),
              /*openDims=*/false, inlineMesh);
          if (auto customCall =
                  mlir::dyn_cast<mlir::stablehlo::CustomCallOp>(op);
              customCall && op->getNumResults() == 1 &&
              !resHloSharding.IsManual()) {
            StringRef target = customCall.getCallTargetName();
            if ((target == kShardingCustomCallTargetName &&
                 op->hasAttr(kManualAxes)) ||
                target == kSPMDFullToShardShapeCallTargetName ||
                target == kSPMDShardToFullShapeCallTargetName) {
              std::vector<int64_t> unspecDims;
              if (std::optional<mlir::Attribute> backendConfig =
                      customCall.getBackendConfig()) {
                if (auto configStr =
                        mlir::dyn_cast<StringAttr>(*backendConfig)) {
                  if (xla::sharding_op_util::ParseAttributes(
                          absl::string_view(configStr.getValue().data(),
                                            configStr.getValue().size()),
                          &unspecDims)
                          .ok() &&
                      !unspecDims.empty()) {
                    shardingAttr = shardingAttr.openShardingDims(unspecDims)
                                       .replaceReplicatedAxes(manualAxisRefs);
                  }
                }
              }
            }
          }
          if (!manualAxisRefs.empty() && !resHloSharding.IsManual() &&
              shardingAttr.getReplicatedAxes().empty()) {
            shardingAttr = shardingAttr.replaceReplicatedAxes(manualAxisRefs);
          }
          newShardings.push_back(shardingAttr);
        }
      }
      mlir::sdy::setShardings(op, newShardings);
      op->removeAttr(kXlaShardingAttr);
    }
  });

  propagateInlineManualShardings(moduleOp, funcOp, globalMesh);

  return mlir::success();
}

class ImportShardingsPass
    : public PassWrapper<ImportShardingsPass, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportShardingsPass)

  ImportShardingsPass(ArrayRef<bool> allowPropagationToArgs,
                      ArrayRef<bool> allowPropagationToResults, bool inlineMesh,
                      int64_t numPartitions)
      : allowPropagationToArgs(allowPropagationToArgs),
        allowPropagationToResults(allowPropagationToResults),
        inlineMesh(inlineMesh),
        numPartitions(numPartitions) {}

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    auto [namedAxes, deviceIdsForSingleDeviceMesh, meshDeviceIds] =
        findMeshAxesAndIds(moduleOp, numPartitions);
    if (namedAxes.empty() && deviceIdsForSingleDeviceMesh.empty()) {
      // There are no shardings in the `moduleOp`.
      return;
    }

    MeshAttr globalMesh =
        meshDeviceIds.empty()
            ? MeshAttr::get(moduleOp.getContext(), namedAxes)
            : MeshAttr::get(moduleOp.getContext(), namedAxes, meshDeviceIds);
    SmallDenseMap<int64_t, StringRef> deviceIdToSingleDeviceMeshName;
    if (!inlineMesh) {
      mlir::SymbolTableCollection symbolTableCollection;
      mlir::SymbolTable& symbolTable =
          symbolTableCollection.getSymbolTable(moduleOp);

      OpBuilder opBuilder = mlir::OpBuilder::atBlockBegin(moduleOp.getBody());
      // Create a global mesh containing all the axes.
      symbolTable.insert(MeshOp::create(opBuilder, moduleOp.getLoc(),
                                        kGlobalMeshName, globalMesh));

      for (int64_t deviceId : deviceIdsForSingleDeviceMesh) {
        // Create a mesh name with its deviceId as a suffix for each
        // single-device mesh.
        std::string meshName = absl::StrCat("single_device_", deviceId);
        auto meshOp = MeshOp::create(
            opBuilder, moduleOp.getLoc(), meshName,
            MeshAttr::getSingleDevice(moduleOp.getContext(), deviceId));
        symbolTable.insert(meshOp);
        deviceIdToSingleDeviceMeshName[deviceId] = meshOp.getSymName();
      }
    }

    for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
      // `allowPropagationToArgs` and `allowPropagationToResults` apply
      // only to the main/entry function.
      bool isMain = funcOp.getSymName() == "main";
      if (mlir::failed(importShardings(
              moduleOp, funcOp, globalMesh, deviceIdToSingleDeviceMeshName,
              isMain ? allowPropagationToArgs : ArrayRef<bool>(),
              isMain ? allowPropagationToResults : ArrayRef<bool>(),
              inlineMesh))) {
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
  bool inlineMesh;
  int64_t numPartitions;
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportShardingsPass(
    ArrayRef<bool> allowPropagationToArgs,
    ArrayRef<bool> allowPropagationToResults, bool inlineMesh,
    int64_t numPartitions) {
  return std::make_unique<ImportShardingsPass>(allowPropagationToArgs,
                                               allowPropagationToResults,
                                               inlineMesh, numPartitions);
}

void registerStablehloImportShardingsPass() {
  mlir::registerPass([]() {
    return createImportShardingsPass(ArrayRef<bool>(), ArrayRef<bool>(),
                                     /*inlineMesh=*/false);
  });
}

// NOTE: Keep it as a wrapper of SDY import pipeline that first converts from
// mhlo shardings to Shardy dialect shardings and calls SDY import pipeline.
// This way Shardy XLA Pass tests run the logic that actually runs in prod.
void addStablehloImportPipeline(mlir::OpPassManager& pm,
                                ArrayRef<bool> allowPropagationToArgs,
                                ArrayRef<bool> allowPropagationToResults,
                                bool enableHloShardingV3, bool inlineMesh,
                                bool liftAndDedupMeshes,
                                int64_t numPartitions) {
  pm.addNestedPass<FuncOp>(
      mlir::stablehlo_ext::createStablehloCanonicalizeFromHloImportPass());
  pm.addPass(createImportShardingsPass(allowPropagationToArgs,
                                       allowPropagationToResults, inlineMesh,
                                       numPartitions));
  if (!inlineMesh && !liftAndDedupMeshes && !enableHloShardingV3) {
    pm.addPass(mlir::sdy::createLiftInlinedMeshesPass(
        mlir::sdy::LiftInlinedMeshesPassOptions{
            /*useSingleDeviceMeshNames=*/true}));
  }
  addSdyRoundTripImportPipeline(pm, /*enableConstantImport=*/true,
                                liftAndDedupMeshes, enableHloShardingV3);
}

void registerStablehloImportPipeline() {
  mlir::PassPipelineRegistration<> importPipeline(
      "xla-sdy-stablehlo-import-pipeline",
      "Run passes to import a StableHLO module with `mhlo.shardings` into the "
      "SDY (Shardy) dialect.",
      [](mlir::OpPassManager& pm) {
        addStablehloImportPipeline(pm, ArrayRef<bool>(), ArrayRef<bool>());
      });
}

}  // namespace sdy
}  // namespace xla
