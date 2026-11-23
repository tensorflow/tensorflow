/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/spmd/shardy/extensions/custom_spmd_partitioning.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/ArithDialect.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/Passes.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/utils.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/mlir_hlo/stablehlo_ext/transforms/passes.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"
#include "xla/service/spmd/shardy/stablehlo_round_trip/sharding_import_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sdy {

std::optional<xla::HloSharding> convertValueShardingToHloSharding(
    mlir::Operation* op, mlir::Value value,
    const mlir::SymbolTable& symbolTable) {
  // Skip sdy::getSharding on block args unlinked during signature conversion.
  auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value);
  mlir::sdy::TensorShardingAttr sdySharding =
      (blockArg &&
       (!blockArg.getOwner() || !blockArg.getOwner()->getParentOp()))
          ? nullptr
          : mlir::sdy::getSharding(value);
  if (!sdySharding) {
    return xla::HloSharding::Replicate();
  }
  mlir::sdy::MeshAttr mesh = sdySharding.getMesh(symbolTable);
  if (!mesh) {
    op->emitOpError("unknown mesh: ") << sdySharding.getMeshOrRef();
    return std::nullopt;
  }
  return xla::sdy::convertToHloSharding(
      sdySharding, [&](mlir::sdy::TensorShardingAttr) { return mesh; });
}

absl::string_view getRawBackendConfig(
    mlir::stablehlo::CustomCallOp customCall) {
  if (std::optional<mlir::Attribute> backendConfigAttr =
          customCall.getBackendConfig()) {
    if (auto strAttr = llvm::dyn_cast<mlir::StringAttr>(*backendConfigAttr)) {
      llvm::StringRef ref = strAttr.getValue();
      return absl::string_view(ref.data(), ref.size());
    }
  }
  return {};
}

namespace {

constexpr llvm::StringLiteral kChannelHandleAttrName = "channel_handle";

// Pairs a resolved `MeshAttr` with its original inline or symbol-ref attribute.
struct ResolvedMesh {
  mlir::sdy::MeshAttr mesh;
  mlir::Attribute meshOrRef;
};

// Finds the active mesh and its attribute/symbol reference for `customCall`.
std::optional<ResolvedMesh> resolveMeshForCustomCall(
    mlir::stablehlo::CustomCallOp customCall,
    const mlir::SymbolTable& symbolTable) {
  std::optional<ResolvedMesh> fallbackEmptyMesh;
  auto inspectSharding = [&](mlir::sdy::TensorShardingAttr sharding)
      -> mlir::FailureOr<std::optional<ResolvedMesh>> {
    if (!sharding) {
      return std::optional<ResolvedMesh>(std::nullopt);
    }
    mlir::sdy::MeshAttr mesh = sharding.getMesh(symbolTable);
    if (!mesh) {
      customCall.emitOpError("unknown mesh: ") << sharding.getMeshOrRef();
      return mlir::failure();
    }
    if (mesh.isSingleDevice()) {
      return std::optional<ResolvedMesh>(std::nullopt);
    }
    if (!mesh.empty()) {
      return std::optional<ResolvedMesh>(
          ResolvedMesh{mesh, sharding.getMeshOrRef()});
    }
    if (!fallbackEmptyMesh.has_value()) {
      fallbackEmptyMesh = ResolvedMesh{mesh, sharding.getMeshOrRef()};
    }
    return std::optional<ResolvedMesh>(std::nullopt);
  };

  // Prefer a non-empty, non-single-device mesh directly referenced on
  // `customCall`.
  for (mlir::Value value : llvm::concat<mlir::Value>(customCall.getOperands(),
                                                     customCall.getResults())) {
    mlir::FailureOr<std::optional<ResolvedMesh>> found =
        inspectSharding(mlir::sdy::getSharding(value));
    if (mlir::failed(found)) {
      return std::nullopt;
    }
    if (found->has_value()) {
      return **found;
    }
  }

  // Fall back to enclosing `sdy.manual_computation` boundary shardings.
  for (auto parent =
           customCall->getParentOfType<mlir::sdy::ManualComputationOp>();
       parent;
       parent = parent->getParentOfType<mlir::sdy::ManualComputationOp>()) {
    for (mlir::sdy::TensorShardingAttr sharding :
         llvm::concat<const mlir::sdy::TensorShardingAttr>(
             parent.getInShardings().getShardings(),
             parent.getOutShardings().getShardings())) {
      mlir::FailureOr<std::optional<ResolvedMesh>> found =
          inspectSharding(sharding);
      if (mlir::failed(found)) {
        return std::nullopt;
      }
      if (found->has_value()) {
        return **found;
      }
    }
  }

  // Fall back to the module's global mesh or an empty mesh.
  if (auto moduleOp = customCall->getParentOfType<mlir::ModuleOp>()) {
    if (mlir::sdy::MeshOp meshOp = mlir::sdy::getGlobalMeshOp(moduleOp)) {
      return ResolvedMesh{meshOp.getMesh(),
                          mlir::FlatSymbolRefAttr::get(meshOp)};
    }
  }
  if (fallbackEmptyMesh.has_value()) {
    return *fallbackEmptyMesh;
  }
  mlir::sdy::MeshAttr emptyMesh =
      mlir::sdy::MeshAttr::get(customCall.getContext(), {});
  return ResolvedMesh{emptyMesh, emptyMesh};
}

// Factorizes a tiled V2 `HloSharding` over `resolvedMesh` into a closed Shardy
// `TensorShardingAttr`.
mlir::sdy::TensorShardingAttr convertTiledV2ShardingToSdySharding(
    mlir::Operation* op, const xla::HloSharding& rawSharding, int64_t rank,
    const ResolvedMesh& resolvedMesh) {
  auto [mesh, meshOrRef] = resolvedMesh;
  mlir::MLIRContext* ctx = op->getContext();
  if (absl::c_linear_search(rawSharding.subgroup_types(),
                            xla::OpSharding::UNREDUCED)) {
    op->emitOpError(
        "unreduced sharding is not supported in custom call partitioning: ")
        << rawSharding.ToString();
    return nullptr;
  }
  if (rawSharding.tile_assignment().num_elements() != mesh.getTotalSize()) {
    op->emitOpError("sharding device count (")
        << rawSharding.tile_assignment().num_elements()
        << ") does not match mesh total size (" << mesh.getTotalSize()
        << "): " << rawSharding.ToString();
    return nullptr;
  }
  if (rawSharding.TiledDataRank() != rank) {
    op->emitOpError("tiled sharding rank (")
        << rawSharding.TiledDataRank() << ") does not match tensor rank ("
        << rank << "): " << rawSharding.ToString();
    return nullptr;
  }

  xla::TileAssignment tileAssignment = rawSharding.tile_assignment();
  if (!mesh.getDeviceIds().empty()) {
    // Remap physical device IDs to logical mesh coordinates before factorizing.
    llvm::SmallDenseMap<int64_t, int64_t> physToLogical;
    for (auto [logicalIdx, physId] : llvm::enumerate(mesh.getDeviceIds())) {
      physToLogical[physId] = logicalIdx;
    }
    auto logicalArray =
        std::make_shared<xla::Array<int64_t>>(tileAssignment.dimensions());
    bool validDevices = true;
    tileAssignment.array().Each(
        [&](absl::Span<const int64_t> indices, int64_t physId) {
          auto it = physToLogical.find(physId);
          if (it == physToLogical.end()) {
            validDevices = false;
            return;
          }
          (*logicalArray)(indices) = it->second;
        });
    if (!validDevices) {
      op->emitOpError("sharding references device IDs not present in mesh: ")
          << rawSharding.ToString();
      return nullptr;
    }
    tileAssignment = xla::TileAssignment(std::move(logicalArray));
  }

  std::optional<xla::AnalyzeTileAssignmentResult> result;
  if (tileAssignment.iota().has_value()) {
    // Call `GetOrderedSubDimsFromIotaTileAssignment` directly because
    // `AnalyzeTileAssignment` CHECK-fails on unfactorizable V2 iota shardings.
    if (std::optional<std::vector<xla::SubDimInfo>> subDims =
            xla::GetOrderedSubDimsFromIotaTileAssignment(
                *tileAssignment.iota())) {
      std::vector<int64_t> localMesh;
      localMesh.reserve(subDims->size());
      for (const xla::SubDimInfo& subDimInfo : *subDims) {
        localMesh.push_back(subDimInfo.size);
      }
      result = xla::AnalyzeTileAssignmentResult{
          /*sub_dims=*/std::move(*subDims),
          /*local_mesh=*/std::move(localMesh),
          /*iota=*/*tileAssignment.iota(),
      };
    }
  } else {
    result = xla::AnalyzeTileAssignment(tileAssignment);
  }
  if (!result.has_value()) {
    op->emitOpError("failed to factorize custom call sharding over mesh axes: ")
        << rawSharding.ToString();
    return nullptr;
  }

  absl::StatusOr<TileFactorizationResult> factorized =
      factorizeTileAssignmentToSdy(*result, mesh, rank,
                                   ShardingDimClosedness::kClosed);
  if (!factorized.ok()) {
    op->emitOpError("failed to factorize custom call sharding (")
        << rawSharding.ToString() << "): " << factorized.status().message();
    return nullptr;
  }

  // Intentionally leave `replicatedAxes` empty: all dimensions are closed so
  // explicit replicated axes are redundant, and inside an enclosing
  // `sdy.manual_computation` listing an outer manual axis in `replicatedAxes`
  // violates the `ManualComputationOp` and `ReshardOp` verifiers.
  return mlir::sdy::TensorShardingAttr::get(ctx, meshOrRef,
                                            factorized->dimShardings,
                                            /*replicatedAxes=*/{},
                                            /*unreducedAxes=*/{});
}

// Returns true if `xlaMesh` has identical axis sizes and physical device
// assignment to non-empty, multi-device `sdyMesh`.
bool meshMatchesSdyMesh(const xla::Mesh& xlaMesh, mlir::sdy::MeshAttr sdyMesh) {
  if (xlaMesh.IsSingleDevice() || sdyMesh.isSingleDevice()) {
    return false;
  }
  llvm::ArrayRef<mlir::sdy::MeshAxisAttr> sdyAxes = sdyMesh.getAxes();
  if (sdyAxes.empty() ||
      xlaMesh.num_axes() != static_cast<int64_t>(sdyAxes.size())) {
    return false;
  }
  llvm::SmallVector<int64_t> sdyAxisSizes;
  sdyAxisSizes.reserve(sdyAxes.size());
  for (auto [i, axis] : llvm::enumerate(sdyAxes)) {
    if (xlaMesh.axis_size(i) != axis.getSize()) {
      return false;
    }
    sdyAxisSizes.push_back(axis.getSize());
  }
  if (sdyMesh.getDeviceIds().empty()) {
    return xlaMesh.device_assignment() == xla::TileAssignment(sdyAxisSizes);
  }
  auto expectedArray = std::make_shared<xla::Array<int64_t>>(sdyAxisSizes);
  absl::c_copy(sdyMesh.getDeviceIds(), expectedArray->begin());
  return xlaMesh.device_assignment() ==
         xla::TileAssignment(std::move(expectedArray));
}

// Converts a leaf `xla::HloSharding` returned by a custom partitioning callback
// into a verified, closed `mlir::sdy::TensorShardingAttr` bound to
// `resolvedMesh`.
mlir::sdy::TensorShardingAttr convertHloShardingToSdySharding(
    mlir::Operation* op, const xla::HloSharding& rawSharding,
    mlir::Type tensorType, const ResolvedMesh& resolvedMesh) {
  auto [mesh, meshOrRef] = resolvedMesh;
  mlir::MLIRContext* ctx = op->getContext();
  if (rawSharding.IsTuple()) {
    op->emitOpError("expected non-tuple leaf sharding, got: ")
        << rawSharding.ToString();
    return nullptr;
  }
  if (rawSharding.IsUnreduced() ||
      (rawSharding.UseNamedShardingLeaf() &&
       !rawSharding.named_sharding().unreduced_axes().empty())) {
    op->emitOpError(
        "unreduced sharding is not supported in custom call partitioning: ")
        << rawSharding.ToString();
    return nullptr;
  }

  int64_t rank = mlir::sdy::getTensorRank(tensorType);
  mlir::sdy::TensorShardingAttr sdySharding;
  if (rawSharding.IsSingleDevice()) {
    if (mesh.isSingleDevice()) {
      sdySharding = mlir::sdy::TensorShardingAttr::getFullyClosed(
          ctx, /*rank=*/0, meshOrRef);
    } else if (mesh.getTotalSize() == 1) {
      sdySharding =
          mlir::sdy::TensorShardingAttr::getFullyClosed(ctx, rank, meshOrRef);
    } else {
      op->emitOpError(
          "single-device sharding is not supported in custom call partitioning "
          "on a multi-device mesh: ")
          << rawSharding.ToString();
      return nullptr;
    }
  } else if (rawSharding.IsReplicated() || rawSharding.IsManual() ||
             rawSharding.IsUnknown()) {
    sdySharding = mlir::sdy::TensorShardingAttr::getFullyClosed(
        ctx, mesh.isSingleDevice() ? 0 : rank, meshOrRef);
  } else if (rawSharding.UseNamedShardingLeaf()) {
    const xla::NamedSharding& named = rawSharding.named_sharding();
    if (named.manual_axes().empty() && meshMatchesSdyMesh(named.mesh(), mesh)) {
      if (named.num_dimensions() != rank) {
        op->emitOpError("named sharding rank (")
            << named.num_dimensions() << ") does not match tensor rank ("
            << rank << "): " << rawSharding.ToString();
        return nullptr;
      }
      llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dimShardings;
      dimShardings.reserve(rank);
      for (const xla::NamedSharding::DimensionSharding& dimSharding :
           named.dim_shardings()) {
        llvm::SmallVector<mlir::sdy::AxisRefAttr> axes;
        axes.reserve(dimSharding.axes().size());
        for (const xla::AxisRef& axis : dimSharding.axes()) {
          if (axis.size(named.mesh()) == 1) {
            continue;
          }
          llvm::StringRef axisName =
              mesh.getAxes()[axis.mesh_axis_index()].getName();
          if (axis.sub_axis_info().has_value()) {
            axes.push_back(mlir::sdy::AxisRefAttr::get(
                ctx, axisName, axis.pre_size(), axis.size(named.mesh())));
          } else {
            axes.push_back(mlir::sdy::AxisRefAttr::get(ctx, axisName));
          }
        }
        dimShardings.push_back(mlir::sdy::DimensionShardingAttr::get(
            ctx, axes, /*isClosed=*/true));
      }
      // Intentionally leave `replicatedAxes` empty (see V2 path above).
      sdySharding = mlir::sdy::TensorShardingAttr::get(
          ctx, meshOrRef, dimShardings, /*replicatedAxes=*/{},
          /*unreducedAxes=*/{});
    } else {
      // Fallback when the callback's NamedSharding uses a different mesh
      // factorization/device ordering or partial manual axes: lower to V2 and
      // factorize over `resolvedMesh`.
      const xla::HloSharding v2 = xla::HloSharding::V3ToV2Sharding(rawSharding);
      if (v2.IsTiled()) {
        sdySharding =
            convertTiledV2ShardingToSdySharding(op, v2, rank, resolvedMesh);
      } else if (v2.IsReplicated() || v2.IsManual()) {
        sdySharding = mlir::sdy::TensorShardingAttr::getFullyClosed(
            ctx, mesh.isSingleDevice() ? 0 : rank, meshOrRef);
      } else {
        op->emitOpError(
            "unsupported sharding returned by custom call partitioner: ")
            << rawSharding.ToString();
        return nullptr;
      }
    }
  } else if (rawSharding.IsTiled()) {
    sdySharding = convertTiledV2ShardingToSdySharding(op, rawSharding, rank,
                                                      resolvedMesh);
  } else {
    op->emitOpError(
        "unsupported sharding returned by custom call partitioner: ")
        << rawSharding.ToString();
    return nullptr;
  }

  if (!sdySharding) {
    return nullptr;
  }
  if (mlir::failed(sdySharding.verifyForType(
          tensorType, mesh,
          [&](llvm::StringRef msg) {
            return op->emitOpError("invalid callback sharding (")
                   << rawSharding.ToString() << "): " << msg;
          },
          /*checkDivisibility=*/false))) {
    return nullptr;
  }
  return sdySharding;
}

// Marks all non-@main `func.func` ops private so InlinerPass and SymbolDCE
// can inline and clean up callback helper functions.
class MarkNonMainFunctionsPrivatePass
    : public mlir::PassWrapper<MarkNonMainFunctionsPrivatePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkNonMainFunctionsPrivatePass)

 protected:
  void runOnOperation() override {
    for (mlir::func::FuncOp func :
         getOperation().getOps<mlir::func::FuncOp>()) {
      if (func.getSymName() == "main") {
        func.setPublic();
      } else {
        func.setPrivate();
      }
    }
  }
};

// Parses and prepares the callback MLIR module: upgrades VHLO, inlines helper
// calls (while leaving `stablehlo.composite` decomposition refs intact), and
// legalizes any CHLO / complex-math ops to StableHLO/MHLO.
mlir::OwningOpRef<mlir::ModuleOp> parseAndPrepareCallbackModule(
    mlir::Operation* op, absl::string_view mlirModuleStr) {
  mlir::MLIRContext* ctx = op->getContext();
  mlir::OwningOpRef<mlir::ModuleOp> callbackModule;
  std::string errMsg;
  {
    mlir::BaseScopedDiagnosticHandler diagHandler(ctx);
    callbackModule = mlir::parseSourceString<mlir::ModuleOp>(
        llvm::StringRef(mlirModuleStr.data(), mlirModuleStr.size()),
        mlir::ParserConfig{ctx});
    if (!callbackModule) {
      errMsg = std::string(diagHandler.ConsumeStatus().message());
    }
  }
  if (!callbackModule) {
    op->emitOpError(
        "failed to parse custom call partitioner MLIR module "
        "(using StableHLO v")
        << mlir::vhlo::Version::getCurrentVersion() << "): " << errMsg;
    return nullptr;
  }

  // Deserialize VHLO first so `vhlo.func_v1` helpers become `func.func` ops
  // before marking non-@main functions private for inlining and SymbolDCE, then
  // recompose composites and legalize CHLO and complex math to StableHLO/MHLO.
  mlir::PassManager pm(ctx);
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  pm.addPass(std::make_unique<MarkNonMainFunctionsPrivatePass>());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createChloRecomposeOpsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createChloLegalizeToHighLevelMhloPass(
          mlir::mhlo::getDefaultChloToHighLevelMhloOptions()));
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloComplexMathExpanderPass());
  bool failedToLegalize = false;
  {
    mlir::BaseScopedDiagnosticHandler diagHandler(ctx);
    if (mlir::failed(pm.run(*callbackModule))) {
      failedToLegalize = true;
      errMsg = std::string(diagHandler.ConsumeStatus().message());
    }
  }
  if (failedToLegalize) {
    op->emitOpError("failed to legalize custom call partitioner MLIR module: ")
        << errMsg;
    return nullptr;
  }
  return callbackModule;
}

// Inlines a prepared callback MLIR module into `targetRegion` (the body of an
// `sdy.manual_computation`): remaps positive `channel_handle` IDs using
// `nextChannelId` (preserving matched Send/Recv channel pairs), imports any
// remaining private helper functions (such as `stablehlo.composite`
// decompositions) and referenced `sdy.mesh` symbols (e.g. from RGV3 collective
// `replica_groups`) into `symbolTable` with collision-free names, splices
// `mainFunc`'s body into `targetRegion`, and replaces `func.return` with
// `sdy.return`.
mlir::LogicalResult inlineCallbackMlirModule(mlir::Operation* callOp,
                                             mlir::ModuleOp callbackModule,
                                             mlir::func::FuncOp mainFunc,
                                             mlir::Region& targetRegion,
                                             mlir::IRRewriter& rewriter,
                                             mlir::SymbolTable& symbolTable,
                                             int64_t& nextChannelId) {
  // Verify @main has no remaining symbol uses (e.g. recursive calls) and check
  // top-level ops before mutating `nextChannelId` or the outer module.
  if (!mlir::SymbolTable::symbolKnownUseEmpty(mainFunc.getOperation(),
                                              callbackModule.getOperation())) {
    return callOp->emitOpError(
        "custom call partitioner @main cannot be referenced inside the "
        "callback module");
  }
  llvm::SmallVector<mlir::SymbolOpInterface> importedSymbols;
  for (mlir::Operation& topOp : callbackModule.getBody()->getOperations()) {
    if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(&topOp)) {
      if (func != mainFunc) {
        importedSymbols.push_back(func);
      }
      continue;
    }
    if (auto meshOp = llvm::dyn_cast<mlir::sdy::MeshOp>(&topOp)) {
      if (mlir::SymbolTable::symbolKnownUseEmpty(
              meshOp.getOperation(), callbackModule.getOperation())) {
        continue;
      }
      if (auto existing =
              symbolTable.lookup<mlir::sdy::MeshOp>(meshOp.getSymNameAttr());
          existing && existing.getMesh() == meshOp.getMesh()) {
        continue;
      }
      importedSymbols.push_back(meshOp);
      continue;
    }
    return callOp->emitOpError(
               "unexpected top-level op in custom call partitioner MLIR "
               "module: ")
           << topOp.getName();
  }
  mlir::Operation* topLevelOp = nullptr;
  std::optional<mlir::SymbolTable> callbackSymTab;
  if (!importedSymbols.empty()) {
    mlir::Operation* outerModuleOp = symbolTable.getOp();
    topLevelOp = callOp;
    while (topLevelOp && topLevelOp->getParentOp() != outerModuleOp) {
      topLevelOp = topLevelOp->getParentOp();
    }
    if (!topLevelOp) {
      return callOp->emitOpError(
          "expected custom call to be nested inside the symbol table module");
    }
    callbackSymTab.emplace(callbackModule);
    for (mlir::SymbolOpInterface symOp : importedSymbols) {
      if (symbolTable.lookup(symOp.getNameAttr()) != nullptr &&
          mlir::failed(callbackSymTab->renameToUnique(symOp, {&symbolTable}))) {
        return callOp->emitOpError(
                   "failed to update symbol uses for callback symbol: ")
               << symOp.getName();
      }
    }
  }

  mlir::MLIRContext* ctx = callbackModule.getContext();
  // Remap positive channel handles consistently so paired Send/Recv ops share
  // a fresh channel ID.
  llvm::SmallDenseMap<int64_t, int64_t> remappedChannelIds;
  callbackModule->walk([&](mlir::Operation* op) {
    auto channelHandle = op->getAttrOfType<mlir::stablehlo::ChannelHandleAttr>(
        kChannelHandleAttrName);
    if (!channelHandle || channelHandle.getHandle() <= 0) {
      return;
    }
    auto [it, inserted] = remappedChannelIds.try_emplace(
        channelHandle.getHandle(), nextChannelId);
    if (inserted) {
      ++nextChannelId;
    }
    op->setAttr(kChannelHandleAttrName,
                mlir::stablehlo::ChannelHandleAttr::get(
                    ctx, it->second, channelHandle.getType()));
  });

  if (!importedSymbols.empty()) {
    // Insert before the enclosing top-level op so an active `moduleOp.walk`
    // does not subsequently visit newly imported shard-local symbols (and so
    // any imported `sdy.mesh` stays after existing global meshes).
    mlir::Block::iterator insertPt(topLevelOp);
    for (mlir::SymbolOpInterface symOp : importedSymbols) {
      callbackSymTab->remove(symOp);
      symOp->remove();
      symbolTable.insert(symOp, insertPt);
    }
  }

  // Splice @main into the manual computation body and rewrite its terminator.
  rewriter.inlineRegionBefore(mainFunc.getBody(), targetRegion,
                              targetRegion.end());
  mlir::Block& bodyBlock = targetRegion.front();
  auto funcReturn = llvm::cast<mlir::func::ReturnOp>(bodyBlock.getTerminator());
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(funcReturn);
  mlir::sdy::ReturnOp::create(rewriter, funcReturn.getLoc(),
                              funcReturn.getOperands());
  rewriter.eraseOp(funcReturn);
  return mlir::success();
}

// Holds unpacked callback MLIR module text and flat operand/result leaf
// shardings.
struct UnpackedPartitionCallbackResult {
  std::string mlirModule;
  std::vector<xla::HloSharding> argShardings;
  std::vector<xla::HloSharding> resultLeafShardings;
};

// Invokes `partitioner->PartitionToMlir` on `customCall`'s unpadded global
// shapes and shardings and unpacks tuple/leaf result shardings.
std::optional<mlir::FailureOr<UnpackedPartitionCallbackResult>>
invokePartitionToMlir(mlir::stablehlo::CustomCallOp customCall,
                      const xla::CustomCallPartitioner* partitioner,
                      const mlir::SymbolTable& symbolTable) {
  std::vector<xla::Shape> argShapes;
  std::vector<std::optional<xla::HloSharding>> argShardings;
  argShapes.reserve(customCall.getNumOperands());
  argShardings.reserve(customCall.getNumOperands());
  for (mlir::Value operand : customCall.getOperands()) {
    argShapes.push_back(xla::TypeToShape(operand.getType()));
    std::optional<xla::HloSharding> sharding =
        convertValueShardingToHloSharding(customCall, operand, symbolTable);
    if (!sharding.has_value()) {
      return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
    }
    argShardings.push_back(std::move(*sharding));
  }

  std::vector<xla::Shape> resultShapes;
  std::vector<xla::HloSharding> resultLeafShardings;
  resultShapes.reserve(customCall.getNumResults());
  resultLeafShardings.reserve(customCall.getNumResults());
  for (mlir::Value result : customCall.getResults()) {
    resultShapes.push_back(xla::TypeToShape(result.getType()));
    std::optional<xla::HloSharding> sharding =
        convertValueShardingToHloSharding(customCall, result, symbolTable);
    if (!sharding.has_value()) {
      return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
    }
    resultLeafShardings.push_back(std::move(*sharding));
  }

  xla::Shape resultShape;
  xla::HloSharding resultSharding = xla::HloSharding::Replicate();
  if (customCall.getNumResults() == 1) {
    resultShape = std::move(resultShapes[0]);
    resultSharding = std::move(resultLeafShardings[0]);
  } else {
    resultShape = xla::ShapeUtil::MakeTupleShape(resultShapes);
    resultSharding = xla::HloSharding::Tuple(resultShape, resultLeafShardings);
  }

  absl::StatusOr<std::optional<xla::CustomCallPartitionResult>>
      partitionResult = partitioner->PartitionToMlir(
          argShapes, argShardings, resultShape, resultSharding,
          getRawBackendConfig(customCall));
  if (!partitionResult.ok()) {
    customCall.emitOpError("custom call partitioning failed: ")
        << partitionResult.status().message();
    return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
  }
  if (!partitionResult->has_value()) {
    return std::nullopt;
  }

  if ((*partitionResult)->arg_shardings.size() != customCall.getNumOperands()) {
    customCall.emitOpError(
        "custom call partitioner returned wrong number of operand shardings: "
        "expected ")
        << customCall.getNumOperands() << ", got "
        << (*partitionResult)->arg_shardings.size();
    return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
  }

  std::vector<xla::HloSharding> desiredResultLeafShardings;
  if (customCall.getNumResults() == 1) {
    if ((*partitionResult)->result_sharding.IsTuple()) {
      customCall.emitOpError(
          "expected non-tuple result sharding for single-result custom call");
      return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
    }
    desiredResultLeafShardings.push_back(
        std::move((*partitionResult)->result_sharding));
  } else if (customCall.getNumResults() == 0) {
    const xla::HloSharding& resSharding = (*partitionResult)->result_sharding;
    bool validEmptyTuple = resSharding.IsTuple() &&
                           (resSharding.tuple_elements().empty() ||
                            (resSharding.tuple_elements().size() == 1 &&
                             resSharding.tuple_elements()[0].IsReplicated()));
    if (!validEmptyTuple && !resSharding.IsReplicated()) {
      customCall.emitOpError(
          "custom call partitioner returned invalid result sharding for "
          "0-result custom call");
      return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
    }
  } else {
    if (!(*partitionResult)->result_sharding.IsTuple() ||
        (*partitionResult)->result_sharding.tuple_elements().size() !=
            customCall.getNumResults()) {
      customCall.emitOpError(
          "custom call partitioner returned invalid tuple result sharding: "
          "expected ")
          << customCall.getNumResults() << " elements";
      return mlir::FailureOr<UnpackedPartitionCallbackResult>(mlir::failure());
    }
    desiredResultLeafShardings.assign(
        (*partitionResult)->result_sharding.tuple_elements().begin(),
        (*partitionResult)->result_sharding.tuple_elements().end());
  }

  return mlir::FailureOr<UnpackedPartitionCallbackResult>(
      UnpackedPartitionCallbackResult{
          std::move((*partitionResult)->mlir_module),
          std::move((*partitionResult)->arg_shardings),
          std::move(desiredResultLeafShardings)});
}

// Verifies that `sharding` does not bind enclosing manual axes or conflict
// with the newly entered `manualAxes`.
mlir::LogicalResult validateShardingAgainstManualAxes(
    mlir::stablehlo::CustomCallOp customCall,
    mlir::sdy::TensorShardingAttr sharding,
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> parentManualAxes,
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> manualAxes) {
  mlir::sdy::AxisRefAttr badAxis;
  mlir::sdy::AxisRefAttr conflictingAxis;
  bool overlapsParent = false;
  sharding.anyOfAxisRef([&](mlir::sdy::AxisRefAttr axis) {
    for (mlir::sdy::AxisRefAttr parentAxis : parentManualAxes) {
      if (axis.overlaps(parentAxis)) {
        badAxis = axis;
        conflictingAxis = parentAxis;
        overlapsParent = true;
        return true;
      }
    }
    for (mlir::sdy::AxisRefAttr manualAxis : manualAxes) {
      if (!axis.canCoexist(manualAxis)) {
        badAxis = axis;
        conflictingAxis = manualAxis;
        return true;
      }
    }
    return false;
  });
  if (!badAxis) {
    return mlir::success();
  }
  if (overlapsParent) {
    return customCall.emitOpError(
               "custom call partitioner returned sharding binding axis ")
           << badAxis.toString()
           << " which overlaps already manual enclosing axis "
           << conflictingAxis.toString() << ": " << sharding;
  }
  return customCall.emitOpError(
             "custom call partitioner returned sharding axis ")
         << badAxis.toString() << " that cannot coexist with manual axis "
         << conflictingAxis.toString() << ": " << sharding;
}

// Computes divisibility-padded global types for `globalTypes` and verifies that
// each tiled shard-local type matches the corresponding `localTypes` entry on
// `@main`.
mlir::FailureOr<llvm::SmallVector<mlir::Type>> computeAndVerifyPaddedTypes(
    mlir::stablehlo::CustomCallOp customCall, mlir::TypeRange globalTypes,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> shardings,
    llvm::ArrayRef<mlir::Type> localTypes, mlir::sdy::MeshAttr mesh,
    llvm::StringRef kind) {
  llvm::SmallVector<mlir::Type> paddedTypes;
  paddedTypes.reserve(globalTypes.size());
  for (auto [i, item] :
       llvm::enumerate(llvm::zip_equal(globalTypes, shardings, localTypes))) {
    auto [globalType, sharding, localType] = item;
    mlir::Type paddedType =
        mlir::sdy::getDivisiblePaddedType(globalType, sharding, mesh);
    auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(paddedType);
    mlir::Type expectedLocalType =
        rankedType ? sharding.getLocalTensorType(rankedType, mesh) : paddedType;
    if (expectedLocalType != localType) {
      return customCall.emitOpError("custom call partitioner @main ")
             << kind << " " << i << " type " << localType
             << " does not match tiled local type " << expectedLocalType;
    }
    paddedTypes.push_back(paddedType);
  }
  return paddedTypes;
}

// Validates that `values` carry no unreduced axes, converts each callback
// `HloSharding` to a verified Shardy `TensorShardingAttr` on `resolvedMesh`,
// and checks compatibility against enclosing and inner manual axes.
mlir::FailureOr<llvm::SmallVector<mlir::sdy::TensorShardingAttr>>
convertAndValidateCallbackShardings(
    mlir::stablehlo::CustomCallOp customCall, mlir::ValueRange values,
    llvm::ArrayRef<xla::HloSharding> hloShardings,
    const ResolvedMesh& resolvedMesh,
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> parentManualAxes,
    llvm::ArrayRef<mlir::sdy::AxisRefAttr> manualAxes) {
  llvm::SmallVector<mlir::sdy::TensorShardingAttr> desiredShardings;
  desiredShardings.reserve(values.size());
  for (auto [value, hloSharding] : llvm::zip_equal(values, hloShardings)) {
    if (mlir::sdy::TensorShardingAttr origSharding =
            mlir::sdy::getSharding(value);
        origSharding && !origSharding.getUnreducedAxes().empty()) {
      return customCall.emitOpError(
                 "unreduced sharding is not supported in custom call "
                 "partitioning: ")
             << origSharding;
    }
    mlir::sdy::TensorShardingAttr sdySharding = convertHloShardingToSdySharding(
        customCall, hloSharding, value.getType(), resolvedMesh);
    if (!sdySharding ||
        mlir::failed(validateShardingAgainstManualAxes(
            customCall, sdySharding, parentManualAxes, manualAxes))) {
      return mlir::failure();
    }
    desiredShardings.push_back(sdySharding);
  }
  return desiredShardings;
}

// Reshards `val` to `targetSharding` unless its current sharding is already
// equivalent on the same mesh (preserving single-device and cross-mesh
// transitions).
mlir::Value reshardIfNeeded(mlir::Location loc, mlir::Value val,
                            mlir::sdy::TensorShardingAttr targetSharding,
                            const mlir::SymbolTable& symbolTable,
                            mlir::IRRewriter& rewriter) {
  mlir::sdy::TensorShardingAttr curSharding = mlir::sdy::getSharding(val);
  if (curSharding == targetSharding) {
    return val;
  }
  bool sameMeshOrUnannotated =
      !curSharding ||
      curSharding.getMeshOrRef() == targetSharding.getMeshOrRef();
  bool neitherSingleDevice =
      (!curSharding ||
       !mlir::sdy::isSingleDeviceSharding(curSharding, symbolTable)) &&
      !mlir::sdy::isSingleDeviceSharding(targetSharding, symbolTable);
  if (sameMeshOrUnannotated && neitherSingleDevice &&
      targetSharding.isEquivalent(curSharding)) {
    return val;
  }
  return mlir::sdy::ReshardOp::create(rewriter, loc, val, targetSharding);
}

}  // namespace

void registerCustomSPMDPartitioningDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::shape::ShapeDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);
}

std::optional<mlir::LogicalResult> resolveCustomCallPartitioning(
    mlir::stablehlo::CustomCallOp customCall, mlir::IRRewriter& rewriter,
    mlir::SymbolTable& symbolTable, int64_t& nextChannelId) {
  llvm::StringRef callTargetName = customCall.getCallTargetName();
  const xla::CustomCallPartitioner* partitioner = xla::GetCustomCallPartitioner(
      absl::string_view(callTargetName.data(), callTargetName.size()));
  if (!partitioner || !partitioner->SupportsMlirPartitioning()) {
    return std::nullopt;
  }

  std::optional<mlir::FailureOr<UnpackedPartitionCallbackResult>> unpacked =
      invokePartitionToMlir(customCall, partitioner, symbolTable);
  if (!unpacked.has_value()) {
    return std::nullopt;
  }
  if (mlir::failed(*unpacked)) {
    return mlir::failure();
  }

  std::optional<ResolvedMesh> resolvedMesh =
      resolveMeshForCustomCall(customCall, symbolTable);
  if (!resolvedMesh.has_value()) {
    return mlir::failure();
  }
  auto [mesh, meshOrRef] = *resolvedMesh;
  mlir::MLIRContext* ctx = customCall.getContext();

  // Compute the remaining free mesh axes that the inner manual computation
  // will enter.
  llvm::SmallVector<mlir::sdy::AxisRefAttr> parentManualAxes;
  for (auto parent =
           customCall->getParentOfType<mlir::sdy::ManualComputationOp>();
       parent;
       parent = parent->getParentOfType<mlir::sdy::ManualComputationOp>()) {
    llvm::append_range(parentManualAxes, parent.getManualAxes());
  }
  llvm::SmallVector<mlir::sdy::AxisRefAttr> fullMeshAxes;
  fullMeshAxes.reserve(mesh.getAxes().size());
  for (mlir::sdy::MeshAxisAttr axis : mesh.getAxes()) {
    fullMeshAxes.push_back(mlir::sdy::AxisRefAttr::get(ctx, axis.getName()));
  }
  llvm::SmallVector<mlir::sdy::AxisRefAttr> manualAxes =
      mlir::sdy::getAxisSetDiff(fullMeshAxes, parentManualAxes, mesh);

  // Convert and validate callback operand and result shardings before mutating
  // the outer module.
  mlir::FailureOr<llvm::SmallVector<mlir::sdy::TensorShardingAttr>>
      desiredArgShardings = convertAndValidateCallbackShardings(
          customCall, customCall.getOperands(), (*unpacked)->argShardings,
          *resolvedMesh, parentManualAxes, manualAxes);
  if (mlir::failed(desiredArgShardings)) {
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<mlir::sdy::TensorShardingAttr>>
      desiredResultShardings = convertAndValidateCallbackShardings(
          customCall, customCall.getResults(), (*unpacked)->resultLeafShardings,
          *resolvedMesh, parentManualAxes, manualAxes);
  if (mlir::failed(desiredResultShardings)) {
    return mlir::failure();
  }

  mlir::OwningOpRef<mlir::ModuleOp> callbackModule =
      parseAndPrepareCallbackModule(customCall, (*unpacked)->mlirModule);
  if (!callbackModule) {
    return mlir::failure();
  }

  auto mainFunc = callbackModule->lookupSymbol<mlir::func::FuncOp>("main");
  if (!mainFunc || !llvm::hasSingleElement(mainFunc.getBody()) ||
      !llvm::isa<mlir::func::ReturnOp>(
          mainFunc.getBody().front().getTerminator())) {
    return customCall.emitOpError(
        "expected custom call partitioner MLIR module to contain a "
        "single-block @main function returning via func.return");
  }
  if (mainFunc.getNumArguments() != customCall.getNumOperands() ||
      mainFunc.getNumResults() != customCall.getNumResults()) {
    return customCall.emitOpError(
               "custom call partitioner @main signature arity (")
           << mainFunc.getNumArguments() << " -> " << mainFunc.getNumResults()
           << ") does not match custom call (" << customCall.getNumOperands()
           << " -> " << customCall.getNumResults() << ")";
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type>> paddedInputTypes =
      computeAndVerifyPaddedTypes(customCall, customCall.getOperandTypes(),
                                  *desiredArgShardings,
                                  mainFunc.getArgumentTypes(), mesh, "arg");
  if (mlir::failed(paddedInputTypes)) {
    return mlir::failure();
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Type>> paddedResultTypes =
      computeAndVerifyPaddedTypes(customCall, customCall.getResultTypes(),
                                  *desiredResultShardings,
                                  mainFunc.getResultTypes(), mesh, "result");
  if (mlir::failed(paddedResultTypes)) {
    return mlir::failure();
  }

  // Reshard and pad operands to divisible global shapes, inline @main into
  // sdy.manual_computation, then slice and reshard results back.
  mlir::Location loc = customCall.getLoc();
  llvm::SmallVector<mlir::Value> paddedOperands;
  paddedOperands.reserve(customCall.getNumOperands());
  for (auto [i, operand] : llvm::enumerate(customCall.getOperands())) {
    mlir::sdy::TensorShardingAttr inSharding = (*desiredArgShardings)[i];
    mlir::Value inVal =
        reshardIfNeeded(loc, operand, inSharding, symbolTable, rewriter);
    paddedOperands.push_back(mlir::sdy::padHighSideToType(
        loc, inVal, (*paddedInputTypes)[i], inSharding, rewriter));
  }

  auto manualComp = mlir::sdy::ManualComputationOp::create(
      rewriter, loc, *paddedResultTypes, paddedOperands,
      mlir::sdy::TensorShardingPerValueAttr::get(ctx, *desiredArgShardings),
      mlir::sdy::TensorShardingPerValueAttr::get(ctx, *desiredResultShardings),
      manualAxes);
  if (mlir::failed(inlineCallbackMlirModule(
          customCall, *callbackModule, mainFunc, manualComp.getRegion(),
          rewriter, symbolTable, nextChannelId))) {
    rewriter.eraseOp(manualComp);
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> finalResults;
  finalResults.reserve(customCall.getNumResults());
  for (auto [i, origResult] : llvm::enumerate(customCall.getResults())) {
    mlir::Value outVal = mlir::sdy::sliceHighSideToType(
        loc, manualComp.getResult(i), origResult.getType(),
        (*desiredResultShardings)[i], rewriter);
    mlir::sdy::TensorShardingAttr origOutSharding =
        mlir::sdy::getSharding(origResult);
    if (!origOutSharding) {
      origOutSharding = mlir::sdy::TensorShardingAttr::getFullyClosed(
          ctx, mesh.isSingleDevice() ? 0 : mlir::sdy::getTensorRank(outVal),
          meshOrRef);
    }
    finalResults.push_back(
        reshardIfNeeded(loc, outVal, origOutSharding, symbolTable, rewriter));
  }

  rewriter.replaceOp(customCall, finalResults);
  return mlir::success();
}

}  // namespace sdy
}  // namespace xla
