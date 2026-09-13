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

#include "xla/service/spmd/shardy/sdy_round_trip/shard_map_export.h"

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ModuleOp;
using ::mlir::StringRef;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

namespace stablehlo = ::mlir::stablehlo;
namespace sdy = ::mlir::sdy;

// Returns `shardings` with `manualAxes` stripped from each element, i.e. the
// local view of tensors that are sharded over those axes.
//
// A `NamedSharding` may not mention the same axis twice, so an op whose result
// is a local tensor carries the manual axes in its `xla.sdy.manual_axes`
// attribute and the remaining axes in its dimension shardings.
sdy::TensorShardingPerValueAttr localizeShardings(
    sdy::TensorShardingPerValueAttr shardings,
    mlir::ArrayRef<mlir::StringAttr> manualAxes, mlir::MLIRContext* context) {
  llvm::SmallVector<sdy::TensorShardingAttr> localShardings;
  localShardings.reserve(shardings.size());
  for (sdy::TensorShardingAttr sharding : shardings.getShardings()) {
    localShardings.push_back(sdy::eraseManualAxes(sharding, manualAxes));
  }
  return sdy::TensorShardingPerValueAttr::get(context, localShardings);
}

class SdyRoundTripShardMapExportPass
    : public mlir::PassWrapper<SdyRoundTripShardMapExportPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SdyRoundTripShardMapExportPass)

  SdyRoundTripShardMapExportPass() = default;
  explicit SdyRoundTripShardMapExportPass(bool enableHloShardingV3) {
    this->enableHloShardingV3 = enableHloShardingV3;
  }
  SdyRoundTripShardMapExportPass(const SdyRoundTripShardMapExportPass& other)
      : mlir::PassWrapper<SdyRoundTripShardMapExportPass,
                          mlir::OperationPass<ModuleOp>>(other) {
    this->enableHloShardingV3 = other.enableHloShardingV3.getValue();
  }

  Option<bool> enableHloShardingV3{
      *this, "enable-hlo-sharding-v3",
      llvm::cl::desc("Whether to enable HloShardingV3 which is the mesh and "
                     "axis based sharding representation."),
      llvm::cl::init(false)};

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    mlir::SymbolTableCollection symbolTableCollection;
    mlir::SymbolTable& symbolTable =
        symbolTableCollection.getSymbolTable(moduleOp);
    auto rewriter = mlir::IRRewriter(moduleOp.getContext());
    moduleOp->walk([&](sdy::ManualComputationOp manualComputation) {
      rewriter.setInsertionPointToEnd(&moduleOp.getRegion().front());
      mlir::Location loc = manualComputation.getLoc();
      mlir::Region& manualCompBody = manualComputation.getBody();
      mlir::TypeRange manualCompBodyArgTypes =
          manualCompBody.getArgumentTypes();
      mlir::TypeRange localResultTypes =
          sdy::getBodyTerminatorOpOperandTypes(manualComputation);
      auto funcOp = FuncOp::create(
          rewriter, loc, kManualComputationFuncName,
          rewriter.getFunctionType(manualCompBodyArgTypes, localResultTypes));
      mlir::StringAttr funcName = symbolTable.insert(funcOp);

      rewriter.setInsertionPoint(manualComputation);
      mlir::ValueRange operands = manualComputation->getOperands();
      if (!operands.empty()) {
        llvm::SmallVector<mlir::Value> globalOperands(operands.begin(),
                                                      operands.end());
        sdy::TensorShardingPerValueAttr inShardings =
            manualComputation.getInShardings();
        if (enableHloShardingV3) {
          inShardings = sdy::inlineMesh(symbolTable, inShardings);
          // `GlobalToLocalShape`'s result is the *local* tensor, so it can only
          // carry the local in shardings. Materialize the reshard that
          // `ManualComputationOp` performs implicitly on its operands as an
          // explicit `@Sharding` custom call. This keeps the exported module
          // sharding consistent - every op's sharding describes its own result
          // - and lets the import side read the global in shardings straight
          // off the operand instead of inferring them.
          for (auto [globalOperand, inSharding] :
               llvm::zip_equal(globalOperands, inShardings.getShardings())) {
            auto shardingOp = stablehlo::CustomCallOp::create(
                rewriter, loc, globalOperand.getType(), globalOperand);
            shardingOp.setCallTargetName(kShardingCustomCallTargetName);
            // Import reads the global in shardings back off these wrappers, so
            // they have to survive the HLO round trip intact. Without a side
            // effect a wrapper of a zero-element tensor is replaced by a
            // constant (see `ZeroSizedHloElimination`, which likewise spares
            // the side-effecting `xla.sdy.GlobalToLocalShape`), which would
            // lose that operand's sharding. The side effect also stops CSE from
            // merging wrappers that carry different shardings.
            shardingOp.setHasSideEffect(true);
            shardingOp->setAttr(mlir::sdy::kShardingAttr,
                                sdy::TensorShardingPerValueAttr::get(
                                    moduleOp.getContext(), inSharding));
            globalOperand = shardingOp.getResult(0);
          }
        }
        auto globalToLocalShape = stablehlo::CustomCallOp::create(
            rewriter, loc, manualCompBodyArgTypes, globalOperands);
        globalToLocalShape.setCallTargetName(kGlobalToLocalShapeCallTargetName);
        // We mark `xla.sdy.GlobalToLocalShape` as side-effecting to avoid
        // CSE deduping it with another taking the same operands, as it would
        // ignore the frontend attributes that could be different.
        globalToLocalShape.setHasSideEffect(true);
        if (enableHloShardingV3) {
          globalToLocalShape->setAttr(
              mlir::sdy::kShardingAttr,
              localizeShardings(inShardings, manualComputation.getManualAxes(),
                                moduleOp.getContext()));
          globalToLocalShape->setAttr(kManualAxes,
                                      manualComputation.getManualAxesAttr());
        } else {
          setFrontendAttribute(globalToLocalShape, kInShardings, inShardings);
          setFrontendAttribute(globalToLocalShape, kManualAxes,
                               manualComputation.getManualAxesAttr());
        }
        operands = globalToLocalShape->getResults();
      }

      sdy::TensorShardingPerValueAttr outShardings =
          manualComputation.getOutShardings();
      if (enableHloShardingV3) {
        outShardings = sdy::inlineMesh(symbolTable, outShardings);
      }

      auto callOp =
          CallOp::create(rewriter, loc, localResultTypes, funcName, operands);
      setFrontendAttribute(callOp, kXlaInlineableAttr,
                           rewriter.getStringAttr("xla_late"));
      if (enableHloShardingV3 && !outShardings.getShardings().empty()) {
        // The call op produces the local tensors, so it carries the local view
        // of the out shardings together with the manual axes. This is also the
        // only carrier of the manual axes when the manual computation has no
        // operands, i.e. when there is no `xla.sdy.GlobalToLocalShape`.
        callOp->setAttr(
            mlir::sdy::kShardingAttr,
            localizeShardings(outShardings, manualComputation.getManualAxes(),
                              moduleOp.getContext()));
        callOp->setAttr(kManualAxes, manualComputation.getManualAxesAttr());
      }

      mlir::ResultRange results = manualComputation->getResults();
      if (!results.empty()) {
        auto localToGlobalShape = stablehlo::CustomCallOp::create(
            rewriter, loc, manualComputation.getResultTypes(),
            callOp->getResults());
        localToGlobalShape.setCallTargetName(kLocalToGlobalShapeCallTargetName);
        // We mark `xla.sdy.LocalToGlobalShape` as side-effecting to avoid
        // CSE removing it if it has no users.
        localToGlobalShape.setHasSideEffect(true);
        if (enableHloShardingV3) {
          localToGlobalShape->setAttr(mlir::sdy::kShardingAttr, outShardings);
        } else {
          setFrontendAttribute(localToGlobalShape, kOutShardings, outShardings);
          setFrontendAttribute(localToGlobalShape, kManualAxes,
                               manualComputation.getManualAxesAttr());
        }
        results = localToGlobalShape->getResults();
      }
      sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
          manualCompBody, funcOp.getBody());
      rewriter.replaceOp(manualComputation, results);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-round-trip-shard-map-export";
  }

  StringRef getDescription() const override {
    return "Converts a `ManualComputationOp` to the following."
           "1. A separate function for the body of the `ManualComputationOp`."
           "2. A `CallOp` calling the function in #1, marked as not inlinable."
           "3. A pair of `CustomCallOp`s that change the shape of the "
           "   arguments/results."
           " Under HloShardingV3 the shardings and manual axes are attached as "
           "native attributes, and each operand is additionally wrapped in a "
           "`@Sharding` custom call holding the global in sharding so that "
           "every op's sharding describes its own result. Otherwise they are "
           "saved as frontend attrs on the pair of `CustomCallOp`s.";
  }
  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<stablehlo::StablehloDialect>();
  }
};

}  // namespace

void registerSdyRoundTripShardMapExportPass() {
  mlir::registerPass([]() { return createSdyRoundTripShardMapExportPass(); });
}

std::unique_ptr<mlir::Pass> createSdyRoundTripShardMapExportPass(
    bool enableHloShardingV3) {
  return std::make_unique<SdyRoundTripShardMapExportPass>(enableHloShardingV3);
}

}  // namespace sdy
}  // namespace xla
