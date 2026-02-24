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

#include "xla/service/spmd/shardy/round_trip_common/export_named_computations.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayAttr;
using ::mlir::ModuleOp;
using ::mlir::NamedAttribute;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::SymbolTableCollection;
using ::mlir::SymbolUserMap;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;

using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::NamedComputationOp;
using ::mlir::sdy::TensorShardingPerValueAttr;

using ComputationKey = std::tuple<StringRef, TensorShardingPerValueAttr,
                                  TensorShardingPerValueAttr, ManualAxesAttr>;

StringAttr createFuncOp(NamedComputationOp namedComputationOp,
                        mlir::IRRewriter& rewriter, SymbolTable& symbolTable,
                        std::optional<TensorShardingPerValueAttr> inShardings,
                        std::optional<TensorShardingPerValueAttr> outShardings,
                        ManualAxesAttr manualAxesAttr) {
  auto funcOp = FuncOp::create(
      rewriter, namedComputationOp.getLoc(), namedComputationOp.getName(),
      rewriter.getFunctionType(namedComputationOp.getBody().getArgumentTypes(),
                               namedComputationOp.getResultTypes()),
      rewriter.getStringAttr("private"),
      /*argAttrs=*/ArrayAttr(), /*resultAttrs=*/ArrayAttr());

  rewriter.setInsertionPointToStart(funcOp->getBlock());
  mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::func::ReturnOp>(
      namedComputationOp.getBody(), funcOp.getBody());

  // Copy the input shardings to the func.
  if (inShardings.has_value()) {
    for (auto [i, sharding] : llvm::enumerate(inShardings->getShardings())) {
      funcOp.setArgAttr(i, kShardingAttr, sharding);
      if (manualAxesAttr) {
        funcOp.setArgAttr(i, kManualAxes, manualAxesAttr);
      }
    }
  }

  // Copy the output shardings to the func.
  if (outShardings.has_value()) {
    for (auto [i, sharding] : llvm::enumerate(outShardings->getShardings())) {
      funcOp.setResultAttr(i, kShardingAttr, sharding);
      if (manualAxesAttr) {
        funcOp.setResultAttr(i, kManualAxes, manualAxesAttr);
      }
    }
  }
  return symbolTable.insert(funcOp);
}

StringAttr createFuncOpOrGetFromCache(
    NamedComputationOp namedComputationOp,
    llvm::SmallDenseMap<ComputationKey, StringAttr>& funcCache,
    mlir::IRRewriter& rewriter, SymbolTable& symbolTable,
    ManualAxesAttr manualAxesAttr,
    std::optional<TensorShardingPerValueAttr> inShardings,
    std::optional<TensorShardingPerValueAttr> outShardings,
    bool dedupFunctionsFully) {
  ComputationKey key = {
      namedComputationOp.getName(),
      dedupFunctionsFully ? TensorShardingPerValueAttr()
                          : inShardings.value_or(TensorShardingPerValueAttr()),
      dedupFunctionsFully ? TensorShardingPerValueAttr()
                          : outShardings.value_or(TensorShardingPerValueAttr()),
      manualAxesAttr};
  if (auto it = funcCache.find(key); it != funcCache.end()) {
    return it->second;
  }
  StringAttr funcSymName =
      createFuncOp(namedComputationOp, rewriter, symbolTable, inShardings,
                   outShardings, manualAxesAttr);
  funcCache.try_emplace(key, funcSymName);
  return funcSymName;
}

// Converts a `NamedComputationOp` into a `CallOp`.
class ExportNamedComputationsPass
    : public mlir::PassWrapper<ExportNamedComputationsPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportNamedComputationsPass)

  explicit ExportNamedComputationsPass(bool dedupFunctionsFully) {
    this->dedupFunctionsFully = dedupFunctionsFully;
  }

  ExportNamedComputationsPass() = default;

  explicit ExportNamedComputationsPass(
      const ExportNamedComputationsPass& other) {
    this->dedupFunctionsFully = other.dedupFunctionsFully;
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);
    mlir::Block& moduleBlock = moduleOp.getRegion().front();
    llvm::SmallDenseMap<ComputationKey, StringAttr> funcCache;

    if (dedupFunctionsFully) {
      using FuncNameKey = std::pair<StringRef, ManualAxesAttr>;
      llvm::SmallDenseMap<ComputationKey, int64_t> funcCallSiteCounts;
      llvm::SmallDenseMap<FuncNameKey, std::pair<NamedComputationOp, int64_t>>
          funcToNamedComputations;
      // TODO(enver): Instead of a SmallDenseMap and a separate SmallVector to
      // guarantee a deterministic iteration order, consider using
      // llvm::MapVector.
      // Required to iterate on functions in a deterministic order.
      llvm::SmallVector<FuncNameKey> funcNames;
      moduleOp.walk([&](NamedComputationOp namedComputationOp) {
        ManualAxesAttr manualAxesAttr =
            namedComputationOp->getAttrOfType<ManualAxesAttr>(kManualAxes);
        ComputationKey key = {namedComputationOp.getName(),
                              namedComputationOp.getInShardings().value_or(
                                  TensorShardingPerValueAttr()),
                              namedComputationOp.getOutShardings().value_or(
                                  TensorShardingPerValueAttr()),
                              manualAxesAttr};
        const int64_t callSiteCount = funcCallSiteCounts[key]++;
        FuncNameKey funcNameKey =
            std::pair(namedComputationOp.getName(), manualAxesAttr);
        if (auto [it, inserted] = funcToNamedComputations.try_emplace(
                funcNameKey, namedComputationOp, callSiteCount);
            !inserted) {
          auto& [cachedNamedComputationOp, cachedCallSiteCount] = it->second;
          if (callSiteCount > cachedCallSiteCount) {
            cachedNamedComputationOp = namedComputationOp;
            cachedCallSiteCount = callSiteCount;
          }
        } else {  // inserted is true.
          funcNames.push_back(funcNameKey);
        }
      });

      for (FuncNameKey funcNameKey : funcNames) {
        auto& [namedComputationOp, callSiteCount] =
            funcToNamedComputations.find(funcNameKey)->second;
        mlir::IRRewriter rewriter(namedComputationOp);
        rewriter.setInsertionPointToEnd(&moduleBlock);
        ManualAxesAttr manualAxesAttr =
            namedComputationOp->getAttrOfType<ManualAxesAttr>(kManualAxes);
        StringAttr funcSymName =
            createFuncOp(namedComputationOp, rewriter, symbolTable,
                         namedComputationOp.getInShardings(),
                         namedComputationOp.getOutShardings(), manualAxesAttr);
        funcCache.try_emplace(
            ComputationKey{namedComputationOp.getName(),
                           TensorShardingPerValueAttr(),
                           TensorShardingPerValueAttr(), manualAxesAttr},
            funcSymName);
      }
    }

    // NOTE: The walk needs to be in post order, which is the default order, to
    // account for nested named computations.
    moduleOp.walk([&](NamedComputationOp namedComputationOp) {
      mlir::IRRewriter rewriter(namedComputationOp);
      rewriter.setInsertionPointToEnd(&moduleBlock);

      ManualAxesAttr manualAxesAttr =
          namedComputationOp->getAttrOfType<ManualAxesAttr>(kManualAxes);
      std::optional<TensorShardingPerValueAttr> inShardings =
          namedComputationOp.getInShardings();
      std::optional<TensorShardingPerValueAttr> outShardings =
          namedComputationOp.getOutShardings();
      if (manualAxesAttr) {
        CHECK(!manualAxesAttr.empty());
        CHECK(inShardings.has_value());
        CHECK(outShardings.has_value());
      }
      StringAttr funcSymName = createFuncOpOrGetFromCache(
          namedComputationOp, funcCache, rewriter, symbolTable, manualAxesAttr,
          inShardings, outShardings, dedupFunctionsFully);

      // Replace the `NamedComputationOp` with a `CallOp`.
      rewriter.setInsertionPoint(namedComputationOp);
      mlir::SmallVector<NamedAttribute> callOpAttrs(
          namedComputationOp->getDiscardableAttrs());
      auto callOp = rewriter.replaceOpWithNewOp<CallOp>(
          namedComputationOp, namedComputationOp.getResultTypes(), funcSymName,
          namedComputationOp.getOperands());
      callOp->setAttrs(callOpAttrs);

      // Copy the func output shardings to the call op.
      FuncOp funcOp = symbolTable.lookup<FuncOp>(funcSymName);
      if (TensorShardingPerValueAttr funcResultShardings =
              getFuncResultShardings(callOp, funcOp, symbolTable);
          funcResultShardings) {
        mlir::sdy::setShardings(callOp, funcResultShardings);
        if (outShardings.has_value()) {
          for (auto [funcResultSharding, outSharding, result] : llvm::zip_equal(
                   funcResultShardings.getShardings(),
                   outShardings->getShardings(), callOp.getResults())) {
            if (!funcResultSharding.isEquivalent(outSharding)) {
              rewriter.setInsertionPointAfterValue(result);
              auto copyOp =
                  mlir::mhlo::CopyOp::create(rewriter, result.getLoc(), result);
              mlir::sdy::setShardings(copyOp, outSharding);
              rewriter.replaceAllUsesExcept(result, copyOp, copyOp);
            }
          }
        }
        if (manualAxesAttr) {
          callOp->setAttr(kManualAxes, manualAxesAttr);
        }
      }
    });

    // Drop uncalled inlineable manual computation funcs.
    // TODO(enver): Drop generically, not just inlined manual computation funcs.
    llvm::SmallVector<FuncOp> uncalledInlineableManualComputationFuncs;
    SymbolUserMap symbolUserMap(symbolTableCollection, moduleOp);
    for (FuncOp funcOp : moduleOp.getOps<FuncOp>()) {
      if (StringRef funcSymName = funcOp.getName();
          funcSymName.contains(kInlineableManualComputationFuncName) &&
          symbolUserMap.useEmpty(funcOp)) {
        uncalledInlineableManualComputationFuncs.push_back(funcOp);
      }
    }
    // TODO(enver): Erase directly without collecting on a vector.
    for (FuncOp funcOp : uncalledInlineableManualComputationFuncs) {
      symbolTable.erase(funcOp);
    }
  }

  StringRef getArgument() const override {
    return "xla-sdy-export-named-computations";
  }

  StringRef getDescription() const override {
    return "Converts a `NamedComputationOp` to a `CallOp` with a new private "
           "function called the `NamedComputationOp`'s `name`. The new "
           "`FuncOp` and `CallOp` have the same shardings as the original "
           "`NamedComputationOp`s operands/results.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect, mlir::mhlo::MhloDialect>();
  }

  Option<bool> dedupFunctionsFully{
      *this, "dedup-functions-fully",
      llvm::cl::desc(
          "If true, regardless of the input and output shardings of functions, "
          "it keeps one callee function for each caller function. The default "
          "is false, meaning it will deduplicate only if the input and output "
          "shardings are the same."),
      llvm::cl::init(false)};
};

}  // namespace

std::unique_ptr<mlir::Pass> createExportNamedComputationsPass(
    bool dedupFunctionsFully) {
  return std::make_unique<ExportNamedComputationsPass>(dedupFunctionsFully);
}

void registerExportNamedComputationsPass() {
  mlir::registerPass(std::make_unique<ExportNamedComputationsPass>);
}

}  // namespace sdy
}  // namespace xla
