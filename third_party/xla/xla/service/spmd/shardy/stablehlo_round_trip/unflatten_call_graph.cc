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

#include "xla/service/spmd/shardy/stablehlo_round_trip/unflatten_call_graph.h"

#include <cstdint>
#include <memory>
#include <tuple>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Analysis/CallGraph.h"
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

using ::mlir::IRRewriter;
using ::mlir::ModuleOp;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::SdyDialect;

using ::mlir::sdy::getOriginalFuncName;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

using ComputationKey =
    std::tuple<StringAttr /*name*/, ManualAxesAttr /*manualAxes*/,
               TensorShardingPerValueAttr /*inShardings*/,
               TensorShardingPerValueAttr /*outShardings*/
               >;

namespace {
FuncOp getFuncOpOrDie(StringRef funcSymName, const SymbolTable& symbolTable) {
  FuncOp funcOp = symbolTable.lookup<FuncOp>(funcSymName);
  CHECK(funcOp) << "Failed to lookup function: " << funcSymName.str();
  return funcOp;
}
TensorShardingPerValueAttr getFuncArgShardings(FuncOp funcOp,
                                               const SymbolTable& symbolTable,
                                               bool ignoreShardings) {
  if (ignoreShardings) {
    return TensorShardingPerValueAttr();
  }
  return mlir::sdy::getFuncArgShardings(funcOp, symbolTable);
}
TensorShardingPerValueAttr getFuncResultShardings(
    FuncOp funcOp, const SymbolTable& symbolTable, bool ignoreShardings) {
  if (ignoreShardings) {
    return TensorShardingPerValueAttr();
  }
  return mlir::sdy::getFuncResultShardings(funcOp, symbolTable);
}

ManualAxesAttr getManualAxesAttr(FuncOp funcOp) {
  return funcOp->getAttrOfType<ManualAxesAttr>(mlir::sdy::kFuncManualAxes);
}

ComputationKey getComputationKey(FuncOp funcOp, const SymbolTable& symbolTable,
                                 bool ignoreShardings = false) {
  return {getOriginalFuncName(funcOp), getManualAxesAttr(funcOp),
          getFuncArgShardings(funcOp, symbolTable, ignoreShardings),
          getFuncResultShardings(funcOp, symbolTable, ignoreShardings)};
}

ComputationKey getComputationKey(CallOp callOp, const SymbolTable& symbolTable,
                                 bool ignoreShardings) {
  return getComputationKey(getFuncOpOrDie(callOp.getCallee(), symbolTable),
                           symbolTable, ignoreShardings);
}

llvm::SmallDenseMap<ComputationKey, FuncOp> populateFuncCache(
    ModuleOp moduleOp, const SymbolTable& symbolTable,
    bool dedupFunctionsFully) {
  llvm::SmallDenseMap<ComputationKey, FuncOp> funcCache;
  moduleOp.walk([&](CallOp callOp) {
    FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
    ComputationKey funcCacheKey = getComputationKey(
        funcOp, symbolTable, /*ignoreShardings=*/dedupFunctionsFully);
    // Keep the attribute for the original func name as other calls to the
    // same function would still need it to deduplicate.
    funcCache.try_emplace(funcCacheKey, funcOp);
  });

  // Count the calls sites and pick the funcOp with the largest calls.
  if (dedupFunctionsFully) {
    llvm::SmallDenseMap<ComputationKey, int64_t> callCounts;
    moduleOp.walk([&](CallOp callOp) {
      FuncOp funcOp = getFuncOpOrDie(callOp.getCallee(), symbolTable);
      ComputationKey funcCacheKey =
          getComputationKey(funcOp, symbolTable, /*ignoreShardings=*/true);

      // Increment the call count of `funcOp`.
      ComputationKey funcOpKey = getComputationKey(funcOp, symbolTable);
      callCounts[funcOpKey]++;

      // Update `funcCache` with `funcOp` if it has larger call count.
      auto cachedFuncOpIt = funcCache.find(funcCacheKey);
      ComputationKey cachedFuncOpKey =
          getComputationKey(cachedFuncOpIt->second, symbolTable);
      if (callCounts[funcOpKey] > callCounts[cachedFuncOpKey]) {
        cachedFuncOpIt->second = funcOp;
      }
    });
  }
  return funcCache;
}
}  // namespace

class UnflattenCallGraphPass
    : public mlir::PassWrapper<UnflattenCallGraphPass,
                               mlir::OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnflattenCallGraphPass)

  explicit UnflattenCallGraphPass(bool dedupFunctionsFully) {
    this->dedupFunctionsFully = dedupFunctionsFully;
  }

  UnflattenCallGraphPass() = default;

  explicit UnflattenCallGraphPass(const UnflattenCallGraphPass& other) {
    this->dedupFunctionsFully = other.dedupFunctionsFully;
  }

  // Unflattens the graph. It deduplicates functions with the same
  // input/output shardings *and* the same origin as desribed by the
  // 'original_func_name' attribute attached to the functions.
  //
  // However, when `dedupFunctionsFully` is enabled it disregards input/output
  // shardings and deduplicates all functions on the same origin. This means
  // it needs to pick one of the input/output shardings, and copy operations
  // before and after some calls in order to match the input/output shardings
  // the selected function expects. It currently picks the function arbitrarily.
  // TODO(enver): Pick the function that has the most callers, similar to
  // ExportNamedComputations pass.
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    IRRewriter rewriter(moduleOp.getContext());

    llvm::SmallDenseMap<ComputationKey, FuncOp> funcCache =
        populateFuncCache(moduleOp, symbolTable, dedupFunctionsFully);
    moduleOp.walk([&](CallOp callOp) {
      if (isManualComputation(callOp, /*isInlineable=*/true)) {
        return;
      }
      ComputationKey funcCacheKey = getComputationKey(
          callOp, symbolTable, /*ignoreShardings=*/dedupFunctionsFully);
      FuncOp funcOp = funcCache[funcCacheKey];
      callOp.setCallee(funcOp.getName());
      insertReshardsOnFuncArguments(funcOp, callOp, symbolTable, rewriter);
      insertReshardsOnFuncResults(funcOp, callOp, symbolTable, rewriter);
    });

    moduleOp.walk([&](FuncOp funcOp) {
      funcOp->removeAttr(mlir::sdy::kOriginalFuncName);
      funcOp->removeAttr(mlir::sdy::kFuncManualAxes);
    });
  }

  StringRef getArgument() const override {
    return "xla-sdy-unflatten-call-graph";
  }

  StringRef getDescription() const override {
    return "Unflattens the call graph.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect, mlir::mhlo::MhloDialect>();
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

std::unique_ptr<mlir::Pass> createUnflattenCallGraphPass(
    bool dedupFunctionsFully) {
  return std::make_unique<UnflattenCallGraphPass>(dedupFunctionsFully);
}

void registerUnflattenCallGraphPass() {
  mlir::registerPass([]() {
    return createUnflattenCallGraphPass(/*dedupFunctionsFully=*/false);
  });
}
}  // namespace sdy
}  // namespace xla
