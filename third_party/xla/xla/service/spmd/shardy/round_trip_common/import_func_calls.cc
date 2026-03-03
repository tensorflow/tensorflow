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

#include "xla/service/spmd/shardy/round_trip_common/import_func_calls.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Threading.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
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
#include "xla/service/spmd/shardy/utils.h"

namespace xla {
namespace sdy {

namespace {

using ::mlir::IRRewriter;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::CallOp;
using ::mlir::func::FuncOp;
using ::mlir::sdy::getTensorRank;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::NamedComputationOp;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;

// Returns the first non-maximal mesh on the argument shardings, if there is
// one. Otherwise returns `std::nullopt`.
// TODO(enver): Move to utils and potentially with a common helper that takes an
// std::function to get the sharding given an index.
std::optional<mlir::Attribute> getMeshOrRefOnArguments(
    FuncOp funcOp, const SymbolTable& symbolTable) {
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    if (TensorShardingAttr sdySharding =
            funcOp.getArgAttrOfType<TensorShardingAttr>(argNum, kShardingAttr);
        sdySharding && !sdySharding.getMesh(symbolTable).isMaximal()) {
      return std::make_optional(sdySharding.getMeshOrRef());
    }
  }
  return std::nullopt;
}

TensorShardingPerValueAttr getFuncArgShardings(CallOp callOp, FuncOp funcOp,
                                               const SymbolTable& symbolTable) {
  std::optional<mlir::Attribute> meshOrRef =
      getMeshOrRefOnArguments(funcOp, symbolTable);
  if (!meshOrRef) {
    return nullptr;
  }
  mlir::SmallVector<TensorShardingAttr> argShardings;
  argShardings.reserve(funcOp.getNumArguments());
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    TensorShardingAttr sdySharding =
        funcOp.getArgAttrOfType<TensorShardingAttr>(argNum, kShardingAttr);
    argShardings.push_back(sdySharding
                               ? sdySharding
                               : TensorShardingAttr::getFullyOpen(
                                     funcOp.getContext(),
                                     getTensorRank(callOp.getOperand(argNum)),
                                     *meshOrRef));
  }
  return TensorShardingPerValueAttr::get(funcOp.getContext(), argShardings);
}

void importCallOp(
    CallOp callOp,
    llvm::SmallDenseMap<StringRef, mlir::Region*>& calleeNameToMovedRegion,
    IRRewriter& rewriter, SymbolTable& symbolTable) {
  mlir::SmallVector<mlir::NamedAttribute> namedCompAttrs;
  llvm::copy_if(callOp->getDiscardableAttrs(),
                std::back_inserter(namedCompAttrs),
                [](const mlir::NamedAttribute& attr) {
                  return attr.getName() != kShardingAttr;
                });

  StringRef calleeName = callOp.getCallee();
  FuncOp funcOp = symbolTable.lookup<FuncOp>(calleeName);
  CHECK(funcOp) << "Failed to lookup function: " << calleeName.str();

  rewriter.setInsertionPoint(callOp);
  TensorShardingPerValueAttr callOpResultShardings =
      mlir::sdy::getShardingPerValue(callOp);
  auto namedCompOp = NamedComputationOp::create(
      rewriter, callOp->getLoc(), callOp->getResultTypes(), calleeName,
      callOp.getOperands(),
      /*inShardings=*/getFuncArgShardings(callOp, funcOp, symbolTable),
      // TODO(b/439018088): Take func result shardings if call op result
      // shardings are empty.
      /*outShardings=*/
      callOpResultShardings
          ? callOpResultShardings
          : getFuncResultShardings(callOp, funcOp, symbolTable));
  namedCompOp->setAttrs(namedCompAttrs);

  mlir::Region& namedCompRegion = namedCompOp.getRegion();
  if (auto movedRegionIt = calleeNameToMovedRegion.find(calleeName);
      movedRegionIt != calleeNameToMovedRegion.end()) {
    static llvm::once_flag onceFlag;
    mlir::sdy::emitOpWarningOnce(
        onceFlag, callOp,
        llvm::formatv("function @{0} has multiple call ops, we "
                      "need to clone the function body for each call",
                      calleeName)
            .str());
    rewriter.cloneRegionBefore(*movedRegionIt->second, namedCompRegion,
                               namedCompRegion.begin());
  } else {
    mlir::sdy::inlineRegionAndConvertTerminatorOp<mlir::sdy::ReturnOp>(
        funcOp.getBody(), namedCompRegion);
    calleeNameToMovedRegion[calleeName] = &namedCompRegion;
  }

  rewriter.replaceOp(callOp, namedCompOp);
}

class ImportFuncCallsPass
    : public mlir::PassWrapper<ImportFuncCallsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  using Base = mlir::PassWrapper<ImportFuncCallsPass,
                                 mlir::OperationPass<mlir::ModuleOp>>;

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportFuncCallsPass)

  ImportFuncCallsPass(bool enableNativeNonFlatSupport)
      : enableNativeNonFlatSupport(enableNativeNonFlatSupport) {}

  void runOnOperation() final {
    // TODO(enver): Inline for calls inside manual computations.
    if (enableNativeNonFlatSupport) {
      return;
    }

    mlir::ModuleOp moduleOp = getOperation();

    IRRewriter rewriter(moduleOp.getContext());
    SymbolTable symbolTable(moduleOp);
    // For every callee name, the first CallOp encountered with that symbol will
    // move the body of the callee into the created NamedComputationOp, and map
    // the symbol name to the moved region. Subsequent CallOps with that symbol
    // will clone the mapped region.
    llvm::SmallDenseMap<StringRef, mlir::Region*> calleeNameToMovedRegion;

    mlir::CallGraph callGraph(moduleOp);
    llvm::ReversePostOrderTraversal<const mlir::CallGraph*> rpo(&callGraph);
    for (mlir::CallGraphNode* node : llvm::reverse(rpo)) {
      if (node->isExternal()) {
        continue;
      }
      node->getCallableRegion()->walk([&](CallOp op) {
        importCallOp(op, calleeNameToMovedRegion, rewriter, symbolTable);
      });
    }

    // Erase all func ops that now have no call ops.
    for (auto [calleeName, _] : calleeNameToMovedRegion) {
      symbolTable.erase(symbolTable.lookup(calleeName));
    }
  }

  StringRef getArgument() const override { return "xla-sdy-import-func-calls"; }

  StringRef getDescription() const override {
    return "Creates a pass to convert a CallOp to a NamedComputationOp with "
           "the function body inlined and the name of the callee. Note that "
           "the func bodies are cloned if the func is used by multiple calls.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<mlir::sdy::SdyDialect>();
  }

  ImportFuncCallsPass() : Base() {}
  ImportFuncCallsPass(const ImportFuncCallsPass& other) : Base(other) {}
  ImportFuncCallsPass& operator=(const ImportFuncCallsPass&) = delete;
  ImportFuncCallsPass(ImportFuncCallsPass&&) = delete;
  ImportFuncCallsPass& operator=(ImportFuncCallsPass&&) = delete;
  ~ImportFuncCallsPass() override = default;

 private:
  bool enableNativeNonFlatSupport;
};

}  // namespace

std::unique_ptr<mlir::Pass> createImportFuncCallsPass(
    bool enableNativeNonFlatSupport) {
  return std::make_unique<ImportFuncCallsPass>(enableNativeNonFlatSupport);
}

void registerImportFuncCallsPass() {
  mlir::registerPass([] {
    return createImportFuncCallsPass(/*enableNativeNonFlatSupport=*/false);
  });
}

}  // namespace sdy
}  // namespace xla
