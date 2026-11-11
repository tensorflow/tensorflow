/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/optimization_region_passes.h"

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/transforms/tfl_carrier.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace TFL {

namespace {

// True for a `custom_call @tfl.*` carrier (either HLO dialect).
bool IsCarrier(Operation* op) {
  if (auto call = dyn_cast<stablehlo::CustomCallOp>(op)) {
    return call.getCallTargetName().starts_with("tfl.");
  }
  if (auto call = dyn_cast<mhlo::CustomCallOp>(op)) {
    return call.getCallTargetName().starts_with("tfl.");
  }
  return false;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// The region attribute contract
////////////////////////////////////////////////////////////////////////////////

bool IsOptimizationRegion(stablehlo::CompositeOp composite) {
  return composite.getName() == kOptimizationRegionName;
}

bool IsOptimizeNoneRegion(func::FuncOp func) {
  auto attrs = func->getAttrOfType<DictionaryAttr>(kOptimizationFuncAttr);
  if (!attrs) return false;
  auto optimize = attrs.getAs<StringAttr>(kRegionOptimizeAttr);
  return optimize && optimize.getValue() == kRegionOptimizeNone;
}

FailureOr<bool> ParseOptimizeNone(DictionaryAttr attrs, Operation* op) {
  if (!attrs || attrs.empty()) return false;
  // A frontend contract: a misspelt key or an unsupported selection must not
  // silently mean "default". The vocabulary of per-entry selection
  // (`disable`, `enable`, other `optimize` values) is reserved.
  auto unsupported =
      [&]() {
        return op->emitError()
               << "odml.optimization_region: only optimize = \"none\" is "
                  "supported; per-entry selection is a separate change (got "
               << attrs << ")";
      };
  if (attrs.size() != 1) return unsupported();
  auto optimize = attrs.getAs<StringAttr>(kRegionOptimizeAttr);
  if (!optimize || optimize.getValue() != kRegionOptimizeNone) {
    return unsupported();
  }
  return true;
}

namespace {

// Stamps the decomposition of every `odml.optimization_region` `CompositeOp`
// (`stablehlo.composite` or `mhlo.composite`: which one a region is at the
// time of marking depends on the pipeline).
template <typename CompositeOp>
LogicalResult StampOptimizationRegionsOf(ModuleOp module) {
  SymbolTable symbol_table(module);
  Builder builder(module.getContext());
  bool failed_parse = false;
  module.walk([&](CompositeOp composite) {
    if (composite.getName() != kOptimizationRegionName) return;
    auto decomposition =
        symbol_table.lookup<func::FuncOp>(composite.getDecomposition());
    if (!decomposition) return;
    FailureOr<bool> none =
        ParseOptimizeNone(composite.getCompositeAttributes(), composite);
    if (failed(none)) {
      failed_parse = true;
      return;
    }
    DictionaryAttr stamp =
        *none ? builder.getDictionaryAttr(builder.getNamedAttr(
                    kRegionOptimizeAttr,
                    builder.getStringAttr(kRegionOptimizeNone)))
              : builder.getDictionaryAttr({});
    if (Attribute existing = decomposition->getAttr(kOptimizationFuncAttr)) {
      if (existing != stamp) {
        composite.emitError()
            << "odml.optimization_region decomposition @"
            << decomposition.getName()
            << " is used with conflicting odml.optimization_region attributes";
        failed_parse = true;
      }
      return;
    }
    decomposition->setAttr(kOptimizationFuncAttr, stamp);
  });
  return failure(failed_parse);
}

}  // namespace

LogicalResult StampOptimizationRegions(ModuleOp module) {
  if (failed(StampOptimizationRegionsOf<stablehlo::CompositeOp>(module))) {
    return failure();
  }
  return StampOptimizationRegionsOf<mhlo::CompositeOp>(module);
}

////////////////////////////////////////////////////////////////////////////////
// MarkOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

void MarkOptimizationRegionsPass::runOnOperation() {
  if (failed(StampOptimizationRegions(getOperation()))) signalPassFailure();
}

////////////////////////////////////////////////////////////////////////////////
// MaterializeOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

namespace {

// The carrier convention (see tfl_carrier.h): the call target is the TFL op
// name and backend_config is the op's attribute dictionary in MLIR syntax.
// Carriers are `mhlo.custom_call` right after legalization and
// `stablehlo.custom_call` once the remaining HLO has been converted back; both
// are handled.
template <typename CustomCallOp>
LogicalResult MaterializeCarriers(func::FuncOp func) {
  SmallVector<CustomCallOp> carriers;
  func.walk([&](CustomCallOp call) {
    if (call.getCallTargetName().starts_with("tfl.")) carriers.push_back(call);
  });
  for (CustomCallOp call : carriers) {
    llvm::StringRef name = call.getCallTargetName();
    // Q/DQ carriers are never kept opaque (see IsCustomCallLegal in
    // legalize_hlo_conversions/custom_call.cc): their legalization builds the
    // quantized result type from the parameters in backend_config, which the
    // generic materialization below cannot reproduce.
    if (name == "tfl.quantize" || name == "tfl.dequantize") {
      return call.emitError()
             << "odml.optimization_region: carrier for " << name
             << " reached materialization; quantize/dequantize must be "
                "legalized normally";
    }
    // A misspelt or unknown op name would otherwise become an unregistered
    // op that fails much later, far from the carrier.
    if (!RegisteredOperationName::lookup(name, call.getContext())) {
      return call.emitError()
             << "odml.optimization_region: carrier names unknown op '" << name
             << "'";
    }
    DictionaryAttr attributes = ParseSerializedTFLOpAttributes(
        call.getBackendConfig(), call.getContext());
    if (!attributes) {
      return call.emitError() << "odml.optimization_region: carrier for "
                              << name << " has no parsable backend_config";
    }
    OpBuilder builder(call);
    OperationState state(call.getLoc(), name, call.getOperands(),
                         call.getResultTypes(), attributes.getValue());
    Operation* op = builder.create(state);
    call.replaceAllUsesWith(op);
    call.erase();
  }
  return success();
}

}  // namespace

void MaterializeOptimizationRegionsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!IsOptimizeNoneRegion(func)) return;
  if (failed(MaterializeCarriers<mhlo::CustomCallOp>(func)) ||
      failed(MaterializeCarriers<stablehlo::CustomCallOp>(func))) {
    signalPassFailure();
  }
}

////////////////////////////////////////////////////////////////////////////////
// SkipOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

SkipOptimizationRegionsPass::SkipOptimizationRegionsPass(
    const SkipOptimizationRegionsPass& other)
    : PassWrapper(other) {
  pipeline_ = other.pipeline_;
  if (other.pm_) pm_ = std::make_unique<OpPassManager>(*other.pm_);
}

SkipOptimizationRegionsPass::SkipOptimizationRegionsPass(
    std::unique_ptr<mlir::Pass> inner)
    : pm_(std::make_unique<OpPassManager>(func::FuncOp::getOperationName(),
                                          OpPassManager::Nesting::Implicit)) {
  pm_->addPass(std::move(inner));
}

LogicalResult SkipOptimizationRegionsPass::initializeOptions(
    StringRef options,
    function_ref<LogicalResult(const Twine&)> error_handler) {
  if (failed(PassWrapper::initializeOptions(options, error_handler))) {
    return failure();
  }
  if (pipeline_.empty()) {
    if (!pm_)
      return error_handler("tfl-skip-optimization-regions needs a pipeline");
    return success();
  }
  pm_ = std::make_unique<OpPassManager>(func::FuncOp::getOperationName(),
                                        OpPassManager::Nesting::Implicit);
  return parsePassPipeline(pipeline_, *pm_, llvm::errs());
}

void SkipOptimizationRegionsPass::getDependentDialects(
    DialectRegistry& registry) const {
  if (pm_) pm_->getDependentDialects(registry);
}

void SkipOptimizationRegionsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (IsOptimizeNoneRegion(func)) {
    markAllAnalysesPreserved();
    return;
  }
  if (!pm_) return;
  if (failed(runPipeline(*pm_, func))) signalPassFailure();
}

std::unique_ptr<mlir::Pass> CreateSkipOptimizationRegionsPass(
    std::unique_ptr<mlir::Pass> inner) {
  return std::make_unique<SkipOptimizationRegionsPass>(std::move(inner));
}

////////////////////////////////////////////////////////////////////////////////
// InlineOptimizationRegionsPass
////////////////////////////////////////////////////////////////////////////////

namespace {

// Clones `callee`'s single block in place of `composite`.
LogicalResult InlineComposite(stablehlo::CompositeOp composite,
                              func::FuncOp callee) {
  if (callee.getBody().empty()) {
    return composite.emitError()
           << "odml.optimization_region composite has no decomposition";
  }
  // Decompositions are single-block functions by construction; anything
  // else cannot be inlined by block cloning.
  if (!llvm::hasSingleElement(callee.getBody())) {
    return composite.emitError()
           << "odml.optimization_region decomposition must have a single block";
  }
  Block& body = callee.getBody().front();
  auto return_op = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!return_op) {
    return composite.emitError()
           << "odml.optimization_region decomposition must end in func.return";
  }

  OpBuilder builder(composite);
  IRMapping mapper;
  for (auto [arg, operand] :
       llvm::zip(body.getArguments(), composite.getOperands())) {
    mapper.map(arg, operand);
  }
  for (Operation& op : body.without_terminator()) {
    builder.clone(op, mapper);
  }
  SmallVector<Value> results;
  for (Value v : return_op.getOperands()) {
    results.push_back(mapper.lookup(v));
  }
  composite.replaceAllUsesWith(results);
  composite.erase();
  return success();
}

}  // namespace

void InlineOptimizationRegionsPass::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbol_table(module);

  // Carriers must have been materialized by now; an opaque custom_call must
  // never reach the exporter.
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!IsOptimizeNoneRegion(func)) continue;
    Operation* carrier = nullptr;
    func.walk([&](Operation* op) {
      if (!carrier && IsCarrier(op)) carrier = op;
    });
    if (carrier) {
      carrier->emitError()
          << "odml.optimization_region: carrier was not materialized before "
             "inlining (MaterializeOptimizationRegionsPass must run first)";
      return signalPassFailure();
    }
  }

  // Innermost first: a composite whose decomposition contains no further
  // region composite can be inlined; repeat until none is left. The body of
  // an inlined region may itself contain composites (nested regions) that
  // then become inlineable in a later round.
  SetVector<func::FuncOp> decompositions;
  while (true) {
    SmallVector<stablehlo::CompositeOp> composites;
    module.walk([&](stablehlo::CompositeOp composite) {
      if (IsOptimizationRegion(composite)) composites.push_back(composite);
    });
    if (composites.empty()) break;

    bool progress = false;
    for (stablehlo::CompositeOp composite : composites) {
      auto callee =
          symbol_table.lookup<func::FuncOp>(composite.getDecomposition());
      if (!callee) {
        composite.emitError()
            << "odml.optimization_region composite has no decomposition";
        return signalPassFailure();
      }
      bool nested = false;
      callee.walk([&](stablehlo::CompositeOp inner) {
        if (IsOptimizationRegion(inner)) nested = true;
      });
      if (nested) continue;
      if (failed(InlineComposite(composite, callee))) {
        return signalPassFailure();
      }
      decompositions.insert(callee);
      progress = true;
    }
    if (!progress) {
      composites.front().emitError()
          << "odml.optimization_region: regions are mutually nested and "
             "cannot be inlined";
      return signalPassFailure();
    }
  }

  // Drop decomposition functions that no longer have any users. The same
  // function can back several composites, so check after all inlining.
  for (func::FuncOp callee : decompositions) {
    if (SymbolTable::symbolKnownUseEmpty(callee, module)) {
      callee.erase();
    }
  }
}

}  // namespace TFL
}  // namespace mlir
