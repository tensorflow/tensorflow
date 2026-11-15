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

#include "tensorflow/compiler/mlir/lite/python/preserve_regions.h"

#include <cassert>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/WalkResult.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir::TFL {
namespace {

bool IsPreserveRegion(mlir::Operation* op) {
  auto composite = dyn_cast_or_null<stablehlo::CompositeOp>(op);
  return composite && composite.getName() == kPreserveRegionCompositeName;
}

SmallVector<stablehlo::CompositeOp> CollectPreserveRegions(
    mlir::Operation* root) {
  SmallVector<stablehlo::CompositeOp> composites;
  root->walk([&](stablehlo::CompositeOp composite) {
    if (IsPreserveRegion(composite)) composites.push_back(composite);
  });
  return composites;
}

// Returns the names of all symbols referenced from inside `module`.
llvm::StringSet<> ReferencedSymbols(ModuleOp module) {
  llvm::StringSet<> referenced;
  if (auto uses = SymbolTable::getSymbolUses(&module.getBodyRegion())) {
    for (const SymbolTable::SymbolUse& use : *uses) {
      referenced.insert(use.getSymbolRef().getRootReference().getValue());
    }
  }
  return referenced;
}

// True for an op that only materializes a constant and can be cloned freely.
bool IsClonableConstant(mlir::Operation* op) {
  if (op == nullptr || op->getNumOperands() != 0 || op->getNumResults() != 1 ||
      op->getNumRegions() != 0) {
    return false;
  }
  // `tfl.pseudo_qconst` and friends are not `ConstantLike`.
  return op->hasTrait<OpTrait::ConstantLike>() ||
         isa<TFL::QConstOp, TFL::SparseConstOp, TFL::SparseQConstOp>(op);
}

// Replaces `composite` with a copy of the body of `callee`. Fails if `callee`
// is not a single-block function ending in `func.return`.
LogicalResult InlineBody(stablehlo::CompositeOp composite,
                         func::FuncOp callee) {
  if (callee.isExternal() || !llvm::hasSingleElement(callee.getBody())) {
    return composite.emitError()
           << kPreserveRegionCompositeName << ": decomposition @"
           << callee.getSymName() << " must have exactly one block";
  }
  Block& body = callee.getBody().front();
  auto return_op = dyn_cast<func::ReturnOp>(body.getTerminator());
  if (!return_op) {
    return composite.emitError()
           << kPreserveRegionCompositeName << ": decomposition @"
           << callee.getSymName() << " must end in func.return";
  }

  OpBuilder builder(composite);
  IRMapping mapping;
  mapping.map(body.getArguments(), composite.getOperands());
  for (mlir::Operation& op : body.without_terminator())
    builder.clone(op, mapping);
  SmallVector<mlir::Value> results;
  for (mlir::Value value : return_op.getOperands()) {
    results.push_back(mapping.lookupOrDefault(value));
  }
  composite.replaceAllUsesWith(results);
  composite.erase();
  return success();
}

// Fails if the preserve-region decompositions reachable from `func` through
// nested preserve regions form a cycle, or if one of them does not resolve.
// `on_stack` and `done` carry the DFS state across calls.
LogicalResult CheckNestingIsAcyclic(func::FuncOp func, SymbolTable& symbols,
                                    llvm::DenseSet<mlir::Operation*>& on_stack,
                                    llvm::DenseSet<mlir::Operation*>& done) {
  if (done.contains(func)) return success();
  on_stack.insert(func);
  for (stablehlo::CompositeOp composite : CollectPreserveRegions(func)) {
    auto callee = symbols.lookup<func::FuncOp>(composite.getDecomposition());
    if (!callee) {
      return composite.emitError()
             << kPreserveRegionCompositeName << ": decomposition @"
             << composite.getDecomposition() << " not found";
    }
    if (on_stack.contains(callee)) {
      return composite.emitError()
             << kPreserveRegionCompositeName << ": decomposition @"
             << callee.getSymName() << " is recursive";
    }
    if (failed(CheckNestingIsAcyclic(callee, symbols, on_stack, done))) {
      return failure();
    }
  }
  on_stack.erase(func);
  done.insert(func);
  return success();
}

// Inlines all preserve regions nested in `func`, innermost first. Nested
// regions carry the same contract as the enclosing one, so flattening them
// changes nothing about the result. The nesting must have been checked with
// `CheckNestingIsAcyclic`.
LogicalResult FlattenNestedRegions(func::FuncOp func, SymbolTable& symbols) {
  for (SmallVector<stablehlo::CompositeOp> nested =
           CollectPreserveRegions(func);
       !nested.empty(); nested = CollectPreserveRegions(func)) {
    for (stablehlo::CompositeOp composite : nested) {
      auto callee = symbols.lookup<func::FuncOp>(composite.getDecomposition());
      assert(callee && "nesting must be checked with CheckNestingIsAcyclic");
      if (failed(InlineBody(composite, callee))) return failure();
    }
  }
  return success();
}

// Clones the constant operands of `composite` into the entry block of `body`
// and redirects the uses of the corresponding arguments to the clones. The
// arguments themselves stay, so the signature still matches the composite.
void SinkConstantOperands(stablehlo::CompositeOp composite, func::FuncOp body) {
  OpBuilder builder = OpBuilder::atBlockBegin(&body.front());
  for (auto [operand, argument] :
       llvm::zip(composite.getOperands(), body.getArguments())) {
    mlir::Operation* def = operand.getDefiningOp();
    if (!IsClonableConstant(def) || argument.use_empty()) continue;
    mlir::Operation* clone = builder.clone(*def);
    argument.replaceAllUsesWith(clone->getResult(0));
  }
}

// Erases the ops in `roots` and, transitively, their operands' defining ops,
// as long as they are trivially dead.
void EraseTriviallyDead(llvm::ArrayRef<mlir::Operation*> roots) {
  llvm::SetVector<mlir::Operation*> worklist(roots.begin(), roots.end());
  llvm::DenseSet<mlir::Operation*> erased;
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (erased.contains(op) || !isOpTriviallyDead(op)) continue;
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation* def = operand.getDefiningOp()) worklist.insert(def);
    }
    erased.insert(op);
    op->erase();
  }
}

std::string TypesToString(TypeRange types) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "(";
  llvm::interleaveComma(types, os);
  os << ")";
  return out;
}

// A region call site and its body, prepared before `module` is modified.
struct Site {
  stablehlo::CompositeOp composite;
  func::FuncOp callee;
  OwningOpRef<func::FuncOp> body;
};

}  // namespace

bool HasPreserveRegions(ModuleOp module) {
  bool found = false;
  module.walk([&](stablehlo::CompositeOp composite) {
    if (!IsPreserveRegion(composite)) return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

FailureOr<OwningOpRef<ModuleOp>> SplitPreserveRegions(ModuleOp module) {
  SymbolTable main_symbols(module);

  // Resolve every region and check the nesting before modifying anything.
  llvm::SetVector<mlir::Operation*> decompositions;
  llvm::DenseSet<mlir::Operation*> on_stack, done;
  for (stablehlo::CompositeOp composite : CollectPreserveRegions(module)) {
    auto callee =
        main_symbols.lookup<func::FuncOp>(composite.getDecomposition());
    if (!callee) {
      return composite.emitError()
             << kPreserveRegionCompositeName << ": decomposition @"
             << composite.getDecomposition() << " not found";
    }
    if (failed(CheckNestingIsAcyclic(callee, main_symbols, on_stack, done))) {
      return failure();
    }
    decompositions.insert(callee);
  }

  // A decomposition with nested regions is flattened into copies of itself;
  // the nested composites in the original are never retargeted. That is only
  // fine if the original goes away: it must be private (so it is erased once
  // unused) and nothing but preserve regions (which get the copies) may refer
  // to it.
  for (mlir::Operation* op : decompositions) {
    auto func = cast<func::FuncOp>(op);
    if (CollectPreserveRegions(func).empty()) continue;
    if (!func.isPrivate()) {
      return func.emitError()
             << kPreserveRegionCompositeName << ": decomposition @"
             << func.getSymName()
             << " contains nested preserve regions and is public, which is "
                "not supported";
    }
    auto uses = SymbolTable::getSymbolUses(func, module);
    if (!uses) continue;
    for (const SymbolTable::SymbolUse& use : *uses) {
      if (IsPreserveRegion(use.getUser())) continue;
      return use.getUser()->emitError()
             << kPreserveRegionCompositeName << ": decomposition @"
             << func.getSymName()
             << " contains nested preserve regions and is also referenced "
                "outside of a preserve region, which is not supported";
    }
  }

  // Prepare one body per call site, without touching `module` yet. Call sites
  // inside decompositions are nested regions and get flattened into the body
  // of their enclosing region instead.
  SmallVector<Site> sites;
  for (stablehlo::CompositeOp composite : CollectPreserveRegions(module)) {
    if (decompositions.contains(
            composite->getParentOfType<func::FuncOp>().getOperation())) {
      continue;
    }
    auto callee =
        main_symbols.lookup<func::FuncOp>(composite.getDecomposition());
    OwningOpRef<func::FuncOp> body = callee.clone();
    if (failed(FlattenNestedRegions(*body, main_symbols))) return failure();

    // The body moves to another module; anything else it refers to (e.g. the
    // decomposition of another kind of composite) would dangle there.
    auto uses = SymbolTable::getSymbolUses(&body->getBody());
    if (uses && !uses->empty()) {
      return composite.emitError()
             << kPreserveRegionCompositeName << ": region body @"
             << callee.getSymName() << " refers to symbol "
             << uses->begin()->getSymbolRef()
             << "; only nested preserve regions are supported";
    }
    sites.push_back({composite, callee, std::move(body)});
  }

  // Move the bodies. One copy per call site: constants are sunk per site, and
  // the body may be transformed differently depending on them. The main
  // module keeps a private declaration under the same name, so that the
  // composite stays valid while the main pipeline runs.
  OwningOpRef<ModuleOp> regions = ModuleOp::create(module.getLoc());
  SymbolTable region_symbols(*regions);
  int index = 0;
  for (Site& site : sites) {
    auto placeholder = func::FuncOp::create(
        site.callee.getLoc(),
        (site.callee.getSymName() + ".preserve_region." + llvm::Twine(index++))
            .str(),
        site.callee.getFunctionType());
    placeholder.setPrivate();
    StringAttr name = main_symbols.insert(placeholder);
    site.composite.setDecomposition(name.getValue());

    func::FuncOp body = site.body.release();
    body.setPrivate();
    body.setSymName(name.getValue());
    [[maybe_unused]] StringAttr region_name = region_symbols.insert(body);
    assert(region_name == name && "region body must keep the placeholder name");
    SinkConstantOperands(site.composite, body);
  }

  // Drop private decompositions nothing refers to anymore. Erasing an outer
  // body can leave a nested one unused, so repeat until nothing changes.
  SmallVector<mlir::Operation*> candidates(decompositions.begin(),
                                           decompositions.end());
  for (bool erased = true; erased;) {
    erased = false;
    llvm::StringSet<> referenced = ReferencedSymbols(module);
    for (mlir::Operation*& op : candidates) {
      if (op == nullptr) continue;
      auto func = cast<func::FuncOp>(op);
      if (!func.isPrivate() || referenced.contains(func.getSymName())) {
        continue;
      }
      main_symbols.erase(op);
      op = nullptr;
      erased = true;
    }
  }
  return regions;
}

LogicalResult StitchPreserveRegions(ModuleOp module, ModuleOp regions) {
  SymbolTable main_symbols(module);
  SymbolTable region_symbols(regions);

  llvm::SetVector<mlir::Operation*> placeholders;
  for (stablehlo::CompositeOp composite : CollectPreserveRegions(module)) {
    StringRef name = composite.getDecomposition();
    auto body = region_symbols.lookup<func::FuncOp>(name);
    if (!body) {
      return composite.emitError()
             << kPreserveRegionCompositeName << ": no region body named @"
             << name << "; was the module split with SplitPreserveRegions?";
    }
    if (TypeRange(composite.getOperandTypes()) !=
            TypeRange(body.getArgumentTypes()) ||
        TypeRange(composite.getResultTypes()) !=
            TypeRange(body.getResultTypes())) {
      return composite.emitError()
             << kPreserveRegionCompositeName
             << ": region body signature does not match the composite after "
                "optimization; body @"
             << name << " has type " << TypesToString(body.getArgumentTypes())
             << " -> " << TypesToString(body.getResultTypes())
             << ", composite has type "
             << TypesToString(composite.getOperandTypes()) << " -> "
             << TypesToString(composite.getResultTypes())
             << ". Both pipelines must legalize boundary types the same way.";
    }
    if (mlir::Operation* placeholder = main_symbols.lookup(name)) {
      placeholders.insert(placeholder);
    }

    llvm::SetVector<mlir::Operation*> operand_defs;
    for (mlir::Value operand : composite.getOperands()) {
      if (mlir::Operation* def = operand.getDefiningOp())
        operand_defs.insert(def);
    }
    if (failed(InlineBody(composite, body))) return failure();
    // What only fed the region is now dead: the body uses its own constants.
    EraseTriviallyDead(operand_defs.getArrayRef());
  }

  llvm::StringSet<> referenced = ReferencedSymbols(module);
  for (mlir::Operation* placeholder : placeholders) {
    if (!referenced.contains(cast<func::FuncOp>(placeholder).getSymName())) {
      main_symbols.erase(placeholder);
    }
  }
  return success();
}

}  // namespace mlir::TFL
