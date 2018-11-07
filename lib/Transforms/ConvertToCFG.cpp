//===- ConvertToCFG.cpp - ML function to CFG function converstion ---------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements APIs to convert ML functions into CFG functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/CFGFunction.h"
#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
using namespace mlir;

//===----------------------------------------------------------------------===//
// ML function converter
//===----------------------------------------------------------------------===//

namespace {
// Generates CFG function equivalent to the given ML function.
class FunctionConverter {
public:
  FunctionConverter(CFGFunction *cfgFunc)
      : cfgFunc(cfgFunc), builder(cfgFunc) {}
  CFGFunction *convert(const MLFunction *mlFunc);

private:
  CFGFunction *cfgFunc;
  CFGFuncBuilder builder;
};
} // end anonymous namespace

CFGFunction *FunctionConverter::convert(const MLFunction *mlFunc) {
  builder.createBlock();

  // Creates return instruction with no operands.
  // TODO: convert return operands.
  builder.createReturn(mlFunc->getReturnStmt()->getLoc(), {});

  // TODO: convert ML function body.

  return cfgFunc;
}

//===----------------------------------------------------------------------===//
// Module converter
//===----------------------------------------------------------------------===//

namespace {
// ModuleConverter class does CFG conversion for the whole module.
class ModuleConverter : public ModulePass {
public:
  explicit ModuleConverter() {}

  PassResult runOnModule(Module *m) override;

  static char passID;

private:
  // Generates CFG functions for all ML functions in the module.
  void convertMLFunctions();
  // Generates CFG function for the given ML function.
  CFGFunction *convert(const MLFunction *mlFunc);
  // Replaces all ML function references in the module
  // with references to the generated CFG functions.
  void replaceReferences();
  // Replaces function references in the given function.
  void replaceReferences(CFGFunction *cfgFunc);
  void replaceReferences(MLFunction *mlFunc);
  // Removes all ML funtions from the module.
  void removeMLFunctions();

  // Map from ML functions to generated CFG functions.
  llvm::DenseMap<const MLFunction *, CFGFunction *> generatedFuncs;
  Module *module = nullptr;
};
} // end anonymous namespace

char ModuleConverter::passID = 0;

// Iterates over all functions in the module generating CFG functions
// equivalent to ML functions and replacing references to ML functions
// with references to the generated ML functions.
PassResult ModuleConverter::runOnModule(Module *m) {
  module = m;
  convertMLFunctions();
  replaceReferences();
  return success();
}

void ModuleConverter::convertMLFunctions() {
  for (Function &fn : *module) {
    if (auto *mlFunc = dyn_cast<MLFunction>(&fn))
      generatedFuncs[mlFunc] = convert(mlFunc);
  }
}

// Creates CFG function equivalent to the given ML function.
CFGFunction *ModuleConverter::convert(const MLFunction *mlFunc) {
  // TODO: ensure that CFG function name is unique.
  auto *cfgFunc =
      new CFGFunction(mlFunc->getLoc(), mlFunc->getName().str() + "_cfg",
                      mlFunc->getType(), mlFunc->getAttrs());
  module->getFunctions().push_back(cfgFunc);

  // Generates the body of the CFG function.
  return FunctionConverter(cfgFunc).convert(mlFunc);
}

void ModuleConverter::replaceReferences() {
  for (Function &fn : *module) {
    switch (fn.getKind()) {
    case Function::Kind::CFGFunc:
      replaceReferences(&cast<CFGFunction>(fn));
      break;
    case Function::Kind::MLFunc:
      replaceReferences(&cast<MLFunction>(fn));
      break;
    case Function::Kind::ExtFunc:
      // nothing to do for external functions
      break;
    }
  }
}

void ModuleConverter::replaceReferences(CFGFunction *func) {
  // TODO: NOP for now since function attributes are not yet implemented.
}

void ModuleConverter::replaceReferences(MLFunction *func) {
  // TODO: NOP for now since function attributes are not yet implemented.
}

// Removes all ML functions from the module.
void ModuleConverter::removeMLFunctions() {
  // Delete ML functions from the module.
  for (auto it = module->begin(), e = module->end(); it != e;) {
    // Manipulate iterator carefully to avoid deleting a function we're pointing
    // at.
    Function &fn = *it++;
    if (auto mlFunc = dyn_cast<MLFunction>(&fn))
      mlFunc->eraseFromModule();
  }
}

//===----------------------------------------------------------------------===//
// Entry point method
//===----------------------------------------------------------------------===//

/// Replaces all ML functions in the module with equivalent CFG functions.
/// Function references are appropriately patched to refer to the newly
/// generated CFG functions.
ModulePass *mlir::createConvertToCFGPass() { return new ModuleConverter(); }

static PassRegistration<ModuleConverter>
    pass("convert-to-cfg",
         "Convert all ML functions in the module to CFG ones");
