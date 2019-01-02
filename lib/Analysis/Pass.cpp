//===- Pass.cpp - Pass infrastructure implementation ----------------------===//
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
// This file implements common pass infrastructure.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/PassNameParser.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"
using namespace mlir;

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void Pass::anchor() {}

/// Out of line virtual method to ensure vtables and metadata are emitted to a
/// single .o file.
void ModulePass::anchor() {}

/// Function passes walk a module and look at each function with their
/// corresponding hooks and terminates upon error encountered.
PassResult FunctionPass::runOnModule(Module *m) {
  for (auto &fn : *m) {
    // All function passes ignore external functions.
    if (fn.empty())
      continue;

    if (runOnFunction(&fn))
      return failure();
  }
  return success();
}

// TODO: The pass registry and pass name parsing should be moved out.
static llvm::ManagedStatic<llvm::DenseMap<const void *, PassInfo>> passRegistry;

void mlir::registerPass(StringRef arg, StringRef description,
                        const void *passID,
                        const PassAllocatorFunction &function) {
  bool inserted = passRegistry
                      ->insert(std::make_pair(
                          passID, PassInfo(arg, description, passID, function)))
                      .second;
  assert(inserted && "Pass registered multiple times");
  (void)inserted;
}

/// Returns the pass info for the specified pass class or null if unknown.
const PassInfo *mlir::Pass::lookupPassInfo(const void *passID) {
  auto it = passRegistry->find(passID);
  if (it == passRegistry->end())
    return nullptr;
  return &it->getSecond();
}

PassNameParser::PassNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const PassInfo *>(opt) {
  for (const auto &kv : *passRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
}

void PassNameParser::printOptionInfo(const llvm::cl::Option &O,
                                     size_t GlobalWidth) const {
  PassNameParser *TP = const_cast<PassNameParser *>(this);
  llvm::array_pod_sort(TP->Values.begin(), TP->Values.end(),
                       [](const PassNameParser::OptionInfo *VT1,
                          const PassNameParser::OptionInfo *VT2) {
                         return VT1->Name.compare(VT2->Name);
                       });
  using llvm::cl::parser;
  parser<const PassInfo *>::printOptionInfo(O, GlobalWidth);
}
