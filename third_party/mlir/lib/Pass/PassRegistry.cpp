//===- PassRegistry.cpp - Pass Registration Utilities ---------------------===//
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

#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;

/// Static mapping of all of the registered passes.
static llvm::ManagedStatic<llvm::DenseMap<const PassID *, PassInfo>>
    passRegistry;

/// Static mapping of all of the registered pass pipelines.
static llvm::ManagedStatic<llvm::StringMap<PassPipelineInfo>>
    passPipelineRegistry;

/// Utility to create a default registry function from a pass instance.
static PassRegistryFunction
buildDefaultRegistryFn(PassAllocatorFunction allocator) {
  return [=](PassManager &pm) { pm.addPass(allocator()); };
}

//===----------------------------------------------------------------------===//
// PassPipelineInfo
//===----------------------------------------------------------------------===//

/// Constructor that accepts a pass allocator function instead of the standard
/// registry function. This is useful for registering specializations of
/// existing passes.
PassPipelineRegistration::PassPipelineRegistration(
    StringRef arg, StringRef description, PassAllocatorFunction allocator) {
  registerPassPipeline(arg, description, buildDefaultRegistryFn(allocator));
}

void mlir::registerPassPipeline(StringRef arg, StringRef description,
                                const PassRegistryFunction &function) {
  PassPipelineInfo pipelineInfo(arg, description, function);
  bool inserted = passPipelineRegistry->try_emplace(arg, pipelineInfo).second;
  assert(inserted && "Pass pipeline registered multiple times");
  (void)inserted;
}

//===----------------------------------------------------------------------===//
// PassInfo
//===----------------------------------------------------------------------===//

PassInfo::PassInfo(StringRef arg, StringRef description, const PassID *passID,
                   PassAllocatorFunction allocator)
    : PassRegistryEntry(arg, description, buildDefaultRegistryFn(allocator)) {}

void mlir::registerPass(StringRef arg, StringRef description,
                        const PassID *passID,
                        const PassAllocatorFunction &function) {
  PassInfo passInfo(arg, description, passID, function);
  bool inserted = passRegistry->try_emplace(passID, passInfo).second;
  assert(inserted && "Pass registered multiple times");
  (void)inserted;
}

/// Returns the pass info for the specified pass class or null if unknown.
const PassInfo *mlir::Pass::lookupPassInfo(const PassID *passID) {
  auto it = passRegistry->find(passID);
  if (it == passRegistry->end())
    return nullptr;
  return &it->getSecond();
}

//===----------------------------------------------------------------------===//
// PassNameParser
//===----------------------------------------------------------------------===//

PassNameParser::PassNameParser(llvm::cl::Option &opt)
    : llvm::cl::parser<const PassRegistryEntry *>(opt) {}

void PassNameParser::initialize() {
  llvm::cl::parser<const PassRegistryEntry *>::initialize();

  /// Add the pass entries.
  for (const auto &kv : *passRegistry) {
    addLiteralOption(kv.second.getPassArgument(), &kv.second,
                     kv.second.getPassDescription());
  }
  /// Add the pass pipeline entries.
  for (const auto &kv : *passPipelineRegistry) {
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
  parser<const PassRegistryEntry *>::printOptionInfo(O, GlobalWidth);
}
