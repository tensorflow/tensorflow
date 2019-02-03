//===- OpStats.cpp - Prints stats of operations in module -----------------===//
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

#include "mlir/IR/InstVisitor.h"
#include "mlir/IR/Instruction.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
struct PrintOpStatsPass : public ModulePass, InstWalker<PrintOpStatsPass> {
  explicit PrintOpStatsPass(llvm::raw_ostream &os = llvm::errs())
      : ModulePass(&PrintOpStatsPass::passID), os(os) {}

  // Prints the resultant operation statistics post iterating over the module.
  PassResult runOnModule(Module *m) override;

  // Updates the operation statistics for the given instruction.
  void visitInstruction(Instruction *inst);

  // Print summary of op stats.
  void printSummary();

  static char passID;

private:
  llvm::StringMap<int64_t> opCount;

  llvm::raw_ostream &os;
};
} // namespace

char PrintOpStatsPass::passID = 0;

PassResult PrintOpStatsPass::runOnModule(Module *m) {
  for (auto &fn : *m)
    walk(&fn);
  printSummary();
  return success();
}

void PrintOpStatsPass::visitInstruction(Instruction *inst) {
  ++opCount[inst->getName().getStringRef()];
}

void PrintOpStatsPass::printSummary() {
  os << "Operations encountered:\n";
  os << "-----------------------\n";
  std::vector<StringRef> sorted(opCount.keys().begin(), opCount.keys().end());
  llvm::sort(sorted);

  // Split an operation name from its dialect prefix.
  auto splitOperationName = [](StringRef opName) {
    auto splitName = opName.split('.');
    return splitName.second.empty() ? std::make_pair("", splitName.first)
                                    : splitName;
  };

  // Compute the largest dialect and operation name.
  StringRef dialectName, opName;
  size_t maxLenOpName = 0, maxLenDialect = 0;
  for (const auto &key : sorted) {
    std::tie(dialectName, opName) = splitOperationName(key);
    maxLenDialect = std::max(maxLenDialect, dialectName.size());
    maxLenOpName = std::max(maxLenOpName, opName.size());
  }

  for (const auto &key : sorted) {
    std::tie(dialectName, opName) = splitOperationName(key);

    // Left-align the names (aligning on the dialect) and right-align the count
    // below. The alignment is for readability and does not affect CSV/FileCheck
    // parsing.
    if (dialectName.empty())
      os.indent(maxLenDialect + 3);
    else
      os << llvm::right_justify(dialectName, maxLenDialect + 2) << '.';

    // Left justify the operation name.
    os << llvm::left_justify(opName, maxLenOpName) << " , " << opCount[key]
       << '\n';
  }
}

static PassRegistration<PrintOpStatsPass>
    pass("print-op-stats", "Print statistics of operations");
