//===- ViewFunctionGraph.cpp - View/write graphviz graphs -----------------===//
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

#include "mlir/Transforms/ViewFunctionGraph.h"
#include "mlir/IR/FunctionGraphTraits.h"
#include "mlir/Pass.h"

using namespace mlir;

namespace llvm {

// Specialize DOTGraphTraits to produce more readable output.
template <>
struct llvm::DOTGraphTraits<const Function *> : public DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(const BasicBlock *basicBlock,
                                  const Function *);
};

std::string llvm::DOTGraphTraits<const Function *>::getNodeLabel(
    const BasicBlock *basicBlock, const Function *) {
  // Reuse the print output for the node labels.
  std::string outStreamStr;
  raw_string_ostream os(outStreamStr);
  basicBlock->print(os);
  std::string &outStr = os.str();

  if (outStr[0] == '\n')
    outStr.erase(outStr.begin());

  // Process string output to left justify the block.
  for (unsigned i = 0; i != outStr.length(); ++i) {
    if (outStr[i] == '\n') {
      outStr[i] = '\\';
      outStr.insert(outStr.begin() + i + 1, 'l');
    }
  }

  return outStr;
}

} // end namespace llvm

void mlir::viewGraph(const Function &function, const llvm::Twine &name,
                     bool shortNames, const llvm::Twine &title,
                     llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&function, name, shortNames, title, program);
}

llvm::raw_ostream &mlir::writeGraph(llvm::raw_ostream &os,
                                    const Function *function, bool shortNames,
                                    const llvm::Twine &title) {
  return llvm::WriteGraph(os, function, shortNames, title);
}

void mlir::Function::viewGraph() const {
  ::mlir::viewGraph(*this, llvm::Twine("cfgfunc ") + getName().str());
}

namespace {
struct PrintCFGPass : public FunctionPass {
  PrintCFGPass(llvm::raw_ostream &os = llvm::errs(), bool shortNames = false,
               const llvm::Twine &title = "")
      : FunctionPass(&PrintCFGPass::passID), os(os), shortNames(shortNames),
        title(title) {}
  PassResult runOnCFGFunction(Function *function) override {
    mlir::writeGraph(os, function, shortNames, title);
    return success();
  }

  static char passID;

private:
  llvm::raw_ostream &os;
  bool shortNames;
  const llvm::Twine &title;
};
} // namespace

char PrintCFGPass::passID = 0;

FunctionPass *mlir::createPrintCFGGraphPass(llvm::raw_ostream &os,
                                            bool shortNames,
                                            const llvm::Twine &title) {
  return new PrintCFGPass(os, shortNames, title);
}

static PassRegistration<PrintCFGPass> pass("print-cfg-graph",
                                           "Print CFG graph per function");
