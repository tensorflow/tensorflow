//===- ViewRegionGraph.cpp - View/write graphviz graphs -------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/ViewRegionGraph.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace llvm {

// Specialize DOTGraphTraits to produce more readable output.
template <> struct DOTGraphTraits<Region *> : public DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(Block *Block, Region *);
};

std::string DOTGraphTraits<Region *>::getNodeLabel(Block *Block, Region *) {
  // Reuse the print output for the node labels.
  std::string outStreamStr;
  raw_string_ostream os(outStreamStr);
  Block->print(os);
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

void mlir::viewGraph(Region &region, const Twine &name, bool shortNames,
                     const Twine &title, llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&region, name, shortNames, title, program);
}

raw_ostream &mlir::writeGraph(raw_ostream &os, Region &region, bool shortNames,
                              const Twine &title) {
  return llvm::WriteGraph(os, &region, shortNames, title);
}

void mlir::Region::viewGraph(const Twine &regionName) {
  ::mlir::viewGraph(*this, regionName);
}
void mlir::Region::viewGraph() { viewGraph("region"); }

namespace {
struct PrintCFGPass : public FunctionPass<PrintCFGPass> {
  PrintCFGPass(raw_ostream &os = llvm::errs(), bool shortNames = false,
               const Twine &title = "")
      : os(os), shortNames(shortNames), title(title.str()) {}
  void runOnFunction() override {
    mlir::writeGraph(os, getFunction().getBody(), shortNames, title);
  }

private:
  raw_ostream &os;
  bool shortNames;
  std::string title;
};
} // namespace

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>>
mlir::createPrintCFGGraphPass(raw_ostream &os, bool shortNames,
                              const Twine &title) {
  return std::make_unique<PrintCFGPass>(os, shortNames, title);
}

static PassRegistration<PrintCFGPass> pass("print-cfg-graph",
                                           "Print CFG graph per Function");
