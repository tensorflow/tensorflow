//===- ViewRegionGraph.cpp - View/write graphviz graphs -------------------===//
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

void mlir::viewGraph(Region &region, const llvm::Twine &name, bool shortNames,
                     const llvm::Twine &title,
                     llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&region, name, shortNames, title, program);
}

llvm::raw_ostream &mlir::writeGraph(llvm::raw_ostream &os, Region &region,
                                    bool shortNames, const llvm::Twine &title) {
  return llvm::WriteGraph(os, &region, shortNames, title);
}

void mlir::Region::viewGraph(const llvm::Twine &regionName) {
  ::mlir::viewGraph(*this, regionName);
}
void mlir::Region::viewGraph() { viewGraph("region"); }

namespace {
struct PrintCFGPass : public FunctionPass<PrintCFGPass> {
  PrintCFGPass(llvm::raw_ostream &os = llvm::errs(), bool shortNames = false,
               const llvm::Twine &title = "")
      : os(os), shortNames(shortNames), title(title.str()) {}
  void runOnFunction() override {
    mlir::writeGraph(os, getFunction().getBody(), shortNames, title);
  }

private:
  llvm::raw_ostream &os;
  bool shortNames;
  std::string title;
};
} // namespace

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>>
mlir::createPrintCFGGraphPass(llvm::raw_ostream &os, bool shortNames,
                              const llvm::Twine &title) {
  return std::make_unique<PrintCFGPass>(os, shortNames, title);
}

static PassRegistration<PrintCFGPass> pass("print-cfg-graph",
                                           "Print CFG graph per Function");
