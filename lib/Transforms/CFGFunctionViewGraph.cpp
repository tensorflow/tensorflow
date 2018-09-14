//===- CFGFunctionViewGraph.h - View/write graphviz graphs ------*- C++ -*-===//
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

#include "mlir/Transforms/CFGFunctionViewGraph.h"

using namespace mlir;

namespace llvm {

// Specialize GraphTraits to treat a CFGFunction as a graph of basic blocks.
template <>
struct llvm::GraphTraits<const CFGFunction *> {
  using NodeRef = const BasicBlock *;
  using ChildIteratorType = BasicBlock::const_succ_iterator;
  using nodes_iterator = pointer_iterator<CFGFunction::const_iterator>;

  static NodeRef getEntryNode(const CFGFunction *function) {
    return &function->front();
  }

  static nodes_iterator nodes_begin(const CFGFunction *F) {
    return nodes_iterator(F->begin());
  }
  static nodes_iterator nodes_end(const CFGFunction *F) {
    return nodes_iterator(F->end());
  }

  static ChildIteratorType child_begin(NodeRef basicBlock) {
    return basicBlock->succ_begin();
  }
  static ChildIteratorType child_end(NodeRef basicBlock) {
    return basicBlock->succ_end();
  }
};

// Specialize DOTGraphTraits to produce more readable output.
template <>
struct llvm::DOTGraphTraits<const CFGFunction *>
    : public DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(const BasicBlock *basicBlock,
                                  const CFGFunction *);
};

std::string llvm::DOTGraphTraits<const CFGFunction *>::getNodeLabel(
    const BasicBlock *basicBlock, const CFGFunction *) {
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

void mlir::viewGraph(const CFGFunction &function, const llvm::Twine &name,
                     bool shortNames, const llvm::Twine &title,
                     llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&function, name, shortNames, title, program);
}

llvm::raw_ostream &mlir::writeGraph(llvm::raw_ostream &os,
                                    const CFGFunction *function,
                                    bool shortNames, const llvm::Twine &title) {
  return llvm::WriteGraph(os, function, shortNames, title);
}

void mlir::CFGFunction::viewGraph() const {
  ::mlir::viewGraph(*this, llvm::Twine("cfgfunc ") + getName().str());
}

namespace {
struct PrintCFGPass : public CFGFunctionPass {
  PrintCFGPass(llvm::raw_ostream &os, bool shortNames, const llvm::Twine &title)
      : os(os), shortNames(shortNames), title(title) {}
  PassResult runOnCFGFunction(CFGFunction *function) override {
    mlir::writeGraph(os, function, shortNames, title);
    return success();
  }

private:
  llvm::raw_ostream &os;
  bool shortNames;
  const llvm::Twine &title;
};
} // namespace

CFGFunctionPass *mlir::createPrintCFGGraphPass(llvm::raw_ostream &os,
                                               bool shortNames,
                                               const llvm::Twine &title) {
  return new PrintCFGPass(os, shortNames, title);
}
