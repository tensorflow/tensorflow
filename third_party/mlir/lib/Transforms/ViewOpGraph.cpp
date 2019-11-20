//===- ViewOpGraph.cpp - View/write op graphviz graphs --------------------===//
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

#include "mlir/Transforms/ViewOpGraph.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/Support/CommandLine.h"

static llvm::cl::opt<int> elideIfLarger(
    "print-op-graph-elide-if-larger",
    llvm::cl::desc("Upper limit to emit elements attribute rather than elide"),
    llvm::cl::init(16));

namespace llvm {

// Specialize GraphTraits to treat Block as a graph of Operations as nodes and
// uses as edges.
template <> struct GraphTraits<mlir::Block *> {
  using GraphType = mlir::Block *;
  using NodeRef = mlir::Operation *;

  using ChildIteratorType = mlir::UseIterator;
  static ChildIteratorType child_begin(NodeRef n) {
    return ChildIteratorType(n);
  }
  static ChildIteratorType child_end(NodeRef n) {
    return ChildIteratorType(n, /*end=*/true);
  }

  // Operation's destructor is private so use Operation* instead and use
  // mapped iterator.
  static mlir::Operation *AddressOf(mlir::Operation &op) { return &op; }
  using nodes_iterator =
      mapped_iterator<mlir::Block::iterator, decltype(&AddressOf)>;
  static nodes_iterator nodes_begin(mlir::Block *b) {
    return nodes_iterator(b->begin(), &AddressOf);
  }
  static nodes_iterator nodes_end(mlir::Block *b) {
    return nodes_iterator(b->end(), &AddressOf);
  }
};

// Specialize DOTGraphTraits to produce more readable output.
template <>
struct DOTGraphTraits<mlir::Block *> : public DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;
  static std::string getNodeLabel(mlir::Operation *op, mlir::Block *);
};

std::string DOTGraphTraits<mlir::Block *>::getNodeLabel(mlir::Operation *op,
                                                        mlir::Block *b) {
  // Reuse the print output for the node labels.
  std::string ostr;
  raw_string_ostream os(ostr);
  os << op->getName() << "\n";

  if (!op->getLoc().isa<mlir::UnknownLoc>()) {
    os << op->getLoc() << "\n";
  }

  // Print resultant types
  mlir::interleaveComma(op->getResultTypes(), os);
  os << "\n";

  for (auto attr : op->getAttrs()) {
    os << '\n' << attr.first << ": ";
    // Always emit splat attributes.
    if (attr.second.isa<mlir::SplatElementsAttr>()) {
      attr.second.print(os);
      continue;
    }

    // Elide "big" elements attributes.
    auto elements = attr.second.dyn_cast<mlir::ElementsAttr>();
    if (elements && elements.getNumElements() > elideIfLarger) {
      os << std::string(elements.getType().getRank(), '[') << "..."
         << std::string(elements.getType().getRank(), ']') << " : "
         << elements.getType();
      continue;
    }

    auto array = attr.second.dyn_cast<mlir::ArrayAttr>();
    if (array && static_cast<int64_t>(array.size()) > elideIfLarger) {
      os << "[...]";
      continue;
    }

    // Print all other attributes.
    attr.second.print(os);
  }
  return os.str();
}

} // end namespace llvm

namespace {
// PrintOpPass is simple pass to write graph per function.
// Note: this is a module pass only to avoid interleaving on the same ostream
// due to multi-threading over functions.
struct PrintOpPass : public mlir::ModulePass<PrintOpPass> {
  explicit PrintOpPass(llvm::raw_ostream &os = llvm::errs(),
                       bool short_names = false, const llvm::Twine &title = "")
      : os(os), title(title.str()), short_names(short_names) {}

  std::string getOpName(mlir::Operation &op) {
    auto symbolAttr = op.getAttrOfType<mlir::StringAttr>(
        mlir::SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return symbolAttr.getValue();
    ++unnamedOpCtr;
    return (op.getName().getStringRef() + llvm::utostr(unnamedOpCtr)).str();
  }

  // Print all the ops in a module.
  void processModule(mlir::ModuleOp module) {
    for (mlir::Operation &op : module) {
      // Modules may actually be nested, recurse on nesting.
      if (auto nestedModule = llvm::dyn_cast<mlir::ModuleOp>(op)) {
        processModule(nestedModule);
        continue;
      }
      auto opName = getOpName(op);
      for (mlir::Region &region : op.getRegions()) {
        for (auto indexed_block : llvm::enumerate(region)) {
          // Suffix block number if there are more than 1 block.
          auto blockName = region.getBlocks().size() == 1
                               ? ""
                               : ("__" + llvm::utostr(indexed_block.index()));
          llvm::WriteGraph(os, &indexed_block.value(), short_names,
                           llvm::Twine(title) + opName + blockName);
        }
      }
    }
  }

  void runOnModule() override { processModule(getModule()); }

private:
  llvm::raw_ostream &os;
  std::string title;
  int unnamedOpCtr = 0;
  bool short_names;
};
} // namespace

void mlir::viewGraph(mlir::Block &block, const llvm::Twine &name,
                     bool shortNames, const llvm::Twine &title,
                     llvm::GraphProgram::Name program) {
  llvm::ViewGraph(&block, name, shortNames, title, program);
}

llvm::raw_ostream &mlir::writeGraph(llvm::raw_ostream &os, mlir::Block &block,
                                    bool shortNames, const llvm::Twine &title) {
  return llvm::WriteGraph(os, &block, shortNames, title);
}

std::unique_ptr<mlir::OpPassBase<mlir::ModuleOp>>
mlir::createPrintOpGraphPass(llvm::raw_ostream &os, bool shortNames,
                             const llvm::Twine &title) {
  return std::make_unique<PrintOpPass>(os, shortNames, title);
}

static mlir::PassRegistration<PrintOpPass> pass("print-op-graph",
                                                "Print op graph per region");
