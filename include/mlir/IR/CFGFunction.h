//===- CFGFunction.h - MLIR CFGFunction Class -------------------*- C++ -*-===//
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

#ifndef MLIR_IR_CFGFUNCTION_H
#define MLIR_IR_CFGFUNCTION_H

#include "mlir/IR/BasicBlock.h"
#include "mlir/IR/Function.h"

namespace mlir {

// This kind of function is defined in terms of a "Control Flow Graph" of basic
// blocks, each of which includes instructions.
class CFGFunction : public Function {
public:
  CFGFunction(StringRef name, FunctionType *type);
  ~CFGFunction();

  //===--------------------------------------------------------------------===//
  // BasicBlock list management
  //===--------------------------------------------------------------------===//

  /// This is the list of blocks in the function.
  typedef llvm::iplist<BasicBlock> BasicBlockListType;
  BasicBlockListType &getBlocks() { return blocks; }
  const BasicBlockListType &getBlocks() const { return blocks; }

  // Iteration over the block in the function.
  using iterator = BasicBlockListType::iterator;
  using const_iterator = BasicBlockListType::const_iterator;
  using reverse_iterator = BasicBlockListType::reverse_iterator;
  using const_reverse_iterator = BasicBlockListType::const_reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  const_iterator begin() const { return blocks.begin(); }
  const_iterator end() const { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }
  const_reverse_iterator rbegin() const { return blocks.rbegin(); }
  const_reverse_iterator rend() const { return blocks.rend(); }

  bool empty() const { return blocks.empty(); }
  void push_back(BasicBlock *block) { blocks.push_back(block); }
  void push_front(BasicBlock *block) { blocks.push_front(block); }

  BasicBlock &back() { return blocks.back(); }
  const BasicBlock &back() const {
    return const_cast<CFGFunction *>(this)->back();
  }

  BasicBlock &front() { return blocks.front(); }
  const BasicBlock &front() const {
    return const_cast<CFGFunction*>(this)->front();
  }

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  /// getSublistAccess() - Returns pointer to member of block list
  static BasicBlockListType CFGFunction::*getSublistAccess(BasicBlock*) {
    return &CFGFunction::blocks;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Function *func) {
    return func->getKind() == Kind::CFGFunc;
  }

  /// Displays the CFG in a window. This is for use from the debugger and
  /// depends on Graphviz to generate the graph.
  /// This function is defined in CFGFunctionViewGraph and only works with that
  /// target linked.
  void viewGraph() const;

private:
  BasicBlockListType blocks;
};

} // end namespace mlir

#endif  // MLIR_IR_CFGFUNCTION_H
