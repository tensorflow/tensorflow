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

#include "mlir/IR/Function.h"
#include "mlir/IR/BasicBlock.h"
#include <vector>

namespace mlir {

// This kind of function is defined in terms of a "Control Flow Graph" of basic
// blocks, each of which includes instructions.
class CFGFunction : public Function {
public:
  CFGFunction(StringRef name, FunctionType *type);

  // FIXME: wrong representation and API, leaks memory etc.
  std::vector<BasicBlock*> blockList;

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Function *func) {
    return func->getKind() == Kind::CFGFunc;
  }

  void print(raw_ostream &os) const;
};


} // end namespace mlir

#endif  // MLIR_IR_CFGFUNCTION_H
