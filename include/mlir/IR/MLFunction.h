//===- MLFunction.h - MLIR MLFunction Class ---------------------*- C++ -*-===//
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
// This file defines MLFunction class
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLFUNCTION_H_
#define MLIR_IR_MLFUNCTION_H_

#include "mlir/IR/Function.h"
#include "mlir/IR/StmtBlock.h"

namespace mlir {

// MLFunction is defined as a sequence of statements that may
// include nested affine for loops, conditionals and operations.
class MLFunction : public Function, public StmtBlock {
public:
  MLFunction(StringRef name, FunctionType *type);

  // TODO: add function arguments and return values once
  // SSA values are implemented

  // Methods for support type inquiry through isa, cast, and dyn_cast
  static bool classof(const Function *func) {
    return func->getKind() == Kind::MLFunc;
  }

  void print(raw_ostream &os) const;
};

} // end namespace mlir

#endif  // MLIR_IR_MLFUNCTION_H_
