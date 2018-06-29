//===- MLStatements.h - MLIR ML Statement Classes ------------*- C++ -*-===//
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
// This file defines the classes for MLFunction statements.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLSTATEMENTS_H
#define MLIR_IR_MLSTATEMENTS_H

#include "mlir/Support/LLVM.h"

namespace mlir {
  class MLFunction;

/// ML function consists of ML statements - for statement, if statement
/// or operation.
class MLStatement {
public:
  enum class Kind {
    For,
    If,
    Operation
  };

  Kind getKind() const { return kind; }

  /// Returns the function that this MLStatement is part of.
  MLFunction *getFunction() const { return function; }

  void print(raw_ostream &os) const;
  void dump() const;

protected:
  MLStatement(Kind kind, MLFunction *function)
      : kind(kind), function(function) {}

private:
  Kind kind;
  MLFunction *function;
};

} //end namespace mlir
#endif  // MLIR_IR_STATEMENTS_H
