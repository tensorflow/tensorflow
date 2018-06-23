//===- Function.h - MLIR Function Class -------------------------*- C++ -*-===//
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
// Functions are the basic unit of composition in MLIR.  There are three
// different kinds of functions: external functions, CFG functions, and ML
// functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTION_H
#define MLIR_IR_FUNCTION_H

#include "mlir/Support/LLVM.h"

namespace mlir {
  class FunctionType;

  /// This is the base class for all of the MLIR function types
  class Function {
    std::string name;
    FunctionType *const type;
  public:
    explicit Function(StringRef name, FunctionType *type);

    void print(raw_ostream &os);
    void dump();
  };
} // end namespace mlir

#endif  // MLIR_IR_FUNCTION_H
