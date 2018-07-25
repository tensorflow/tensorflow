//===- mlir/Pass.h - Base class for passes ----------------------*- C++ -*-===//
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
// This file defines a base class that indicates that a specified class is a
// transformation pass implementation.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_PASS_H
#define MLIR_PASS_H

#include "mlir/IR/MLFunction.h"
#include "mlir/IR/Module.h"

namespace mlir {

class Pass {
protected:
  virtual ~Pass() = default;
};

class FunctionPass : public Pass {};

class CFGFunctionPass : public FunctionPass {};

class MLFunctionPass : public FunctionPass {
public:
  virtual bool runOnMLFunction(MLFunction *f) = 0;
  virtual bool runOnModule(Module *m);
};

} // end namespace mlir

#endif // MLIR_PASS_H
