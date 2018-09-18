//===- mlir/Pass.h - Base classes for compiler passes -----------*- C++ -*-===//
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

#ifndef MLIR_PASS_H
#define MLIR_PASS_H

#include "llvm/Support/Compiler.h"

namespace mlir {
class CFGFunction;
class MLFunction;
class Module;

// Values that can be used by to signal success/failure. This can be implicitly
// converted to/from boolean values, with false representing success and true
// failure.
struct LLVM_NODISCARD PassResult {
  enum ResultEnum { Success, Failure } value;
  PassResult(ResultEnum v) : value(v) {}
  operator bool() const { return value == Failure; }
};

class Pass {
public:
  virtual ~Pass() = default;
  virtual PassResult runOnModule(Module *m) = 0;

  static PassResult success() { return PassResult::Success; }
  static PassResult failure() { return PassResult::Failure; }
};

class ModulePass : public Pass {
public:
  virtual PassResult runOnModule(Module *m) override = 0;
};

class FunctionPass : public Pass {
public:
  virtual PassResult runOnCFGFunction(CFGFunction *f) = 0;
  virtual PassResult runOnMLFunction(MLFunction *f) = 0;

  // Iterates over all functions in a module, halting upon failure.
  virtual PassResult runOnModule(Module *m) override;
};

class CFGFunctionPass : public FunctionPass {
public:
  virtual PassResult runOnMLFunction(MLFunction *f) override {
    // Skip over MLFunction.
    return success();
  }
  virtual PassResult runOnCFGFunction(CFGFunction *f) override = 0;
};

class MLFunctionPass : public FunctionPass {
public:
  virtual PassResult runOnCFGFunction(CFGFunction *f) override {
    // Skip over CFGFunction.
    return success();
  }
  virtual PassResult runOnMLFunction(MLFunction *f) override = 0;
};

} // end namespace mlir

#endif // MLIR_PASS_H
