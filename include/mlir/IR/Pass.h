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

namespace mlir {
class CFGFunction;
class MLFunction;
class Module;

class Pass {
public:
  virtual ~Pass() = default;
};

class ModulePass : public Pass {
public:
  virtual void runOnModule(Module *m) = 0;
};

class FunctionPass : public Pass {
public:
  virtual void runOnCFGFunction(CFGFunction *f) = 0;
  virtual void runOnMLFunction(MLFunction *f) = 0;
  virtual void runOnModule(Module *m);
};

class CFGFunctionPass : public FunctionPass {
public:
  virtual void runOnMLFunction(MLFunction *f) override {}
  virtual void runOnCFGFunction(CFGFunction *f) override = 0;
};

class MLFunctionPass : public FunctionPass {
public:
  virtual void runOnCFGFunction(CFGFunction *f) override {}
  virtual void runOnMLFunction(MLFunction *f) override = 0;
};

} // end namespace mlir

#endif // MLIR_PASS_H
