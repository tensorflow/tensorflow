//===- Pass.h - Base classes for compiler passes ----------------*- C++ -*-===//
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

#ifndef MLIR_PASS_PASS_H
#define MLIR_PASS_PASS_H

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Function;
class Module;

/// A special type used by transformation passes to provide an address that can
/// act as a unique identifier during pass registration.
struct alignas(8) PassID {};

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
  explicit Pass(const PassID *passID) : passID(passID) {}
  virtual ~Pass() = default;
  virtual PassResult runOnModule(Module *m) = 0;

  /// Returns the unique identifier that corresponds to this pass.
  const PassID *getPassID() const { return passID; }

  static PassResult success() { return PassResult::Success; }
  static PassResult failure() { return PassResult::Failure; }

  /// Returns the pass info for the specified pass class or null if unknown.
  static const PassInfo *lookupPassInfo(const PassID *passID);

  /// Returns the pass info for this pass.
  const PassInfo *lookupPassInfo() const { return lookupPassInfo(passID); }

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();

  /// Unique identifier for pass.
  const PassID *const passID;
};

class ModulePass : public Pass {
public:
  explicit ModulePass(const PassID *passID) : Pass(passID) {}

  virtual PassResult runOnModule(Module *m) override = 0;

private:
  /// Out of line virtual method to ensure vtables and metadata are emitted to a
  /// single .o file.
  virtual void anchor();
};

/// FunctionPass's are run on every function in a module, and multiple functions
/// may be optimized concurrently by different instances of the function pass.
/// By subclassing this, your pass promises only to look at the function psased
/// in to it, it isn't allowed to inspect or modify other functions in the
/// module.
class FunctionPass : public Pass {
public:
  explicit FunctionPass(const PassID *passID) : Pass(passID) {}

  /// Implement this function to be run on every function in the module.
  virtual PassResult runOnFunction(Function *fn) = 0;

  // Iterates over all functions in a module, halting upon failure.
  virtual PassResult runOnModule(Module *m) override;
};
} // end namespace mlir

#endif // MLIR_PASS_PASS_H
