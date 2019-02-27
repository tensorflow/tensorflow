//===- PassRegistry.h - Pass Registration Utilities -------------*- C++ -*-===//
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
// This file contains utilities for registering information about compiler
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSREGISTRY_H_
#define MLIR_PASS_PASSREGISTRY_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <functional>

namespace mlir {
class Pass;

using PassAllocatorFunction = std::function<Pass *()>;

/// A special type used by transformation passes to provide an address that can
/// act as a unique identifier during pass registration.
struct alignas(8) PassID {
  template <typename PassT> static PassID *getID() {
    static PassID id;
    return &id;
  }
};

/// Structure to group information about a pass (argument to invoke via
/// mlir-opt, description, pass allocator and unique ID).
class PassInfo {
public:
  /// PassInfo constructor should not be invoked directly, instead use
  /// PassRegistration or registerPass.
  PassInfo(StringRef arg, StringRef description, const PassID *passID,
           PassAllocatorFunction allocator)
      : arg(arg), description(description), allocator(allocator),
        passID(passID) {}

  /// Returns an allocated instance of this pass.
  Pass *createPass() const {
    assert(allocator &&
           "Cannot call createPass on PassInfo without default allocator");
    return allocator();
  }

  /// Returns the command line option that may be passed to 'mlir-opt' that will
  /// cause this pass to run or null if there is no such argument.
  StringRef getPassArgument() const { return arg; }

  /// Returns a description for the pass, this never returns null.
  StringRef getPassDescription() const { return description; }

private:
  // The argument with which to invoke the pass via mlir-opt.
  StringRef arg;

  // Description of the pass.
  StringRef description;

  // Allocator to construct an instance of this pass.
  PassAllocatorFunction allocator;

  // Unique identifier for pass.
  const PassID *passID;
};

/// Register a specific dialect creation function with the system, typically
/// used through the PassRegistration template.
void registerPass(StringRef arg, StringRef description, const PassID *passID,
                  const PassAllocatorFunction &function);

/// PassRegistration provides a global initializer that registers a Pass
/// allocation routine.
///
/// Usage:
///
///   // At namespace scope.
///   static PassRegistration<MyPass> Unused("unused", "Unused pass");
template <typename ConcretePass> struct PassRegistration {
  PassRegistration(StringRef arg, StringRef description) {
    registerPass(arg, description, &ConcretePass::passID,
                 [&]() { return new ConcretePass(); });
  }
};

/// Adds command line option for each registered pass.
struct PassNameParser : public llvm::cl::parser<const PassInfo *> {
  PassNameParser(llvm::cl::Option &opt);

  void printOptionInfo(const llvm::cl::Option &O,
                       size_t GlobalWidth) const override;
};
} // end namespace mlir

#endif // MLIR_PASS_PASSREGISTRY_H_
