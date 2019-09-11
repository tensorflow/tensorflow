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
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <functional>
#include <memory>

namespace mlir {
class Pass;
class PassManager;

/// A registry function that adds passes to the given pass manager.
using PassRegistryFunction = std::function<void(PassManager &)>;

using PassAllocatorFunction = std::function<std::unique_ptr<Pass>()>;

/// A special type used by transformation passes to provide an address that can
/// act as a unique identifier during pass registration.
using PassID = ClassID;

/// Structure to group information about a passes and pass pipelines (argument
/// to invoke via mlir-opt, description, pass pipeline builder).
class PassRegistryEntry {
public:
  /// Adds this pass registry entry to the given pass manager.
  void addToPipeline(PassManager &pm) const {
    assert(builder &&
           "Cannot call addToPipeline on PassRegistryEntry without builder");
    builder(pm);
  }

  /// Returns the command line option that may be passed to 'mlir-opt' that will
  /// cause this pass to run or null if there is no such argument.
  StringRef getPassArgument() const { return arg; }

  /// Returns a description for the pass, this never returns null.
  StringRef getPassDescription() const { return description; }

protected:
  PassRegistryEntry(StringRef arg, StringRef description,
                    PassRegistryFunction builder)
      : arg(arg), description(description), builder(builder) {}

private:
  // The argument with which to invoke the pass via mlir-opt.
  StringRef arg;

  // Description of the pass.
  StringRef description;

  // Function to register this entry to a pass manager pipeline.
  PassRegistryFunction builder;
};

/// A structure to represent the information of a registered pass pipeline.
class PassPipelineInfo : public PassRegistryEntry {
public:
  PassPipelineInfo(StringRef arg, StringRef description,
                   PassRegistryFunction builder)
      : PassRegistryEntry(arg, description, builder) {}
};

/// A structure to represent the information for a derived pass class.
class PassInfo : public PassRegistryEntry {
public:
  /// PassInfo constructor should not be invoked directly, instead use
  /// PassRegistration or registerPass.
  PassInfo(StringRef arg, StringRef description, const PassID *passID,
           PassAllocatorFunction allocator);
};

/// Register a specific dialect pipeline registry function with the system,
/// typically used through the PassPipelineRegistration template.
void registerPassPipeline(StringRef arg, StringRef description,
                          const PassRegistryFunction &function);

/// Register a specific dialect pass allocator function with the system,
/// typically used through the PassRegistration template.
void registerPass(StringRef arg, StringRef description, const PassID *passID,
                  const PassAllocatorFunction &function);

/// PassRegistration provides a global initializer that registers a Pass
/// allocation routine for a concrete pass instance.  The third argument is
/// optional and provides a callback to construct a pass that does not have
/// a default constructor.
///
/// Usage:
///
///   // At namespace scope.
///   static PassRegistration<MyPass> Unused("unused", "Unused pass");
template <typename ConcretePass> struct PassRegistration {
  PassRegistration(StringRef arg, StringRef description,
                   const PassAllocatorFunction &constructor) {
    registerPass(arg, description, PassID::getID<ConcretePass>(), constructor);
  }

  PassRegistration(StringRef arg, StringRef description) {
    PassAllocatorFunction constructor = [] {
      return std::make_unique<ConcretePass>();
    };
    registerPass(arg, description, PassID::getID<ConcretePass>(), constructor);
  }
};

/// PassPipelineRegistration provides a global initializer that registers a Pass
/// pipeline builder routine.
///
/// Usage:
///
///   // At namespace scope.
///   void pipelineBuilder(PassManager &pm) {
///      pm.addPass(new MyPass());
///      pm.addPass(new MyOtherPass());
///   }
///
///   static PassPipelineRegistration Unused("unused", "Unused pass",
///                                          pipelineBuilder);
struct PassPipelineRegistration {
  PassPipelineRegistration(StringRef arg, StringRef description,
                           PassRegistryFunction builder) {
    registerPassPipeline(arg, description, builder);
  }

  /// Constructor that accepts a pass allocator function instead of the standard
  /// registry function. This is useful for registering specializations of
  /// existing passes.
  PassPipelineRegistration(StringRef arg, StringRef description,
                           PassAllocatorFunction allocator);
};

/// Adds command line option for each registered pass.
struct PassNameParser : public llvm::cl::parser<const PassRegistryEntry *> {
  PassNameParser(llvm::cl::Option &opt);

  void initialize();

  void printOptionInfo(const llvm::cl::Option &O,
                       size_t GlobalWidth) const override;
};
} // end namespace mlir

#endif // MLIR_PASS_PASSREGISTRY_H_
