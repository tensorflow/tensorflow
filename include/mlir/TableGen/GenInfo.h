//===- GenInfo.h - Generator info -------------------------------*- C++ -*-===//
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

#ifndef MLIR_TABLEGEN_GENINFO_H_
#define MLIR_TABLEGEN_GENINFO_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <functional>

namespace llvm {
class RecordKeeper;
} // end namespace llvm

namespace mlir {

/// Generator function to invoke.
using GenFunction = std::function<bool(const llvm::RecordKeeper &recordKeeper,
                                       raw_ostream &os)>;

/// Structure to group information about a generator (argument to invoke via
/// mlir-tblgen, description, and generator function).
class GenInfo {
public:
  /// GenInfo constructor should not be invoked directly, instead use
  /// GenRegistration or registerGen.
  GenInfo(StringRef arg, StringRef description, GenFunction generator)
      : arg(arg), description(description), generator(generator){};

  /// Invokes the generator and returns whether the generator failed.
  bool invoke(const llvm::RecordKeeper &recordKeeper, raw_ostream &os) const {
    assert(generator && "Cannot call generator with null generator");
    return generator(recordKeeper, os);
  }

  /// Returns the command line option that may be passed to 'mlir-tblgen' to
  /// invoke this generator.
  StringRef getGenArgument() const { return arg; }

  /// Returns a description for the generator.
  StringRef getGenDescription() const { return description; }

private:
  // The argument with which to invoke the generator via mlir-tblgen.
  StringRef arg;

  // Description of the generator.
  StringRef description;

  // Generator function.
  GenFunction generator;
};

/// GenRegistration provides a global initializer that registers a generator
/// function.
///
/// Usage:
///
///   // At namespace scope.
///   static GenRegistration Print("print", "Print records", [](...){...});
struct GenRegistration {
  GenRegistration(StringRef arg, StringRef description, GenFunction function);
};

} // end namespace mlir

#endif // MLIR_TABLEGEN_GENINFO_H_
