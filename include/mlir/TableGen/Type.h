//===- Type.h - Type class --------------------------------------*- C++ -*-===//
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
// Type wrapper to simplify using TableGen Record defining a MLIR Type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_TYPE_H_
#define MLIR_TABLEGEN_TYPE_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Wrapper class providing helper methods for accessing MLIR Type defined
// in TableGen. This class should closely reflect what is defined as
// class Type in TableGen.
class Type {
public:
  explicit Type(const llvm::Record &def);
  explicit Type(const llvm::Record *def) : Type(*def) {}
  explicit Type(const llvm::DefInit *init);

  // Returns the TableGen def name for this type.
  StringRef getTableGenDefName() const;

  // Returns the method call to invoke upon a MLIR pattern rewriter to
  // construct this type. Returns an empty StringRef if the method call
  // is undefined or unset.
  StringRef getBuilderCall() const;

  // Returns this type's predicate CNF, which is used for checking the
  // validity of this type.
  PredCNF getPredicate() const;

private:
  // The TableGen definition of this type.
  const llvm::Record &def;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_TYPE_H_
