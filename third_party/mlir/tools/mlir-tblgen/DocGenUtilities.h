//===- DocGenUtilities.h - MLIR doc gen utilities ---------------*- C++ -*-===//
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
// This file defines common utilities for generating documents from tablgen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_

namespace llvm {
class raw_ostream;
class StringRef;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Emit the description by aligning the text to the left per line (e.g.
// removing the minimum indentation across the block).
//
// This expects that the description in the tablegen file is already formatted
// in a way the user wanted but has some additional indenting due to being
// nested.
void emitDescription(llvm::StringRef description, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_DOCGENUTILITIES_H_
