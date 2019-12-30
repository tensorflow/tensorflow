//===- DocGenUtilities.h - MLIR doc gen utilities ---------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating documents from tablegen
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
