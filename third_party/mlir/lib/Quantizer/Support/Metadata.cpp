//===- Metadata.cpp - Top level types and metadata ------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/Metadata.h"

#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::quantizer;

void CAGUniformMetadata::printSummary(raw_ostream &os) const {
  if (requiredRange.hasValue()) {
    os << "\n[" << requiredRange.getValue().first << ","
       << requiredRange.getValue().second << "]";
  }

  if (disabledCandidateTypes.any()) {
    os << "\n![";
    mlir::interleaveComma(disabledCandidateTypes.set_bits(), os);
    os << "]";
  }

  if (selectedType) {
    os << "\n" << selectedType;
  }
}
