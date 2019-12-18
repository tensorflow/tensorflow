//===- Metadata.cpp - Top level types and metadata ------------------------===//
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
