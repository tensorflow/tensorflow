//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
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
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Linalg/LinalgTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Linalg/LinalgOps.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

mlir::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect("linalg", context) {
  addTypes<RangeType>();
  addOperations<RangeOp>();
}

Type mlir::LinalgDialect::parseType(StringRef spec, Location loc) const {
  MLIRContext *context = getContext();
  if (spec == "range")
    return RangeType::get(getContext());
  return (context->emitError(loc, "unknown Linalg type: " + spec), Type());
}

/// RangeType prints as just "range".
static void print(RangeType rt, raw_ostream &os) { os << "range"; }

void mlir::LinalgDialect::printType(Type type, raw_ostream &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Linalg type");
  case LinalgTypes::Range:
    print(type.cast<RangeType>(), os);
    break;
  }
}
