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

#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

mlir::linalg::LinalgDialect::LinalgDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<RangeType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgLibraryOps.cpp.inc"
      >();
}
Type mlir::linalg::LinalgDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'range' types.
  if (keyword == "range")
    return RangeType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown Linalg type: " + keyword);
  return Type();
}

/// RangeType prints as just "range".
static void print(RangeType rt, DialectAsmPrinter &os) { os << "range"; }

void mlir::linalg::LinalgDialect::printType(Type type,
                                            DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled Linalg type");
  case LinalgTypes::Range:
    print(type.cast<RangeType>(), os);
    break;
  }
}
