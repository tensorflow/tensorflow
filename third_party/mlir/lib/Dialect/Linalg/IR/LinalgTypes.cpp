//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
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
