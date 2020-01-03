//===- QuantOps.h - Quantization Ops and Types ------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANTOPS_QUANTOPS_H_
#define MLIR_DIALECT_QUANTOPS_QUANTOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace quant {

/// Defines the 'Quantization' dialect
class QuantizationDialect : public Dialect {
public:
  QuantizationDialect(MLIRContext *context);

  /// Parse a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;
};

#define GET_OP_CLASSES
#include "mlir/Dialect/QuantOps/QuantOps.h.inc"

} // namespace quant
} // namespace mlir

#endif // MLIR_DIALECT_QUANTOPS_QUANTOPS_H_
