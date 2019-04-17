//===- Quantization/QuantOps.h - Quantization Ops and Types -----*- C++ -*-===//
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

#ifndef MLIR_QUANTIZATION_QUANTOPS_H_
#define MLIR_QUANTIZATION_QUANTOPS_H_

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
  Type parseType(StringRef spec, Location loc) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, raw_ostream &os) const override;
};

#define GET_OP_CLASSES
#include "mlir/Quantization/QuantOps.h.inc"

} // namespace quant
} // namespace mlir

#endif // MLIR_QUANTIZATION_QUANTOPS_H_
