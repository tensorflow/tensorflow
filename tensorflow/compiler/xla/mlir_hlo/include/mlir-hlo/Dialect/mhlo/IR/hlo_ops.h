/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file defines the operations used in the MHLO dialect.

#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_H
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_H

#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/base.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include order below matters.
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.h.inc"

namespace mlir {
class OpBuilder;

namespace mhlo {

class MhloDialect : public Dialect {
 public:
  explicit MhloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "mhlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Registered hook to verify region arg attributes on operations.
  LogicalResult verifyRegionArgAttribute(mlir::Operation *op,
                                         unsigned regionIndex,
                                         unsigned argIndex,
                                         mlir::NamedAttribute attr) override;

  // Registered hook to verify an attribute from this dialect on operations.
  LogicalResult verifyOperationAttribute(mlir::Operation *op,
                                         mlir::NamedAttribute attr) override;

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;
};

class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
 public:
  using Base::Base;
};

void printConvolutionDimensions(AsmPrinter &p, ConvDimensionNumbersAttr dnums);
void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dnums);
ParseResult parseConvolutionDimensions(AsmParser &parser,
                                       ConvDimensionNumbersAttr &dnums);

}  // end namespace mhlo
}  // end namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h.inc"

namespace mlir {
namespace mhlo {

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction);

}  // end namespace mhlo
}  // end namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_H
