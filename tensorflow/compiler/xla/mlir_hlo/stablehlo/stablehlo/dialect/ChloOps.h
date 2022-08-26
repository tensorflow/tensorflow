/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_CHLO_OPS_H
#define STABLEHLO_DIALECT_CHLO_OPS_H

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "stablehlo/dialect/Base.h"

// Include order matters
#include "stablehlo/dialect/ChloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/ChloAttrs.h.inc"

namespace mlir {
namespace chlo {

class ChloDialect : public Dialect {
 public:
  explicit ChloDialect(MLIRContext* context);
  static StringRef getDialectNamespace() { return "chlo"; }

  Operation* materializeConstant(OpBuilder& builder, Attribute value, Type type,
                                 Location loc) override;

  Attribute parseAttribute(DialectAsmParser& parser, Type type) const override;

  void printAttribute(Attribute attr, DialectAsmPrinter& os) const override;
};

}  // namespace chlo
}  // namespace mlir

namespace mlir {
namespace chlo {
namespace OpTrait {

template <typename ConcreteType>
class Broadcasting
    : public mlir::OpTrait::TraitBase<ConcreteType, Broadcasting> {};

}  // namespace OpTrait
}  // namespace chlo
}  // namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/dialect/ChloOps.h.inc"

#endif  // STABLEHLO_DIALECT_CHLO_OPS_H
