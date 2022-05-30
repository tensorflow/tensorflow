/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file defines enums used in MHLO and LMHLO.
#ifndef MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ATTRS_H
#define MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ATTRS_H

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_enums.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_attrs.h.inc"

namespace mlir {
namespace mhlo {
// Custom printer and parser for struct attributes.
void printConvolutionDimensions(AsmPrinter &p, ConvDimensionNumbersAttr dnums);
void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dnums);
ParseResult parseConvolutionDimensions(AsmParser &parser,
                                       ConvDimensionNumbersAttr &dnums);

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_BASE_ATTRS_H
