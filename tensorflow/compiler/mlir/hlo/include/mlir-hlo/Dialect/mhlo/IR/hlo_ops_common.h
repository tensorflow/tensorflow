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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_COMMON_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_COMMON_H_

// This file defines functionality shared between chlo/mhlo/lhlo dialects.

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace hlo {

// Verifies the source target pairs attached to collective permute.
LogicalResult VerifyCollectivePermuteSourceTargetPairs(
    Operation* op, DenseIntElementsAttr attr);

LogicalResult VerifyReduceScatter(Operation* op, TypeRange operand_types,
                                  TypeRange result_types,
                                  uint64_t scatter_dimension);

// Custom formatting for convolution window attributes.
void printWindowAttributes(OpAsmPrinter& p, Operation* op,
                           llvm::Optional<DenseIntElementsAttr> window_strides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhs_dilation,
                           llvm::Optional<DenseIntElementsAttr> rhs_dilation,
                           llvm::Optional<DenseElementsAttr> window_reversal);

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& window_strides,
                                  DenseIntElementsAttr& padding,
                                  DenseIntElementsAttr& lhs_dilation,
                                  DenseIntElementsAttr& rhs_dilation,
                                  DenseElementsAttr& window_reversal);

}  // namespace hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_DIALECT_MHLO_IR_HLO_OPS_COMMON_H_
