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

#ifndef MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_H
#define MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_H

// This file defines functionality shared between chlo/mhlo/lhlo dialects.

#include <algorithm>
#include <optional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace hlo {

// Verifies the source target pairs attached to collective permute.
LogicalResult verifyCollectivePermuteSourceTargetPairs(
    Operation* op, DenseIntElementsAttr attr);

LogicalResult verifyReduceScatter(Operation* op, TypeRange operandTypes,
                                  TypeRange resultTypes,
                                  uint64_t scatterDimension);

// Custom formatting for convolution window attributes.
void printWindowAttributes(OpAsmPrinter& p, Operation* op,
                           std::optional<DenseIntElementsAttr> windowStrides,
                           std::optional<DenseIntElementsAttr> padding,
                           std::optional<DenseIntElementsAttr> lhsDilation,
                           std::optional<DenseIntElementsAttr> rhsDilation,
                           std::optional<DenseElementsAttr> windowReversal);

ParseResult parseWindowAttributes(OpAsmParser& parser,
                                  DenseIntElementsAttr& windowStrides,
                                  DenseIntElementsAttr& padding,
                                  DenseIntElementsAttr& lhsDilation,
                                  DenseIntElementsAttr& rhsDilation,
                                  DenseElementsAttr& windowReversal);

}  // namespace hlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_IR_HLO_OPS_COMMON_H
