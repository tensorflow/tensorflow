//===- CommonFolders.h - Common Operation Folders----------------*- C++ -*-===//
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
// This header file declares various common operation folders. These folders
// are intended to be used by dialects to support common folding behavior
// without requiring each dialect to provide its own implementation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_COMMONFOLDERS_H
#define MLIR_DIALECT_COMMONFOLDERS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
/// Performs constant folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class CalculationT =
              llvm::function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOp(llvm::ArrayRef<Attribute> operands,
                            const CalculationT &calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (!operands[0] || !operands[1])
    return {};
  if (operands[0].getType() != operands[1].getType())
    return {};

  if (operands[0].isa<AttrElementT>() && operands[1].isa<AttrElementT>()) {
    auto lhs = operands[0].cast<AttrElementT>();
    auto rhs = operands[1].cast<AttrElementT>();

    return AttrElementT::get(lhs.getType(),
                             calculate(lhs.getValue(), rhs.getValue()));
  } else if (operands[0].isa<SplatElementsAttr>() &&
             operands[1].isa<SplatElementsAttr>()) {
    // Both operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto lhs = operands[0].cast<SplatElementsAttr>();
    auto rhs = operands[1].cast<SplatElementsAttr>();

    auto elementResult = calculate(lhs.getSplatValue<ElementValueT>(),
                                   rhs.getSplatValue<ElementValueT>());
    return DenseElementsAttr::get(lhs.getType(), elementResult);
  } else if (operands[0].isa<ElementsAttr>() &&
             operands[1].isa<ElementsAttr>()) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto lhs = operands[0].cast<ElementsAttr>();
    auto rhs = operands[1].cast<ElementsAttr>();

    auto lhsIt = lhs.getValues<ElementValueT>().begin();
    auto rhsIt = rhs.getValues<ElementValueT>().begin();
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(lhs.getNumElements());
    for (size_t i = 0, e = lhs.getNumElements(); i < e; ++i, ++lhsIt, ++rhsIt)
      elementResults.push_back(calculate(*lhsIt, *rhsIt));
    return DenseElementsAttr::get(lhs.getType(), elementResults);
  }
  return {};
}
} // namespace mlir

#endif // MLIR_DIALECT_COMMONFOLDERS_H
