/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/xtile/ir/xtile_attrs.h"

#include <cstdint>

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

namespace xla::xtile {

mlir::AffineMap LayoutAttr::getAffineMap() const {
  return mlir::AffineMap::getPermutationMap(getMinorToMajor(), getContext());
}

mlir::LogicalResult LayoutAttr::verifyLayout(
    mlir::ArrayRef<int64_t> shape,
    mlir::function_ref<mlir::InFlightDiagnostic()> emit_error) const {
  if (getMinorToMajor().size() != shape.size()) {
    emit_error() << "layout has " << getMinorToMajor().size()
                 << " dimensions, but shape has " << shape.size();
    return mlir::failure();
  }
  if (!mlir::isPermutationVector(getMinorToMajor().asArrayRef())) {
    emit_error() << "layout is not a permutation";
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult LayoutAttr::getStridesAndOffset(
    mlir::ArrayRef<int64_t> shape, mlir::SmallVectorImpl<int64_t>& strides,
    int64_t& offset) const {
  strides.resize(shape.size());
  int64_t size_product = 1;
  for (auto dim : getMinorToMajor().asArrayRef()) {
    strides[dim] = size_product;
    size_product *= shape[dim];
  }
  offset = 0;
  return mlir::success();
}

}  // namespace xla::xtile
