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

#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

namespace xla::xtile {

using mlir::AffineMap;
using mlir::ArrayRef;
using mlir::failure;
using mlir::function_ref;
using mlir::InFlightDiagnostic;
using mlir::isPermutationVector;
using mlir::LogicalResult;
using mlir::SmallVectorImpl;
using mlir::success;

AffineMap LayoutAttr::getAffineMap() const {
  return AffineMap::getPermutationMap(getMinorToMajor(), getContext());
}

LogicalResult LayoutAttr::verifyLayout(
    ArrayRef<int64_t> shape,
    function_ref<InFlightDiagnostic()> emit_error) const {
  if (getMinorToMajor().size() != shape.size()) {
    emit_error() << "layout has " << getMinorToMajor().size()
                 << " dimensions, but shape has " << shape.size();
    return failure();
  }
  if (!isPermutationVector(getMinorToMajor().asArrayRef())) {
    emit_error() << "layout is not a permutation";
    return failure();
  }
  return success();
}

LogicalResult LayoutAttr::getStridesAndOffset(ArrayRef<int64_t> shape,
                                              SmallVectorImpl<int64_t>& strides,
                                              int64_t& offset) const {
  strides.resize(shape.size());
  int64_t size_product = 1;
  for (auto dim : getMinorToMajor().asArrayRef()) {
    strides[dim] = size_product;
    size_product *= shape[dim];
  }
  offset = 0;
  return success();
}

}  // namespace xla::xtile
