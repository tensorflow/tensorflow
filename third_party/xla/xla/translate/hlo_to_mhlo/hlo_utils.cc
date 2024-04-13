/* Copyright 2019 The OpenXLA Authors.

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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "xla/translate/hlo_to_mhlo/hlo_utils.h"

#include <cstddef>
#include <type_traits>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "xla/literal.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/primitive_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using mlir::AffineMap;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::ShapedType;

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(
    const ShapedType& type, const LiteralBase& literal) {
  if constexpr (std::is_same_v<CppType, u4> || std::is_same_v<CppType, s4>) {
    // DenseElementsAttr::get() does not support being passed an i4 array.
    // Instead, create buffer of padded i4 values and call
    // DenseElementsAttr::getFromRawBuffer()
    auto data_span = literal.data<CppType>();
    std::vector<char> int4_padded_data;
    int4_padded_data.reserve(literal.element_count());
    for (size_t i = 0; i < literal.element_count(); i++) {
      int4_padded_data.push_back(static_cast<char>(data_span[i]));
    }
    return ::mlir::DenseElementsAttr::getFromRawBuffer(type, int4_padded_data);
  } else {
    auto data_span = literal.data<CppType>();
    return ::mlir::DenseElementsAttr::get(
        type, llvm::ArrayRef(data_span.data(), data_span.size()));
  }
}

absl::StatusOr<AffineMap> GetPermutationIfAvailable(const Shape& shape,
                                                    mlir::Builder builder) {
  // N.B. IsMonotonicWithDim0Major ignores tiling, and I can't change it because
  // some XLA code relies on it treating tiled layouts as equivalent to untiled
  // layouts, so the check to rule out tiling has to come /before/ the
  // early-return branch, or we'd miss tiled monotonic layouts.
  if (!shape.layout().tiles().empty()) {
    return Internal("Tiled layouts are not yet supported");
  }
  if (!shape.has_layout() ||
      LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return AffineMap();
  }
  if (!shape.is_static()) {
    return Internal("Permutations for dynamic shapes are not yet supported");
  }
  int64_t accumulated_stride = 1;
  llvm::SmallVector<int64_t, 4> strides(shape.rank(), 1);
  for (int64_t dim : LayoutUtil::MinorToMajor(shape)) {
    strides[dim] = accumulated_stride;
    accumulated_stride *= shape.dimensions(dim);
  }
  if (accumulated_stride == 0) {
    return AffineMap();
  }
  return makeStridedLinearLayoutMap(strides, /*offset=*/0,
                                    builder.getContext());
}
}  // namespace

absl::StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMlirType(shape.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  using mlir::MemRefType;
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  auto permutation_or = GetPermutationIfAvailable(shape, builder);
  if (!permutation_or.ok()) return permutation_or.status();
  return MemRefType::get(array, element_type_or.value(),
                         permutation_or.value());
}

absl::StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  // TODO(hinsu): Support remaining XLA primitive types.
  auto element_type = literal.shape().element_type();
  return primitive_util::PrimitiveTypeSwitch<
      absl::StatusOr<mlir::DenseElementsAttr>>(
      [&](auto primitive_type_constant)
          -> absl::StatusOr<mlir::DenseElementsAttr> {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return CreateDenseAttrFromLiteral<
              primitive_util::NativeTypeOf<primitive_type_constant>>(type,
                                                                     literal);
        }
        return Internal("Unsupported type: %s",
                        PrimitiveType_Name(element_type));
      },
      element_type);
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(shape.empty() ? vector.size() : shape,
                                  builder.getIntegerType(64)),
      vector);
}

}  // namespace xla
