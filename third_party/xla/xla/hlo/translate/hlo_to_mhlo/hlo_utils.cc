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

#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
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
  if constexpr (is_intN_v<CppType>) {
    // DenseElementsAttr::get() does not support being passed an i4 array.
    // Instead, create buffer of padded, packed values and call
    // DenseElementsAttr::getFromRawBuffer()
    auto data_span = literal.data<CppType>();
    std::vector<char> packed_padded_data;
    packed_padded_data.reserve(literal.element_count());
    for (size_t i = 0; i < literal.element_count(); i++) {
      packed_padded_data.push_back(static_cast<char>(data_span[i]));
    }
    return ::mlir::DenseElementsAttr::getFromRawBuffer(type,
                                                       packed_padded_data);
  } else if constexpr (std::is_same_v<CppType, tsl::float4_e2m1fn>) {
    // DenseElementsAttr::get() does not support being passed an array of
    // tsl::float4_e2m1fn. So convert each element to APFloat first.
    auto data_span = literal.data<CppType>();
    std::vector<llvm::APFloat> apfloats;
    apfloats.reserve(literal.element_count());
    for (size_t i = 0; i < literal.element_count(); i++) {
      llvm::APFloat apfloat{static_cast<float>(data_span[i])};
      bool losesInfo;
      llvm::APFloat::opStatus status =
          apfloat.convert(llvm::APFloat::Float4E2M1FN(),
                          llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      CHECK_EQ(status, llvm::APFloat::opOK)
          << "Failed to convert " << data_span[i] << " to Float4E2M1FN APFloat";
      CHECK(!losesInfo) << "Lost info when converting " << data_span[i]
                        << " to Float4E2M1FN APFloat";
      apfloats.push_back(apfloat);
    }
    return ::mlir::DenseElementsAttr::get(type, apfloats);
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
  llvm::SmallVector<int64_t, 4> strides(shape.dimensions().size(), 1);
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

namespace {
bool HasMhloTokenType(mlir::TypeRange types) {
  bool use_mhlo = false;
  for (auto type : types) {
    if (!use_mhlo) {
      type.walk([&](mlir::Type type) {
        use_mhlo |= llvm::isa<mlir::mhlo::TokenType>(type);
        if (use_mhlo) return mlir::WalkResult::interrupt();
        return mlir::WalkResult::advance();
      });
    }
  }
  return use_mhlo;
}

}  // namespace

mlir::Value CreateTupleValue(mlir::OpBuilder* func_builder, mlir::Location loc,
                             mlir::ValueRange& flatten_values,
                             mlir::Type type) {
  auto tuple_type = type.dyn_cast<mlir::TupleType>();
  if (!tuple_type) {
    assert(!flatten_values.empty());
    auto retval = flatten_values.front();
    flatten_values = flatten_values.drop_front();
    return retval;
  }

  llvm::SmallVector<mlir::Value> flatten_sub_values;
  for (auto child_type : tuple_type.getTypes())
    flatten_sub_values.push_back(
        CreateTupleValue(func_builder, loc, flatten_values, child_type));

  if (HasMhloTokenType(mlir::TypeRange(flatten_sub_values))) {
    return func_builder->create<mlir::mhlo::TupleOp>(loc, flatten_sub_values)
        .getResult();
  }
  return func_builder->create<mlir::stablehlo::TupleOp>(loc, flatten_sub_values)
      .getResult();
}

mlir::Operation* CreateTupleFromOpResults(mlir::OpBuilder* func_builder,
                                          mlir::Location loc,
                                          mlir::Operation* op,
                                          mlir::Type type) {
  if (!type.isa<mlir::TupleType>()) return op;

  mlir::ValueRange flattened_results_ref(op->getResults());
  auto result =
      CreateTupleValue(func_builder, loc, flattened_results_ref, type);
  mlir::Operation* tuple_op = result.getDefiningOp<mlir::mhlo::TupleOp>();
  if (!tuple_op) {
    tuple_op = result.getDefiningOp<mlir::stablehlo::TupleOp>();
  }
  assert(tuple_op && "builder didn't return the right type");
  return tuple_op;
}

mlir::Operation* WrapVariadicResultsInTuple(mlir::OpBuilder* builder,
                                            mlir::Location loc,
                                            mlir::Operation* op) {
  auto result_types = op->getResultTypes();
  // Consider skipping wrapping result type of size 1.
  assert(result_types.size() != 1 ||
         !llvm::isa<mlir::TupleType>(result_types[0]) &&
             "Cannot wrap single tuple arg in tuple");

  auto tuple_type = builder->getTupleType(result_types);
  return CreateTupleFromOpResults(builder, loc, op, tuple_type);
}

bool IsEmptyTuple(const mlir::Type& type) {
  if (auto tuple_type = llvm::dyn_cast<mlir::TupleType>(type)) {
    return tuple_type.getTypes().empty();
  }
  return false;
}

mlir::TypeRange Untuple(const mlir::Type& type) {
  if (llvm::isa<mlir::TupleType>(type)) {
    return llvm::dyn_cast<mlir::TupleType>(type).getTypes();
  }
  return type;
}

}  // namespace xla
