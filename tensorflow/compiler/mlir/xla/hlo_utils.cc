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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "tensorflow/compiler/mlir/xla/hlo_utils.h"

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
namespace {

using mlir::AffineMap;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::ShapedType;
using xla::LiteralBase;
using xla::StatusOr;

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFromLiteral(
    const ShapedType& type, const LiteralBase& literal) {
  auto data_span = literal.data<CppType>();
  return ::mlir::DenseElementsAttr::get(
      type, llvm::makeArrayRef(data_span.data(), data_span.size()));
}

mlir::APFloat ConvertToAPFloat(bfloat16 val) {
  return llvm::APFloat(llvm::APFloat::BFloat(), llvm::APInt(16, val.value));
}

mlir::APFloat ConvertToAPFloat(half val) {
  llvm::APFloat single_val = llvm::APFloat(static_cast<float>(val));
  bool loses_info = false;
  CHECK_EQ(single_val.convert(llvm::APFloat::IEEEhalf(),
                              llvm::APFloat::rmTowardZero, &loses_info),
           llvm::APFloat::opOK);
  CHECK(!loses_info);
  return single_val;
}

template <typename CppType>
::mlir::DenseElementsAttr CreateDenseAttrFrom16BitFloat(
    const ShapedType& type, const LiteralBase& literal) {
  auto data_span = literal.data<CppType>();
  llvm::SmallVector<mlir::APFloat, 4> vals;
  vals.reserve(data_span.size());
  for (CppType val : data_span) vals.push_back(ConvertToAPFloat(val));
  return ::mlir::DenseElementsAttr::get(type, vals);
}

StatusOr<llvm::SmallVector<AffineMap, 1>> GetPermutationIfAvailable(
    const Shape& shape, mlir::Builder builder) {
  if (!shape.has_layout() ||
      LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return llvm::SmallVector<AffineMap, 1>{};
  }
  if (!shape.is_static()) {
    return tensorflow::errors::Internal(
        "Permutations for dynamic shapes are not yet supported");
  }
  llvm::SmallVector<int64_t, 2> permuted_sizes;
  for (auto dim : llvm::reverse(shape.layout().minor_to_major())) {
    permuted_sizes.push_back(shape.dimensions(dim));
  }
  return llvm::SmallVector<AffineMap, 1>{AffineMap::get(
      permuted_sizes.size(), 0,
      makeCanonicalStridedLayoutExpr(permuted_sizes, builder.getContext()))};
}

}  // namespace

StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMLIRType(shape.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  using mlir::MemRefType;
  auto dimensions = shape.dimensions();
  llvm::SmallVector<int64_t, 4> array(dimensions.begin(), dimensions.end());
  auto permutation_or = GetPermutationIfAvailable(shape, builder);
  if (!permutation_or.ok()) return permutation_or.status();
  return MemRefType::get(array, element_type_or.ValueOrDie(),
                         permutation_or.ValueOrDie());
}

StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, Builder builder) {
  TF_ASSIGN_OR_RETURN(auto type,
                      ConvertTensorShapeToType<mlir::RankedTensorType>(
                          literal.shape(), builder));

  // TODO(hinsu): Support remaining XLA primitive types.
  auto element_type = literal.shape().element_type();
  switch (element_type) {
    case PrimitiveType::PRED:
      return CreateDenseAttrFromLiteral<bool>(type, literal);
    case PrimitiveType::F16:
      return CreateDenseAttrFrom16BitFloat<half>(type, literal);
    case PrimitiveType::BF16:
      return CreateDenseAttrFrom16BitFloat<bfloat16>(type, literal);
    case PrimitiveType::F32:
      return CreateDenseAttrFromLiteral<float>(type, literal);
    case PrimitiveType::F64:
      return CreateDenseAttrFromLiteral<double>(type, literal);
    case PrimitiveType::S8:
      return CreateDenseAttrFromLiteral<int8>(type, literal);
    case PrimitiveType::S16:
      return CreateDenseAttrFromLiteral<int16>(type, literal);
    case PrimitiveType::S32:
      return CreateDenseAttrFromLiteral<int32>(type, literal);
    case PrimitiveType::S64:
      return CreateDenseAttrFromLiteral<int64>(type, literal);
    case PrimitiveType::U8:
      return CreateDenseAttrFromLiteral<uint8>(type, literal);
    case PrimitiveType::U16:
      return CreateDenseAttrFromLiteral<uint16>(type, literal);
    case PrimitiveType::U32:
      return CreateDenseAttrFromLiteral<uint32>(type, literal);
    case PrimitiveType::U64:
      return CreateDenseAttrFromLiteral<uint64>(type, literal);
    case PrimitiveType::C64:
      return CreateDenseAttrFromLiteral<complex64>(type, literal);
    case PrimitiveType::C128:
      return CreateDenseAttrFromLiteral<complex128>(type, literal);
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape) {
  return mlir::DenseIntElementsAttr::get(
      mlir::RankedTensorType::get(shape.empty() ? vector.size() : shape,
                                  builder.getIntegerType(64)),
      vector);
}

StatusOr<mlir::Type> ConvertPrimitiveTypeToMLIRType(PrimitiveType element_type,
                                                    mlir::Builder builder) {
  switch (element_type) {
    case PrimitiveType::PRED:
      return builder.getI1Type();
    case PrimitiveType::F16:
      return builder.getF16Type();
    case PrimitiveType::BF16:
      return builder.getBF16Type();
    case PrimitiveType::F32:
      return builder.getF32Type();
    case PrimitiveType::F64:
      return builder.getF64Type();
    case PrimitiveType::S8:
      return builder.getIntegerType(8);
    case PrimitiveType::S16:
      return builder.getIntegerType(16);
    case PrimitiveType::S32:
      return builder.getIntegerType(32);
    case PrimitiveType::S64:
      return builder.getIntegerType(64);
    case PrimitiveType::U8:
      return builder.getIntegerType(8, /*isSigned=*/false);
    case PrimitiveType::U16:
      return builder.getIntegerType(16, /*isSigned=*/false);
    case PrimitiveType::U32:
      return builder.getIntegerType(32, /*isSigned=*/false);
    case PrimitiveType::U64:
      return builder.getIntegerType(64, /*isSigned=*/false);
    case PrimitiveType::C64:
      return mlir::ComplexType::get(builder.getF32Type());
    case PrimitiveType::C128:
      return mlir::ComplexType::get(builder.getF64Type());
    // TODO(b/130356985): Support unsigned primitive types.
    default:
      return tensorflow::errors::Internal(
          absl::StrCat("Unsupported type: ", PrimitiveType_Name(element_type)));
  }
}

mlir::xla_hlo::GatherDimensionNumbers CreateGatherDimensionNumbers(
    const GatherDimensionNumbers& input, mlir::Builder builder) {
  auto offset_dims = CreateDenseIntElementsAttrFromVector(
      llvm::SmallVector<int64, 4>{input.offset_dims().begin(),
                                  input.offset_dims().end()},
      builder);
  auto collapsed_slice_dims = CreateDenseIntElementsAttrFromVector(
      llvm::SmallVector<int64, 4>{input.collapsed_slice_dims().begin(),
                                  input.collapsed_slice_dims().end()},
      builder);
  auto start_index_map = CreateDenseIntElementsAttrFromVector(
      llvm::SmallVector<int64, 4>{input.start_index_map().begin(),
                                  input.start_index_map().end()},
      builder);

  mlir::IntegerAttr index_vector_dim =
      builder.getI64IntegerAttr(input.index_vector_dim());

  return mlir::xla_hlo::GatherDimensionNumbers::get(
      offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim,
      builder.getContext());
}

}  // namespace xla
