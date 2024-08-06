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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_HLO_UTILS_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_HLO_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<mlir::DenseElementsAttr> CreateDenseElementsAttrFromLiteral(
    const LiteralBase& literal, mlir::Builder builder);

// Creates an DenseIntElementsAttr using the elements of the vector and the
// optional shape.
mlir::DenseIntElementsAttr CreateDenseIntElementsAttrFromVector(
    const llvm::ArrayRef<int64_t> vector, mlir::Builder builder,
    llvm::ArrayRef<int64_t> shape = {});

// Converts the given XLA shape for tensors to the template MLIR type.
template <typename TypeT>
static absl::StatusOr<TypeT> ConvertTensorShapeToType(const Shape& xla_ty,
                                                      mlir::Builder builder) {
  auto element_type_or =
      ConvertPrimitiveTypeToMlirType(xla_ty.element_type(), builder);
  if (!element_type_or.ok()) return element_type_or.status();

  bool is_bounded_dynamic = false;
  int64_t rank = xla_ty.rank();
  llvm::SmallVector<int64_t, 4> shape(rank, mlir::ShapedType::kDynamic);
  llvm::SmallVector<int64_t, 4> bounds(rank, mlir::ShapedType::kDynamic);
  for (int64_t dim = 0; dim < rank; ++dim) {
    int64_t dim_size = xla_ty.dimensions(dim);
    if (xla_ty.is_dynamic_dimension(dim)) {
      if (!xla_ty.is_unbounded_dynamic_dimension(dim)) {
        bounds[dim] = dim_size;
        is_bounded_dynamic = true;
      }
    } else {
      shape[dim] = dim_size;
    }
  }
  using mlir::mhlo::TypeExtensionsAttr;
  mlir::Attribute encoding;
  if (is_bounded_dynamic) {
    encoding = TypeExtensionsAttr::get(builder.getContext(), bounds);
  }

  using mlir::sparse_tensor::SparseTensorEncodingAttr;
  // TODO(b/238903065): We don't yet support bounded dynamism shapes and
  // sparsity at the same time, as we can currently only have one `encoding` on
  // a RankedTensorType, and we don't currently have a meet of
  // SparseTensorEncodingAttr and TypeExtensionsAttr (which holds bounds).
  //
  // For example, we wouldn't be able to represent the xla type
  // `f32[4,<=4]{1,0:D(D,C)}`.
  if (xla_ty.has_layout()) {
    auto layout = xla_ty.layout();
    if (LayoutUtil::IsSparse(layout)) {
      if (is_bounded_dynamic)
        return Unimplemented(
            "MHLO doesn't support bounded dynamic shapes for sparse tensors");
      llvm::SmallVector<mlir::sparse_tensor::LevelType> lts;
      for (size_t i = 0, e = layout.dim_level_types_size(); i < e; ++i) {
        auto dlt = layout.dim_level_type(i);
        bool ordered =
            i < layout.dim_ordered_size() ? layout.dim_ordered(i) : true;
        bool unique =
            i < layout.dim_unique_size() ? layout.dim_unique(i) : true;
        switch (dlt) {
          case DimLevelType::DIM_DENSE:
            lts.push_back(*mlir::sparse_tensor::buildLevelType(
                mlir::sparse_tensor::LevelFormat::Dense, ordered, unique));
            break;
          case DimLevelType::DIM_COMPRESSED:
            lts.push_back(*mlir::sparse_tensor::buildLevelType(
                mlir::sparse_tensor::LevelFormat::Compressed, ordered, unique));
            break;
          case DimLevelType::DIM_SINGLETON:
            lts.push_back(*mlir::sparse_tensor::buildLevelType(
                mlir::sparse_tensor::LevelFormat::Singleton, ordered, unique));
            break;
          case DimLevelType::DIM_LOOSE_COMPRESSED:
            lts.push_back(*mlir::sparse_tensor::buildLevelType(
                mlir::sparse_tensor::LevelFormat::LooseCompressed, ordered,
                unique));
            break;
          default:
            return InvalidArgument("Unknown DimLevelType from HLO");
        }
      }
      auto ordering = layout.minor_to_major();
      llvm::SmallVector<uint32_t> major_to_minor = {ordering.rbegin(),
                                                    ordering.rend()};
      auto id_map = mlir::AffineMap::getPermutationMap(major_to_minor,
                                                       builder.getContext());
      // TODO(atondwal): support sizes other than 32 when XLA does
      encoding = SparseTensorEncodingAttr::get(
          builder.getContext(), lts, id_map, mlir::AffineMap(), 32, 32);
    }
  }
  return TypeT::get(shape, element_type_or.value(), encoding);
}

absl::StatusOr<mlir::MemRefType> ConvertTensorShapeToMemRefType(
    const Shape& shape, mlir::Builder builder);

template <>
inline absl::StatusOr<mlir::MemRefType> ConvertTensorShapeToType(
    const Shape& shape, mlir::Builder builder) {
  if (shape.is_dynamic()) {
    return FailedPrecondition(  // NOLINT
        "MemRefType don't support dynamic shapes");
  }
  return ConvertTensorShapeToMemRefType(shape, builder);
}

// Converts the given XLA shape to the template MLIR type.
template <typename TypeT>
static absl::StatusOr<mlir::Type> ConvertShapeToType(const Shape& shape,
                                                     mlir::Builder builder) {
  if (shape.IsTuple()) {
    llvm::SmallVector<mlir::Type, 4> contents;
    contents.reserve(shape.tuple_shapes_size());
    for (const auto& subtype : shape.tuple_shapes()) {
      TF_ASSIGN_OR_RETURN(auto mlir_subtype,
                          ConvertShapeToType<TypeT>(subtype, builder));
      contents.push_back(mlir_subtype);
    }
    return builder.getTupleType(contents);
  }
  if (shape.IsToken()) {
    return mlir::mhlo::TokenType::get(builder.getContext());
  }
  return ConvertTensorShapeToType<TypeT>(shape, builder);
}

// CreateTupleValue creates a root TupleOp of (nested) tuple-type 'type' using
// the non-tuple-typed values in 'flatten_values'.
//
// e.g., Given 'flatten_values': [V1, V2, V3] &'type': tuple<T1,tuple<T1,T2>>,
//      The function returns %t2 such that:
//       %t1 = mhlo.tuple(V2,V3) : (T2,T3) -> tuple<T2,T3>
//       %t2 = mhlo.tuple(V1,%t1): (T1,tuple<T2,T3>) -> tuple<T1,tuple<T1,T2>>
//
// Note: 1. FlattenTupleValue and CreateTupleValue is a pair of functions to
//          resp. flatten and create tuples in the exact same order.
//       2. `flatten_values`, initially storing the flattened values, will be
//          mutated to a 0-length array by the end of function invocation.
mlir::Value CreateTupleValue(mlir::OpBuilder* func_builder, mlir::Location loc,
                             mlir::ValueRange& flatten_values, mlir::Type type);

// Create a TupleOp using the results of 'op' if 'type' is a mlir::TupleType.
// Otherwise, return 'op'.
mlir::Operation* CreateTupleFromOpResults(mlir::OpBuilder* func_builder,
                                          mlir::Location loc,
                                          mlir::Operation* op, mlir::Type type);

mlir::TypeRange Untuple(const mlir::Type& type);

static std::pair<mlir::Attribute, mlir::ArrayAttr> GetLayoutAttribute(
    mlir::Builder& b, const Shape& shape,
    std::optional<const Layout> maybe_layout = std::nullopt) {
  if (shape.IsTuple()) {
    llvm::SmallVector<mlir::Attribute> element_attrs;
    llvm::SmallVector<mlir::Attribute> tile_attrs;
    for (const auto& tuple_shape : shape.tuple_shapes()) {
      // TODO here we do not dissect the layout of a tuple into sublayouts.
      // Presently ShapeLayout cannot represent an explicit layout for a tuple
      // type so this should never occur. However, if this function were to
      // be used in another context where this assumption were to be lifted.
      // users should be aware of this limitation which will use the default
      // layout for tuple subshapes.
      std::pair<mlir::Attribute, mlir::Attribute> inner =
          GetLayoutAttribute(b, tuple_shape);
      element_attrs.push_back(inner.first);
      tile_attrs.push_back(inner.second);
    }
    return std::make_pair((mlir::Attribute)b.getArrayAttr(element_attrs),
                          b.getArrayAttr(tile_attrs));
  }

  Layout layout = maybe_layout.value_or(
      shape.has_layout() ? shape.layout()
                         : LayoutUtil::GetDefaultLayoutForShape(shape));

  llvm::SmallVector<mlir::Attribute> vec_of_tiles;
  for (const Tile& tile : layout.tiles()) {
    llvm::SmallVector<int64_t> tile_vec = {tile.dimensions().begin(),
                                           tile.dimensions().end()};
    vec_of_tiles.push_back(b.getIndexTensorAttr(tile_vec));
  }
  llvm::SmallVector<int64_t> layout_vec = {layout.minor_to_major().begin(),
                                           layout.minor_to_major().end()};
  return std::make_pair(b.getIndexTensorAttr(layout_vec),
                        b.getArrayAttr(vec_of_tiles));
}

static bool HasCustomLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    return llvm::any_of(shape.tuple_shapes(), HasCustomLayout);
  }
  return shape.has_layout() && !shape.layout().minor_to_major().empty() &&
         shape.layout() != LayoutUtil::GetDefaultLayoutForShape(shape);
}

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_HLO_UTILS_H_
