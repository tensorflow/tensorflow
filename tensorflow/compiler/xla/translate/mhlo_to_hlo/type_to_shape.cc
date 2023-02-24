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

#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/type_to_shape.h"

#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "mlir/Dialect/SparseTensor/IR/Enums.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

using ::int64_t;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::VectorType;
using mlir::mhlo::TypeExtensionsAttr;
using xla::PrimitiveType;
using xla::ShapeUtil;

namespace xla {

PrimitiveType TypeToPrimitiveType(mlir::Type type) {
  if (type.isFloat8E5M2()) {
    return PrimitiveType::F8E5M2;
  } else if (type.isFloat8E4M3FN()) {
    return PrimitiveType::F8E4M3FN;
  } else if (type.isBF16()) {
    return PrimitiveType::BF16;
  } else if (type.isF16()) {
    return PrimitiveType::F16;
  } else if (type.isF32()) {
    return PrimitiveType::F32;
  } else if (type.isF64()) {
    return PrimitiveType::F64;
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    mlir::Type element_ty = complex_type.getElementType();
    if (element_ty.isF32()) {
      return PrimitiveType::C64;

    } else if (element_ty.isF64()) {
      return PrimitiveType::C128;
    }
    return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  } else if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    bool is_unsigned = integer_type.isUnsigned();
    switch (integer_type.getWidth()) {
      case 1:
        return PrimitiveType::PRED;
      case 8:
        return is_unsigned ? PrimitiveType::U8 : PrimitiveType::S8;
      case 16:
        return is_unsigned ? PrimitiveType::U16 : PrimitiveType::S16;
      case 32:
        return is_unsigned ? PrimitiveType::U32 : PrimitiveType::S32;
      case 64:
        return is_unsigned ? PrimitiveType::U64 : PrimitiveType::S64;
      default:
        return PrimitiveType::PRIMITIVE_TYPE_INVALID;
    }
  }
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

std::optional<std::tuple<DimLevelType, bool, bool>> ConvertDimLevelType(
    mlir::sparse_tensor::DimLevelType dlt) {
  auto f = mlir::sparse_tensor::getLevelFormat(dlt);
  if (!f) return std::nullopt;

  bool unique = mlir::sparse_tensor::isUniqueDLT(dlt);
  bool ordered = mlir::sparse_tensor::isOrderedDLT(dlt);
  switch (*f) {
    case mlir::sparse_tensor::LevelFormat::Singleton:
      return std::make_tuple(DimLevelType::DIM_SINGLETON, unique, ordered);
    case mlir::sparse_tensor::LevelFormat::Compressed:
      return std::make_tuple(DimLevelType::DIM_COMPRESSED, unique, ordered);
    case mlir::sparse_tensor::LevelFormat::Dense:
      return std::make_tuple(DimLevelType::DIM_DENSE, unique, ordered);
    default:
      return std::nullopt;
  }
}

Shape TypeToShape(mlir::Type type) {
  PrimitiveType ptype = TypeToPrimitiveType(type);
  if (ptype != PrimitiveType::PRIMITIVE_TYPE_INVALID)
    return ShapeUtil::MakeShape(ptype, {});

  if (type.isIntOrFloat()) {
    auto* context = type.getContext();
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "lowering should have been handled by primitive type lowering for "
        << debugString(type);
  } else if (auto v = type.dyn_cast<mlir::VectorType>()) {
    llvm::SmallVector<int64_t, 4> span(v.getShape().begin(),
                                       v.getShape().end());
    mlir::Type element_type = v.getElementType();
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return ShapeUtil::MakeShape(primitive_type, span);
  } else if (auto m = type.dyn_cast<mlir::MemRefType>()) {
    llvm::SmallVector<int64_t, 6> span(m.getShape().begin(),
                                       m.getShape().end());
    mlir::Type element_type = m.getElementType();
    // Treat a memref of a vector as if it was a memref of primitive type with
    // the vector dimensions at the end.
    if (auto v = element_type.dyn_cast<mlir::VectorType>()) {
      element_type = v.getElementType();
      span.insert(span.end(), v.getShape().begin(), v.getShape().end());
    }
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) return {};
    // For the primitive type case, the shape of the memref is similar to the
    // vector type case (i.e., it is, modulo the layout, the same dimensions
    // and primitive type).
    if (m.getLayout().isIdentity())
      return ShapeUtil::MakeShape(primitive_type, span);

    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (failed(mlir::getStridesAndOffset(m, strides, offset))) return {};

    llvm::SmallVector<std::pair<int64_t, int>, 4> strides_with_indices;
    for (const auto& e : llvm::enumerate(strides)) {
      strides_with_indices.push_back({e.value(), e.index()});
    }
    std::stable_sort(strides_with_indices.begin(), strides_with_indices.end());

    llvm::SmallVector<int64_t, 4> minor_to_major;
    int64_t stride = 1;
    for (const auto& pr : strides_with_indices) {
      minor_to_major.push_back(pr.second);

      // Either the affine map is not perfectly strided, or the dimensions
      // recovered from strides don't match the actual dimensions in shapes.
      if (stride != pr.first && m.getShape()[pr.second] != 1) return {};

      stride *= m.getShape()[pr.second];
    }

    llvm::SmallVector<int64_t, 4> dimensions(m.getShape().begin(),
                                             m.getShape().end());
    return ::xla::ShapeUtil::MakeShapeWithDenseLayout(
        primitive_type, dimensions, minor_to_major);
  } else if (auto t = type.dyn_cast<mlir::RankedTensorType>()) {
    // TODO(jpienaar): This is only handling the base case with primitive
    // element type.
    int64_t rank = t.getRank();
    llvm::SmallVector<int64_t, 4> bounds;
    if (auto extn = t.getEncoding().dyn_cast_or_null<TypeExtensionsAttr>()) {
      bounds = llvm::to_vector<4>(extn.getBounds());
    } else {
      bounds.assign(rank, ShapedType::kDynamic);
    }

    llvm::SmallVector<int64_t, 4> shape(rank, mlir::ShapedType::kDynamic);
    std::vector<bool> is_dynamic(rank, false);
    for (int64_t dim = 0; dim < rank; ++dim) {
      // Only fully static shapes are supported.
      // TODO(b/115638799): Update once xla::Shape can support dynamic shapes.
      int64_t size = t.getDimSize(dim);
      if (size == ShapedType::kDynamic) {
        if (bounds[dim] == ShapedType::kDynamic) return {};
        shape[dim] = bounds[dim];
        is_dynamic[dim] = true;
      } else {
        if (bounds[dim] != ShapedType::kDynamic) return {};
        shape[dim] = size;
      }
    }

    PrimitiveType primitive_type = TypeToPrimitiveType(t.getElementType());
    if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) return {};

    if (auto sparse = mlir::sparse_tensor::getSparseTensorEncoding(type)) {
      // In this case `shape` has no bounds, because MHLO doesn't support
      // sparse tensors with bounded dynamism. This works out for us, because
      // neither does the shape_util MakeShape API.
      if (!t.hasStaticShape()) return {};

      // TODO(atondwal): Handle $pointerBitWidth, $indexBitWidth after they're
      // added to xla
      if (sparse.getPointerBitWidth() != 32 || sparse.getIndexBitWidth() != 32)
        return {};

      llvm::SmallVector<DimLevelType, 3> dim_level_types;
      llvm::SmallVector<bool, 3> level_unique;
      llvm::SmallVector<bool, 3> level_ordered;
      for (auto dlt : sparse.getDimLevelType()) {
        auto new_dlt = ConvertDimLevelType(dlt);
        if (!new_dlt) return {};
        dim_level_types.push_back(std::get<0>(*new_dlt));
        level_unique.push_back(std::get<1>(*new_dlt));
        level_ordered.push_back(std::get<2>(*new_dlt));
      }

      std::vector<int64_t> ordering(rank);
      std::iota(ordering.rbegin(), ordering.rend(), 0);
      // Uses an identity map for dim ordering as the default value.
      auto dimOrder = sparse.getDimOrdering()
                          ? sparse.getDimOrdering()
                          : mlir::AffineMap::getMultiDimIdentityMap(
                                rank, sparse.getContext());
      auto final_ordering = mlir::applyPermutationMap(
          dimOrder, llvm::ArrayRef<int64_t>(ordering));
      auto sparse_shape = ::xla::ShapeUtil::MakeShapeWithSparseLayout(
          primitive_type, shape, final_ordering, dim_level_types, level_unique,
          level_ordered);
      return sparse_shape;
    }

    return ShapeUtil::MakeShape(primitive_type, shape, is_dynamic);
  } else if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    llvm::SmallVector<Shape, 4> shapes;
    shapes.reserve(tuple_type.size());
    for (mlir::Type sub_type : tuple_type.getTypes()) {
      shapes.push_back(TypeToShape(sub_type));
    }
    return ShapeUtil::MakeTupleShape(shapes);

  } else if (type.isa<mlir::mhlo::TokenType>()) {
    return ShapeUtil::MakeTokenShape();
  } else if (auto bundle_type = type.dyn_cast<mlir::mhlo::AsyncBundleType>()) {
    auto tuple_type =
        mlir::TupleType::get(type.getContext(), bundle_type.getTypes());
    return TypeToShape(tuple_type);
  }

  // Return empty XLA shape to signify error. No MLIR Type maps to a empty
  // Shape.
  return {};
}

}  // namespace xla
