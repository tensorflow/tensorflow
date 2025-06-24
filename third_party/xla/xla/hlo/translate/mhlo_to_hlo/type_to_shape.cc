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

#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

using ::int64_t;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::ShapedType;
using mlir::VectorType;
using xla::PrimitiveType;

namespace xla {

std::optional<DimLevelType> ConvertDimLevelType(
    mlir::sparse_tensor::LevelType lt) {
  auto f = mlir::sparse_tensor::getLevelFormat(lt);
  if (!f) return std::nullopt;

  switch (*f) {
    case mlir::sparse_tensor::LevelFormat::Singleton:
      return DimLevelType::DIM_SINGLETON;
    case mlir::sparse_tensor::LevelFormat::Compressed:
      return DimLevelType::DIM_COMPRESSED;
    case mlir::sparse_tensor::LevelFormat::Dense:
      return DimLevelType::DIM_DENSE;
    case mlir::sparse_tensor::LevelFormat::LooseCompressed:
      return DimLevelType::DIM_LOOSE_COMPRESSED;
    default:
      return std::nullopt;
  }
}

Shape TypeToShape(mlir::Type type) {
  PrimitiveType ptype = ConvertMlirTypeToPrimitiveType(type);
  if (ptype != PrimitiveType::PRIMITIVE_TYPE_INVALID)
    return ShapeUtil::MakeShape(ptype, {});

  if (type.isIntOrFloat()) {
    auto* context = type.getContext();
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "lowering should have been handled by primitive type lowering for "
        << debugString(type);
  } else if (auto v = mlir::dyn_cast<mlir::VectorType>(type)) {
    llvm::SmallVector<int64_t, 4> span(v.getShape().begin(),
                                       v.getShape().end());
    mlir::Type element_type = v.getElementType();
    PrimitiveType primitive_type = ConvertMlirTypeToPrimitiveType(element_type);
    if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return ShapeUtil::MakeShape(primitive_type, span);
  } else if (auto m = mlir::dyn_cast<mlir::MemRefType>(type)) {
    llvm::SmallVector<int64_t, 6> span(m.getShape().begin(),
                                       m.getShape().end());
    mlir::Type element_type = m.getElementType();
    // Treat a memref of a vector as if it was a memref of primitive type with
    // the vector dimensions at the end.
    if (auto v = mlir::dyn_cast<mlir::VectorType>(element_type)) {
      element_type = v.getElementType();
      span.insert(span.end(), v.getShape().begin(), v.getShape().end());
    }
    PrimitiveType primitive_type = ConvertMlirTypeToPrimitiveType(element_type);
    if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) return {};
    // For the primitive type case, the shape of the memref is similar to the
    // vector type case (i.e., it is, modulo the layout, the same dimensions
    // and primitive type).
    if (m.getLayout().isIdentity())
      return ShapeUtil::MakeShape(primitive_type, span);

    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (failed(m.getStridesAndOffset(strides, offset))) return {};

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
  } else if (auto t = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    // TODO(jpienaar): This is only handling the base case with primitive
    // element type.
    int64_t rank = t.getRank();
    llvm::SmallVector<int64_t, 4> bounds;
    if (auto extn = mlir::dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>(
            t.getEncoding())) {
      bounds = llvm::to_vector<4>(extn.getBounds());
    } else if (auto extn =
                   mlir::dyn_cast_or_null<mlir::stablehlo::TypeExtensionsAttr>(
                       t.getEncoding())) {
      bounds = llvm::to_vector<4>(extn.getBounds());
    } else {
      bounds.assign(rank, ShapedType::kDynamic);
    }

    llvm::SmallVector<int64_t, 4> shape(rank, mlir::ShapedType::kDynamic);
    std::vector<bool> is_dynamic(rank, false);
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t size = t.getDimSize(dim);
      if (size == ShapedType::kDynamic) {
        shape[dim] = bounds[dim] != ShapedType::kDynamic
                         ? bounds[dim]
                         : Shape::kUnboundedSize;
        is_dynamic[dim] = true;
      } else {
        if (bounds[dim] != ShapedType::kDynamic) return {};
        shape[dim] = size;
      }
    }

    PrimitiveType primitive_type =
        ConvertMlirTypeToPrimitiveType(t.getElementType());
    if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) return {};

    if (auto sparse = mlir::sparse_tensor::getSparseTensorEncoding(type)) {
      // In this case `shape` has no bounds, because MHLO doesn't support
      // sparse tensors with bounded dynamism. This works out for us, because
      // neither does the shape_util MakeShape API.
      if (!t.hasStaticShape()) return {};

      // TODO(atondwal): Handle $posWidth, $crdWidth after they're
      // added to xla
      if (sparse.getPosWidth() != 32 || sparse.getCrdWidth() != 32) return {};

      std::vector<int64_t> ordering(rank);
      std::iota(ordering.rbegin(), ordering.rend(), 0);
      // Uses an identity map for dim ordering as the default value.
      auto dimToLvl = sparse.getDimToLvl()
                          ? sparse.getDimToLvl()
                          : mlir::AffineMap::getMultiDimIdentityMap(
                                rank, sparse.getContext());
      auto final_ordering = mlir::applyPermutationMap(
          dimToLvl, llvm::ArrayRef<int64_t>(ordering));
      return ::xla::ShapeUtil::MakeValidatedShapeWithSparseLayout(
                 primitive_type, shape, final_ordering)
          .value();
    }

    return ShapeUtil::MakeShape(primitive_type, shape, is_dynamic);
  } else if (auto tuple_type = mlir::dyn_cast<mlir::TupleType>(type)) {
    llvm::SmallVector<Shape, 4> shapes;
    shapes.reserve(tuple_type.size());
    for (mlir::Type sub_type : tuple_type.getTypes()) {
      shapes.push_back(TypeToShape(sub_type));
    }
    return ShapeUtil::MakeTupleShape(shapes);

  } else if (mlir::isa<mlir::mhlo::TokenType>(type) ||
             mlir::isa<mlir::stablehlo::TokenType>(type)) {
    return ShapeUtil::MakeTokenShape();
  } else if (auto bundle_type =
                 mlir::dyn_cast<mlir::mhlo::AsyncBundleType>(type)) {
    auto tuple_type =
        mlir::TupleType::get(type.getContext(), bundle_type.getTypes());
    return TypeToShape(tuple_type);
  }

  // Return empty XLA shape to signify error. No MLIR Type maps to a empty
  // Shape.
  return {};
}

}  // namespace xla
