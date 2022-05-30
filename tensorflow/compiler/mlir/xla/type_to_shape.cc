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

#include "tensorflow/compiler/mlir/xla/type_to_shape.h"

#include <string>

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using ::int64_t;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::VectorType;
using xla::PrimitiveType;
using xla::ShapeUtil;

namespace xla {

PrimitiveType TypeToPrimitiveType(mlir::Type type) {
  if (type.isBF16()) {
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

StatusOr<Shape> TypeToShape(
    mlir::Type type, CustomShapeRepresentationFn shape_representation_fn) {
  tensorflow::PartialTensorShape partial_tensor_shape =
      tensorflow::ConvertTypeToTensorShape(type);

  tensorflow::TensorShape fully_defined_tensor_shape;
  if (!partial_tensor_shape.AsTensorShape(&fully_defined_tensor_shape)) {
    return tensorflow::errors::InvalidArgument(
        "XLA HLO only allows fully-defined shape");
  }

  tensorflow::DataType dtype;
  TF_RETURN_IF_ERROR(tensorflow::ConvertToDataType(type, &dtype));

  return shape_representation_fn(fully_defined_tensor_shape, dtype);
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
    return ::xla::ShapeUtil::MakeShapeWithLayout(primitive_type, dimensions,
                                                 minor_to_major);
  } else if (auto t = type.dyn_cast<mlir::RankedTensorType>()) {
    // TODO(jpienaar): This is only handling the base case with primitive
    // element type.
    llvm::SmallVector<int64_t, 4> span(t.getShape().begin(),
                                       t.getShape().end());
    // Only fully static shapes are supported.
    // TODO(b/115638799): Update once xla::Shape can support dynamic shapes.
    if (std::find(t.getShape().begin(), t.getShape().end(), -1) !=
        t.getShape().end())
      return {};
    mlir::Type element_type = t.getElementType();
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    // Only primitive element type supported.
    if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return ShapeUtil::MakeShape(primitive_type, span);
  } else if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    llvm::SmallVector<Shape, 4> shapes;
    shapes.reserve(tuple_type.size());
    for (mlir::Type sub_type : tuple_type.getTypes()) {
      shapes.push_back(TypeToShape(sub_type));
    }
    return ShapeUtil::MakeTupleShape(shapes);

  } else if (type.isa<mlir::mhlo::TokenType>()) {
    return ShapeUtil::MakeTokenShape();
  }

  // Return empty XLA shape to signify error. No MLIR Type maps to a empty
  // Shape.
  return {};
}

}  // namespace xla
