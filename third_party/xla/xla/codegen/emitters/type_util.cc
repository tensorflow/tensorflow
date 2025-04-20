/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/codegen/emitters/type_util.h"

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/layout_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace emitters {

mlir::Type PrimitiveTypeToMlirType(PrimitiveType type, mlir::OpBuilder& b) {
  if (primitive_util::IsIntegralType(type)) {
    return b.getIntegerType(primitive_util::BitWidth(type));
  }
  return PrimitiveTypeToMlirTypeWithSign(type, b);
}

mlir::Type PrimitiveTypeToMlirTypeWithSign(PrimitiveType type,
                                           mlir::OpBuilder& b) {
  if (type == PrimitiveType::PRED) {
    // We lower PRED to i8 for historical reasons. Yes, that means that there
    // are more than two PRED values. Yes, we have tests for that.
    return b.getI8Type();
  }
  return *ConvertPrimitiveTypeToMlirType(type, b);
}

mlir::Type TensorShapeToMlirType(const Shape& shape, mlir::OpBuilder& b) {
  CHECK(shape.IsArray());

  // Default layouts create a lot of clutter in the IR, so only add an
  // encoding when needed.
  mlir::Attribute layout = {};
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    layout = CreateDenseIntElementsAttrFromVector(
        llvm::to_vector(shape.layout().minor_to_major()), b);
  }
  return mlir::RankedTensorType::get(
      llvm::to_vector(shape.dimensions()),
      PrimitiveTypeToMlirType(shape.element_type(), b), layout);
}

llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::OpBuilder& b) {
  llvm::SmallVector<mlir::Type> types;
  types.reserve(shape.IsTuple() ? shape.tuple_shapes().size() : 1);
  if (shape.IsTuple()) {
    types.reserve(shape.tuple_shapes().size());
    for (auto& tuple_shape : shape.tuple_shapes()) {
      if (tuple_shape.IsTuple()) {
        types.append(ShapeToMlirTypes(tuple_shape, b));
      } else {
        types.push_back(TensorShapeToMlirType(tuple_shape, b));
      }
    }
  } else {
    types.push_back(TensorShapeToMlirType(shape, b));
  }
  return types;
}

}  // namespace emitters
}  // namespace xla
