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
#include "xla/service/gpu/fusions/mlir/type_util.h"

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "xla/layout_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/shape.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"

namespace xla {
namespace gpu {
namespace mlir_converter {

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
      *ConvertPrimitiveTypeToMlirType(shape.element_type(), b), layout);
}

llvm::SmallVector<mlir::Type> ShapeToMlirTypes(const Shape& shape,
                                               mlir::OpBuilder& b) {
  llvm::SmallVector<mlir::Type> types;
  types.reserve(shape.IsTuple() ? shape.tuple_shapes_size() : 1);
  if (shape.IsTuple()) {
    types.reserve(shape.tuple_shapes_size());
    for (auto& tuple_shape : shape.tuple_shapes()) {
      types.push_back(TensorShapeToMlirType(tuple_shape, b));
    }
  } else {
    types.push_back(TensorShapeToMlirType(shape, b));
  }
  return types;
}

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
