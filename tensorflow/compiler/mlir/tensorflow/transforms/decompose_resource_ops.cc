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

#include "tensorflow/compiler/mlir/tensorflow/transforms/decompose_resource_ops.h"

#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

namespace mlir {
namespace TF {

namespace {
// Returns int or float DenseElementsAttr with scalar shape with the given
// element type and the integer value.
static DenseElementsAttr GetScalarOfType(Type ty, int64_t raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = ty.dyn_cast_or_null<FloatType>()) {
    FloatAttr attr = FloatAttr::get(float_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  }

  auto int_ty = ty.cast<IntegerType>();
  IntegerAttr attr = IntegerAttr::get(int_ty, raw_value);
  return DenseElementsAttr::get(scalar_ty, attr);
}

// Returns subtype of `resource` if present. Otherwise an unranked tensor type
// of `element_type` is returned.
static Type GetResourceSubtypeOrDefault(Value resource, Type element_type) {
  auto resource_type = resource.getType()
                           .cast<TensorType>()
                           .getElementType()
                           .cast<ResourceType>();
  if (resource_type.getSubtypes().size() == 1)
    return resource_type.getSubtypes().front();

  return UnrankedTensorType::get(element_type);
}

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_decompose_resource_ops.inc"
}  // namespace

void PopulateDecomposeResourceOpsPatterns(MLIRContext *context,
                                          OwningRewritePatternList *patterns) {
  populateWithGenerated(context, patterns);
}

}  // namespace TF
}  // namespace mlir
