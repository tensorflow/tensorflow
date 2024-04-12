/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace mlir {
namespace odml {

DenseIntElementsAttr DenseI64AttrToI32Attr(
    const DenseIntElementsAttr& dense_attr, PatternRewriter& builder) {
  std::vector<int32_t> ret(dense_attr.getNumElements());
  auto range = dense_attr.getValues<int64_t>();
  std::transform(range.begin(), range.end(), ret.begin(),
                 [](int64_t attr) { return static_cast<int32_t>(attr); });
  return DenseIntElementsAttr::get(
      RankedTensorType::get(ret.size(), builder.getIntegerType(32)), ret);
}

bool DenseI64AttrToI32Vector(const DenseIntElementsAttr& dense_attr,
                             std::vector<int32_t>* out_vec) {
  std::vector<int32_t> ret(dense_attr.getNumElements());
  auto range = dense_attr.getValues<int64_t>();
  std::transform(range.begin(), range.end(), ret.begin(),
                 [](int64_t attr) { return static_cast<int32_t>(attr); });
  *out_vec = std::move(ret);
  return true;
}

bool GetI32VectorFromDenseI64CompositeAttr(
    const DictionaryAttr& composite_attrs, const std::string& attr_name,
    std::vector<int32_t>* out_vec) {
  DenseIntElementsAttr attr;
  if (!EnsureAttribute<DenseIntElementsAttr>(composite_attrs, attr_name,
                                             &attr)) {
    return false;
  }

  return DenseI64AttrToI32Vector(attr, out_vec);
}

bool IsSupportedNchwUpsampleBlinear(
    Value input, Value output, const DenseIntElementsAttr& output_size_attr) {
  auto input_shape = input.getType().cast<ShapedType>().getShape();
  auto output_shape = output.getType().cast<ShapedType>().getShape();

  // Only support 4D tensor.
  if (input_shape.size() != 4 || output_shape.size() != 4) {
    return false;
  }

  // Only expects the first two dimensions of input and output to be the same as
  // in NCHW.
  if (input_shape[0] != output_shape[0] || input_shape[1] != output_shape[1]) {
    return false;
  }

  // Supplied output size should be 2D.
  if (output_size_attr.getNumElements() != 2) {
    return false;
  }
  auto output_size = output_size_attr.getValues<int64_t>();
  return output_size[0] == output_shape[2] && output_size[1] == output_shape[3];
}

ShapedType GetNhwcReturnTypeFromNchw(Operation* old_op) {
  auto composite_result_shape =
      old_op->getResults().front().getType().cast<ShapedType>().getShape();
  std::array<int64_t, 4> output_shape;
  // NHWC <- NCHW
  output_shape[0] = composite_result_shape[0];
  output_shape[1] = composite_result_shape[2];
  output_shape[2] = composite_result_shape[3];
  output_shape[3] = composite_result_shape[1];

  auto input_type = old_op->getOperand(0).getType().cast<ShapedType>();

  return RankedTensorType::get(output_shape, input_type.getElementType());
}
}  // namespace odml
}  // namespace mlir
