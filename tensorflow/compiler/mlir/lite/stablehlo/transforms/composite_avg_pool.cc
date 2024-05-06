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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_avg_pool.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/composite_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/util/padding.h"

namespace mlir {
namespace odml {

DenseIntElementsAttr GetPaddingArrayAttr(Builder& builder, Operation* old_op) {
  mhlo::CompositeOp composite_op = llvm::dyn_cast<mhlo::CompositeOp>(old_op);
  auto composite_attrs = composite_op.getCompositeAttributes();
  std::vector<int32_t> padding_vec;
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "padding",
                                        &padding_vec);

  std::vector<int32_t> result_padding_conf(8, 0);  // NHWC
  result_padding_conf[2] = result_padding_conf[3] = padding_vec[0];
  result_padding_conf[4] = result_padding_conf[5] = padding_vec[1];

  return DenseIntElementsAttr::get(
      RankedTensorType::get({4, 2}, builder.getI32Type()), result_padding_conf);
}

ShapedType GetPaddedType(Operation* old_op) {
  auto input_type = mlir::cast<ShapedType>(old_op->getOperand(0).getType());
  auto input_shape = input_type.getShape();  // NCHW
  int64_t batch_size = input_shape[0];
  int64_t channel_size = input_shape[1];
  int64_t height = input_shape[2];
  int64_t width = input_shape[3];

  DenseIntElementsAttr padding_attr;
  mhlo::CompositeOp composite_op = llvm::dyn_cast<mhlo::CompositeOp>(old_op);
  auto composite_attributes = composite_op.getCompositeAttributes();
  EnsureAttribute<DenseIntElementsAttr>(composite_attributes, "padding",
                                        &padding_attr);
  std::vector<int64_t> padding_values(padding_attr.getValues<int64_t>().begin(),
                                      padding_attr.getValues<int64_t>().end());
  int64_t padding_height = padding_values[0];
  int64_t padding_width = padding_values[1];

  std::array<int64_t, 4> output_shape = {
      batch_size, height + 2 * padding_height, width + 2 * padding_width,
      channel_size};  // NHWC
  return RankedTensorType::get(output_shape, input_type.getElementType());
}

// Checks if the provided configuration can be supported by the tensorflow
// "SAME" padding configuration.
static bool IsSamePadding(const std::vector<int32_t>& spatial_dim_sizes,
                          const std::vector<int32_t>& kernel_size,
                          const std::vector<int32_t>& strides,
                          const std::vector<int32_t>& padding_array) {
  for (int dim : llvm::seq<int>(0, spatial_dim_sizes.size())) {
    int64_t discard;
    int64_t pad_low_ignore;
    int64_t pad_high_ignore;
    absl::Status status = tensorflow::GetWindowedOutputSizeVerbose(
        spatial_dim_sizes[dim], kernel_size[dim], 1, strides[dim],
        tensorflow::Padding::SAME, &discard, &pad_low_ignore, &pad_high_ignore);
    if (!status.ok()) {
      return false;
    }
    if (padding_array[dim] != pad_low_ignore ||
        padding_array[dim] != pad_high_ignore) {
      return false;
    }
  }

  return true;
}

enum class PaddingType { kValid, kSame, kCustom };

static PaddingType GetPaddingType(const std::vector<int32_t>& spatial_dim_sizes,
                                  const std::vector<int32_t>& kernel_size,
                                  const std::vector<int32_t>& strides,
                                  const std::vector<int32_t>& padding_array) {
  if (std::all_of(padding_array.begin(), padding_array.end(),
                  [](int32_t padding_value) { return padding_value == 0; })) {
    return PaddingType::kValid;
  }
  if (IsSamePadding(spatial_dim_sizes, kernel_size, strides, padding_array)) {
    return PaddingType::kSame;
  }
  return PaddingType::kCustom;
}

StringAttr GetPaddingStringAttr(Builder& builder, Operation* old_op) {
  mhlo::CompositeOp composite_op = llvm::dyn_cast<mhlo::CompositeOp>(old_op);
  auto composite_attrs = composite_op.getCompositeAttributes();

  auto operand_shape =
      mlir::cast<ShapedType>(composite_op.getOperand(0).getType()).getShape();
  // NC(H)(W)
  std::vector<int32_t> spatial_dim_sizes = {
      static_cast<int32_t>(operand_shape[2]),
      static_cast<int32_t>(operand_shape[3])};

  std::vector<int32_t> padding_vec, kernel_size_vec, strides_vec;
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "kernel_size",
                                        &kernel_size_vec);
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "stride",
                                        &strides_vec);
  GetI32VectorFromDenseI64CompositeAttr(composite_attrs, "padding",
                                        &padding_vec);
  PaddingType padding_type = GetPaddingType(spatial_dim_sizes, kernel_size_vec,
                                            strides_vec, padding_vec);

  switch (padding_type) {
    case PaddingType::kValid:
      return builder.getStringAttr("VALID");
    case PaddingType::kSame:
      return builder.getStringAttr("SAME");
    case PaddingType::kCustom:
      return builder.getStringAttr("CUSTOM");
  }
}

}  // namespace odml
}  // namespace mlir
