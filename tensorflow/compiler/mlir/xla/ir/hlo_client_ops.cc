/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/ir/hlo_client_ops.h"

#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project

namespace mlir {
namespace xla_hlo_client {

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
static Type GetBroadcastType(Builder* builder, Type x, Type y,
                             Type element_type,
                             DenseIntElementsAttr broadcast_dimensions) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  auto y_ranked = y.dyn_cast<RankedTensorType>();
  if (!x_ranked || !y_ranked) {
    return UnrankedTensorType::get(element_type);
  }

  auto shape_x = x_ranked.getShape();
  auto shape_y = y_ranked.getShape();

  if (shape_x.size() == shape_y.size()) {
    llvm::SmallVector<int64_t, 4> out_shape(shape_x.size());
    for (int i = 0; i < shape_x.size(); i++) {
      auto x_val = shape_x[i];
      auto y_val = shape_y[i];
      if (x_val == -1 || y_val == -1) {
        out_shape[i] = -1;
      } else {
        out_shape[i] = std::max(x_val, y_val);
      }
    }
    return RankedTensorType::get(out_shape, element_type);
  }

  // Return unranked tensor for invalid broadcast dimensions.
  if (!broadcast_dimensions) return UnrankedTensorType::get(element_type);

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
                                          shape_large.end());

  // Update according to the broadcast dimensions.
  for (auto index_pair : llvm::enumerate(broadcast_dimensions.getIntValues())) {
    auto old_value = out_shape[index_pair.value().getSExtValue()];
    auto new_value = shape_small[index_pair.index()];
    if (old_value != -1 && (new_value == -1 || new_value > old_value)) {
      out_shape[index_pair.value().getSExtValue()] = new_value;
    }
  }

  return RankedTensorType::get(out_shape, element_type);
}
}  // namespace

#define BINARY_BUILDER(Op)                                                   \
  void Op::build(Builder* builder, OperationState& result, Value left,       \
                 Value right, DenseIntElementsAttr broadcast_dimensions) {   \
    auto type = GetBroadcastType(builder, left.getType().cast<ShapedType>(), \
                                 right.getType().cast<ShapedType>(),         \
                                 getElementTypeOrSelf(right.getType()),      \
                                 broadcast_dimensions);                      \
    return Op::build(builder, result, type, left, right,                     \
                     broadcast_dimensions);                                  \
  }

BINARY_BUILDER(AddOp);
BINARY_BUILDER(AndOp);
BINARY_BUILDER(Atan2Op);
BINARY_BUILDER(DivOp);
BINARY_BUILDER(MaxOp);
BINARY_BUILDER(MinOp);
BINARY_BUILDER(MulOp);
BINARY_BUILDER(OrOp);
BINARY_BUILDER(PowOp);
BINARY_BUILDER(RemOp);
BINARY_BUILDER(ShiftLeftOp);
BINARY_BUILDER(ShiftRightArithmeticOp);
BINARY_BUILDER(ShiftRightLogicalOp);
BINARY_BUILDER(SubOp);
BINARY_BUILDER(XorOp);

#undef BINARY_BUILDER

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/xla/ir/hlo_client_ops.cc.inc"

//===----------------------------------------------------------------------===//
// xla_hlo_client Dialect Constructor
//===----------------------------------------------------------------------===//

XlaHloClientDialect::XlaHloClientDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/xla/ir/hlo_client_ops.cc.inc"
      >();
}

}  // namespace xla_hlo_client
}  // namespace mlir
