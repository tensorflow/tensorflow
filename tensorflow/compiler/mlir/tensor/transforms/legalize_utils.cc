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

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensor/transforms/legalize_utils.h"

namespace mlir {
namespace tensor {

Value castUnrankedTensor(OpBuilder& builder, Location loc, Value tensor,
                         int rank) {

  auto tensor_type = tensor.getType().cast<TensorType>();
  if (tensor_type.isa<RankedTensorType>())
    return tensor;

  SmallVector<int64_t> dims(rank, ShapedType::kDynamic);
  auto element_type = tensor_type.getElementType();
  auto ranked_shape_type = RankedTensorType::get(dims, element_type);
  return builder.create<tensor::CastOp>(loc, ranked_shape_type, tensor);
}

Value getTensorSize(OpBuilder& builder, Location loc, Value tensor) {

  auto rank = builder.create<tensor::RankOp>(loc, tensor);
  auto zero = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto one = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));

  auto createLoopBody = [&](OpBuilder& builder, Location loc, Value index, ValueRange iter_args) {
    auto acc_size = iter_args.front();
    auto dim_size = builder.create<tensor::DimOp>(loc, tensor, index);
    Value next_size = builder.create<arith::MulIOp>(loc, acc_size, dim_size);
    builder.create<scf::YieldOp>(loc, next_size);
  };
  auto for_op = builder.create<scf::ForOp>(loc, zero, rank, one,
                                            ValueRange(one), createLoopBody);
  return for_op.getResult(0);
}

Value multiplyTensorElements(OpBuilder& builder, Location loc, Value tensor) {
  auto tensor_type = tensor.getType().cast<TensorType>();
  auto element_type = tensor_type.getElementType().cast<IntegerType>();

  auto one_tensor_type = RankedTensorType::get({}, element_type);
  Attribute one_attr = builder.getIntegerAttr(element_type, 1);
  auto one_tensor_attr = DenseIntElementsAttr::get(one_tensor_type, {one_attr});
  Value one = builder.create<arith::ConstantOp>(loc, one_tensor_attr);

  auto reduce_op_body = [&](OpBuilder& builder, Location loc, ValueRange operands) {
    Value temp_product = builder.create<arith::MulIOp>(loc, operands);
    builder.create<linalg::YieldOp>(loc, temp_product);
  };

  auto product_tensor = builder.create<linalg::ReduceOp>(
      loc, tensor, one, std::initializer_list<int64_t>{0}, reduce_op_body)
      .getResult(0);

  return builder.create<tensor::ExtractOp>(loc, product_tensor);
}

Value broadcastScalar(OpBuilder& builder, Location loc, Value input,
                      OpFoldResult size, Type result_type) {
  auto input_scalar_type = input.getType();
  assert(!input_scalar_type.isa<ShapedType>());
  if (size.is<Attribute>()) {
    auto size_attr = size.get<Attribute>().cast<IntegerAttr>();
    assert(size_attr.getType().isa<IndexType>());

    // Create a splat tensor of type 'tensor<SIZE x INPUT_TYPE>'. The
    // 'tensor.splat' op expects a static result shape.
    auto temp_result_type = RankedTensorType::get({size_attr.getInt()}, input_scalar_type);
    auto splat = builder.create<tensor::SplatOp>(loc, temp_result_type, input);

    // Cast splat into the requested result type
    return builder.create<tensor::CastOp>(loc, result_type, splat);

  } else {

    auto size_value = size.get<Value>();
    assert(size_value.getType().isa<IndexType>());

    auto input_tensor_type = RankedTensorType::get({}, input_scalar_type);
    auto input_tensor = builder.create<tensor::FromElementsOp>(loc, input_tensor_type, input);

    auto init_tensor_type = RankedTensorType::get({ShapedType::kDynamic}, input_scalar_type);
    auto init_tensor = builder.create<tensor::EmptyOp>(loc, init_tensor_type, size_value);

    auto result = builder.create<linalg::BroadcastOp>(
        loc, input_tensor, init_tensor, std::initializer_list<int64_t>{0})
        .getResult();
    return builder.create<tensor::CastOp>(loc, result_type, result);
  }
}

Value substituteShapeWildcard(OpBuilder& builder, Location loc, Value input,
                              Value shape) {
  auto shape_type = shape.getType().cast<TensorType>();
  auto shape_element_type = shape_type.getElementType();
 
  // Calculate product of shape tensor elements and check if there was a
  // dimension set to -1. We check this condition by looking at the sign of
  // the product. If the product was negative due to any other reason than
  // exactly one dimension being set to -1, the behavior is undefined.
  auto shape_product = multiplyTensorElements(builder, loc, shape);
  auto zero_attr = builder.getIntegerAttr(shape_element_type, 0);
  auto zero = builder.create<arith::ConstantOp>(loc, zero_attr);
  auto condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, shape_product, zero);

  // Control flow
  auto if_op = builder.create<scf::IfOp>(loc, condition, [&](OpBuilder& builder, Location loc) {

    // Calculate size of input tensor and cast it to same type as shape tensor
    auto input_size = getTensorSize(builder, loc, input);
    input_size = builder.create<arith::IndexCastOp>(loc, shape_element_type, input_size);

    auto abs_product = builder.create<math::AbsIOp>(loc, shape_product);

    auto shape_size = linalg::createFoldedDimOp(builder, loc, shape, 0);

    auto wildcard_dim_size = builder.create<arith::DivSIOp>(loc, input_size, abs_product);
    auto wildcard_dim_size_splat = broadcastScalar(builder, loc, wildcard_dim_size, shape_size, shape.getType());

    auto minus_one_attr = builder.getIntegerAttr(shape_element_type, -1);
    auto minus_one = builder.create<arith::ConstantOp>(loc, minus_one_attr);
    auto minus_one_splat = broadcastScalar(builder, loc, minus_one, shape_size, shape.getType());

    auto condition = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, shape, minus_one_splat);
    Value resolved_shape = builder.create<arith::SelectOp>(loc, condition, wildcard_dim_size_splat, shape);
    builder.create<scf::YieldOp>(loc, resolved_shape);
  }, [&](OpBuilder& builder, Location loc) {
    builder.create<scf::YieldOp>(loc, shape);
  });
  return if_op.getResult(0);
}

}  // namespace tensor
}  // namespace mlir


