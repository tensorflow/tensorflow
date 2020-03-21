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

#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace mlir {
namespace TF {
namespace collection_ops_util {

Value CreateScalarConst(int value, OpBuilder builder, Location loc) {
  tensorflow::Tensor scalar_tensor(tensorflow::DT_INT32, {});
  scalar_tensor.scalar<tensorflow::int32>()() = value;
  return builder.create<TF::ConstOp>(
      loc, tensorflow::ConvertTensor(scalar_tensor, &builder).ValueOrDie());
}

Value GetR1Const(ArrayRef<int64_t> r1, OpBuilder builder, Location loc) {
  tensorflow::Tensor shape_tensor(tensorflow::DT_INT32,
                                  {static_cast<int64_t>(r1.size())});
  for (int i = 0; i < r1.size(); ++i) {
    shape_tensor.vec<tensorflow::int32>()(i) = r1[i];
  }
  return builder.create<TF::ConstOp>(
      loc, tensorflow::ConvertTensor(shape_tensor, &builder).ValueOrDie());
}

Value GetIndicesForElement(Value index, Value buffer, OpBuilder builder,
                           Location loc) {
  auto buffer_type = buffer.getType().cast<RankedTensorType>();
  if (buffer_type.getShape().size() == 1) return index;
  // Create a concat of index and trailing zeros.
  llvm::SmallVector<int64_t, 8> zeros(buffer_type.getShape().size() - 1, 0);
  auto zeros_tensor = GetR1Const(zeros, builder, loc);
  return builder.create<TF::ConcatV2Op>(
      loc,
      ArrayRef<Type>{RankedTensorType::get(
          {static_cast<int64_t>(buffer_type.getShape().size())},
          getElementTypeOrSelf(index.getType()))},
      ArrayRef<Value>{index, zeros_tensor, CreateScalarConst(0, builder, loc)},
      ArrayRef<NamedAttribute>{});
}

Value GetElement(Value index, Value buffer, OpBuilder builder, Location loc) {
  auto buffer_type = buffer.getType().cast<RankedTensorType>();
  // Create a slice then reshape to remove the leading trivial dimension of
  // size 1.
  llvm::SmallVector<int64_t, 8> slice_size =
      llvm::to_vector<8>(buffer_type.getShape());
  slice_size[0] = 1;
  auto size_const = GetR1Const(slice_size, builder, loc);
  auto slice_type =
      RankedTensorType::get(slice_size, buffer_type.getElementType());
  auto slice = builder.create<TF::SliceOp>(
      loc, ArrayRef<Type>{slice_type},
      ArrayRef<Value>{buffer, GetIndicesForElement(index, buffer, builder, loc),
                      size_const},
      ArrayRef<NamedAttribute>{});
  auto element_type = RankedTensorType::get(buffer_type.getShape().drop_front(),
                                            buffer_type.getElementType());
  auto reshape = builder.create<TF::ReshapeOp>(
      loc, ArrayRef<Type>{element_type},
      ArrayRef<Value>{slice, GetR1Const(element_type.getShape(), builder, loc)},
      ArrayRef<NamedAttribute>{});
  return reshape.output();
}

Value SetElement(Value index, Value buffer, Value element, OpBuilder builder,
                 Location loc) {
  auto buffer_type = buffer.getType().cast<RankedTensorType>();
  // Reshape the element to add a leading dimension of size 1, then perform a
  // dynamic update slice.
  auto slice_shape = llvm::to_vector<8>(buffer_type.getShape());
  slice_shape[0] = 1;
  auto update_slice = builder.create<TF::ReshapeOp>(
      loc,
      ArrayRef<Type>{
          RankedTensorType::get(slice_shape, buffer_type.getElementType())},
      ArrayRef<Value>{element, GetR1Const(slice_shape, builder, loc)},
      ArrayRef<NamedAttribute>{});
  return builder
      .create<TF::XlaDynamicUpdateSliceOp>(
          loc, ArrayRef<Type>{buffer.getType()},
          ArrayRef<Value>{buffer, update_slice,
                          GetIndicesForElement(index, buffer, builder, loc)},
          ArrayRef<NamedAttribute>{})
      .output();
}

TensorType GetSizeType(OpBuilder builder) {
  return RankedTensorType::get({1}, builder.getIntegerType(32));
}

Value ReshapeScalarToSizeType(OpBuilder builder, Value scalar, Location loc) {
  auto size_type = GetSizeType(builder);
  return builder.create<TF::ReshapeOp>(
      loc, ArrayRef<Type>{size_type},
      ArrayRef<Value>{scalar, GetR1Const(size_type.getShape(), builder, loc)},
      ArrayRef<NamedAttribute>{});
}

LogicalResult CreateInitBufferValue(ArrayRef<int64_t> element_shape,
                                    Value max_size, Operation* op,
                                    Type element_dtype, OpBuilder builder,
                                    Value* buffer) {
  auto max_count_op = max_size.getDefiningOp();
  if (!max_count_op) return op->emitOpError("unknown max element count");
  auto max_count_const_op = llvm::dyn_cast<TF::ConstOp>(max_count_op);
  if (!max_count_const_op) return op->emitOpError("unknown max element count");
  int64_t max_size_const =
      (*max_count_const_op.value().getValues<APInt>().begin()).getSExtValue();
  llvm::SmallVector<int64_t, 8> buffer_shape;
  buffer_shape.push_back(max_size_const);
  for (int64_t dim : element_shape) {
    buffer_shape.push_back(dim);
  }
  auto zero = CreateScalarConst(0, builder, op->getLoc());
  if (getElementTypeOrSelf(zero.getType()) != element_dtype) {
    zero = builder.create<TF::CastOp>(
        op->getLoc(), ArrayRef<Type>{RankedTensorType::get({}, element_dtype)},
        ArrayRef<Value>{zero}, ArrayRef<NamedAttribute>{});
  }
  auto buffer_type = RankedTensorType::get(buffer_shape, element_dtype);
  auto broadcast = builder.create<TF::BroadcastToOp>(
      op->getLoc(), ArrayRef<Type>{buffer_type},
      ArrayRef<Value>{zero, GetR1Const(buffer_shape, builder, op->getLoc())},
      ArrayRef<NamedAttribute>{});
  *buffer = broadcast.output();
  return success();
}
}  // namespace collection_ops_util
}  // namespace TF
}  // namespace mlir
