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

#include <cassert>
#include <cstdint>
#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"

namespace mlir {
namespace TF {
namespace collection_ops_util {

Value CreateScalarConst(int32_t value, OpBuilder builder, Location loc) {
  auto attr = DenseIntElementsAttr::get(
      tensorflow::GetTypeFromTFTensorShape({}, builder.getI32Type()), value);
  return TF::ConstOp::create(builder, loc, attr);
}

Value GetR1Const(ArrayRef<int64_t> r1, OpBuilder builder, Location loc,
                 int bitwidth) {
  llvm::SmallVector<APInt, 4> values;
  int64_t rank = r1.size();
  values.reserve(rank);
  for (int i = 0; i < rank; ++i) values.push_back(APInt(bitwidth, r1[i]));
  auto result_type = tensorflow::GetTypeFromTFTensorShape(
      {rank}, IntegerType::get(builder.getContext(), bitwidth));
  return TF::ConstOp::create(builder, loc,
                             DenseElementsAttr::get(result_type, values));
}

Value GetIndicesForElement(Value index, Value buffer, OpBuilder builder,
                           Location loc) {
  auto buffer_type = mlir::cast<RankedTensorType>(buffer.getType());
  if (buffer_type.getShape().size() == 1) return index;
  // Create a concat of index and trailing zeros.
  llvm::SmallVector<int64_t, 8> zeros(buffer_type.getShape().size() - 1, 0);
  auto zeros_tensor = GetR1Const(zeros, builder, loc);
  return TF::ConcatV2Op::create(
      builder, loc,
      ArrayRef<Type>{tensorflow::GetTypeFromTFTensorShape(
          {static_cast<int64_t>(buffer_type.getShape().size())},
          getElementTypeOrSelf(index.getType()))},
      ArrayRef<Value>{index, zeros_tensor, CreateScalarConst(0, builder, loc)});
}

Value GetElement(Value index, Value buffer, OpBuilder builder, Location loc,
                 bool keep_slice_shape) {
  auto buffer_type = mlir::cast<RankedTensorType>(buffer.getType());
  // Create a slice then reshape to remove the leading trivial dimension of
  // size 1.
  llvm::SmallVector<int64_t, 8> slice_size =
      llvm::to_vector<8>(buffer_type.getShape());
  slice_size[0] = 1;
  auto size_const = GetR1Const(slice_size, builder, loc);
  auto slice_type = tensorflow::GetTypeFromTFTensorShape(
      slice_size, buffer_type.getElementType());
  auto slice = TF::SliceOp::create(
      builder, loc, ArrayRef<Type>{slice_type},
      ArrayRef<Value>{buffer, GetIndicesForElement(index, buffer, builder, loc),
                      size_const});
  if (keep_slice_shape) return slice;
  auto element_type = tensorflow::GetTypeFromTFTensorShape(
      buffer_type.getShape().drop_front(), buffer_type.getElementType());
  auto reshape = TF::ReshapeOp::create(
      builder, loc, ArrayRef<Type>{element_type},
      ArrayRef<Value>{slice,
                      GetR1Const(element_type.getShape(), builder, loc)});
  return reshape.getOutput();
}

Value SetElement(Value index, Value buffer, Value element, OpBuilder builder,
                 Location loc) {
  auto buffer_type = mlir::cast<RankedTensorType>(buffer.getType());
  // Reshape the element to add a leading dimension of size 1 if th element does
  // not have that dimension, then perform a dynamic update slice.
  auto slice_shape = llvm::to_vector<8>(buffer_type.getShape());
  slice_shape[0] = 1;
  auto slice_type = tensorflow::GetTypeFromTFTensorShape(
      slice_shape, buffer_type.getElementType());
  auto update_slice = element;
  if (element.getType() != slice_type) {
    update_slice = TF::ReshapeOp::create(
        builder, loc, ArrayRef<Type>{slice_type},
        ArrayRef<Value>{element, GetR1Const(slice_shape, builder, loc)});
  }
  return TF::XlaDynamicUpdateSliceOp::create(
             builder, loc, ArrayRef<Type>{buffer.getType()},
             ArrayRef<Value>{buffer, update_slice,
                             GetIndicesForElement(index, buffer, builder, loc)})
      .getOutput();
}

TensorType GetSizeType(OpBuilder builder) {
  return tensorflow::GetTypeFromTFTensorShape({1}, builder.getIntegerType(32));
}

Value ReshapeScalarToSizeType(OpBuilder builder, Value scalar, Location loc) {
  auto size_type = GetSizeType(builder);
  return TF::ReshapeOp::create(
      builder, loc, ArrayRef<Type>{size_type},
      ArrayRef<Value>{scalar, GetR1Const(size_type.getShape(), builder, loc)});
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
      (*max_count_const_op.getValue().getValues<APInt>().begin())
          .getSExtValue();
  return CreateInitBufferValue(element_shape, max_size_const, op, element_dtype,
                               builder, buffer);
}

LogicalResult CreateInitBufferValue(ArrayRef<int64_t> element_shape,
                                    int64_t max_size, Operation* op,
                                    Type element_dtype, OpBuilder builder,
                                    Value* buffer) {
  llvm::SmallVector<int64_t, 8> buffer_shape;
  buffer_shape.push_back(max_size);
  for (int64_t dim : element_shape) {
    buffer_shape.push_back(dim);
  }
  auto zero = CreateScalarConst(0, builder, op->getLoc());
  if (getElementTypeOrSelf(zero.getType()) != element_dtype) {
    zero = TF::CastOp::create(
        builder, op->getLoc(),
        ArrayRef<Type>{tensorflow::GetTypeFromTFTensorShape({}, element_dtype)},
        ArrayRef<Value>{zero});
  }
  auto buffer_type =
      tensorflow::GetTypeFromTFTensorShape(buffer_shape, element_dtype);
  auto broadcast = TF::BroadcastToOp::create(
      builder, op->getLoc(), ArrayRef<Type>{buffer_type},
      ArrayRef<Value>{zero, GetR1Const(buffer_shape, builder, op->getLoc())});
  *buffer = broadcast.getOutput();
  return success();
}

std::optional<RankedTensorType> GetElementTypeFromAccess(
    Value collection, ModuleOp module,
    llvm::function_ref<std::optional<Type>(Operation*)> infer_from_op) {
  for (auto& use : collection.getUses()) {
    if (auto while_op = llvm::dyn_cast<TF::WhileOp>(use.getOwner())) {
      auto body = while_op.body_function();
      assert(body);
      auto type_from_body = GetElementTypeFromAccess(
          body.getArgument(use.getOperandNumber()), module, infer_from_op);
      if (type_from_body.has_value()) return type_from_body;
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(use.getOwner())) {
      auto then_branch = if_op.then_function();
      auto else_branch = if_op.else_function();
      assert(then_branch && else_branch);
      auto type_from_then = GetElementTypeFromAccess(
          then_branch.getArgument(use.getOperandNumber() - 1), module,
          infer_from_op);
      if (type_from_then.has_value()) return type_from_then;
      auto type_from_else = GetElementTypeFromAccess(
          else_branch.getArgument(use.getOperandNumber() - 1), module,
          infer_from_op);
      if (type_from_else.has_value()) return type_from_else;
    } else if (auto call = llvm::dyn_cast<CallOpInterface>(use.getOwner())) {
      auto callee = dyn_cast<func::FuncOp>(call.resolveCallable());
      auto type_from_callee = GetElementTypeFromAccess(
          callee.getArgument(use.getOperandNumber()), module, infer_from_op);
      if (type_from_callee.has_value()) return type_from_callee;
    } else if (llvm::isa<TF::IdentityOp, TF::IdentityNOp>(use.getOwner())) {
      auto type_from_alias = GetElementTypeFromAccess(
          use.getOwner()->getResult(use.getOperandNumber()), module,
          infer_from_op);
      if (type_from_alias.has_value()) return type_from_alias;
    } else if (auto type = infer_from_op(use.getOwner())) {
      if (!type) continue;
      auto elem_type = mlir::dyn_cast<RankedTensorType>(*type);
      if (elem_type && elem_type.hasStaticShape()) return elem_type;
    }
  }
  return std::nullopt;
}

// Creates a ReadVariableOp on a local variable.
Value ReadLocalVariable(Value local_var, OpBuilder builder, Location loc) {
  return TF::ReadVariableOp::create(
             builder, loc,
             ArrayRef<Type>{mlir::cast<TF::ResourceType>(
                                getElementTypeOrSelf(local_var.getType()))
                                .getSubtypes()[0]},
             ArrayRef<Value>{local_var})
      .getValue();
}

// Creates an AssignVariableOp on a local variable.
TF::AssignVariableOp WriteLocalVariable(Value local_var, Value value,
                                        OpBuilder builder, Location loc) {
  return TF::AssignVariableOp::create(builder, loc, ArrayRef<Type>{},
                                      ArrayRef<Value>{local_var, value});
}

Value AccumulateBuffers(Value a, Value b, OpBuilder builder, Location loc) {
  if (getElementTypeOrSelf(a.getType()) == builder.getI1Type()) {
    return TF::LogicalOrOp::create(builder, loc, ArrayRef<Type>{a.getType()},
                                   ArrayRef<Value>{a, b});
  }
  return TF::AddV2Op::create(builder, loc, ArrayRef<Type>{a.getType()},
                             ArrayRef<Value>{a, b});
}

namespace {

int64_t GetFirstIfIndicesAreContiguous(Value indices) {
  auto type = mlir::dyn_cast<RankedTensorType>(indices.getType());
  if (!type) return -1;
  auto indices_op = indices.getDefiningOp();
  if (!indices_op) return -1;
  auto const_op = llvm::dyn_cast<TF::ConstOp>(indices_op);
  if (!const_op) return -1;
  int64_t last_index = -1;
  int64_t first_index = -1;
  for (const auto& ind : const_op.getValue().getValues<APInt>()) {
    if (last_index == -1) {
      last_index = ind.getSExtValue();
      first_index = last_index;
      continue;
    }
    if (last_index + 1 != ind.getSExtValue()) return -1;
    last_index++;
  }
  return first_index;
}

}  // namespace

Value GatherElements(Value indices, Value buffer, OpBuilder builder,
                     Location loc) {
  auto buffer_type = mlir::cast<RankedTensorType>(buffer.getType());
  auto result_shape = llvm::to_vector<8>(buffer_type.getShape());
  result_shape[0] =
      mlir::cast<RankedTensorType>(indices.getType()).getDimSize(0);
  int64_t maybe_contiguous_start = GetFirstIfIndicesAreContiguous(indices);
  if (maybe_contiguous_start >= 0) {
    llvm::SmallVector<int64_t, 8> slice_starts(result_shape.size(), 0);
    slice_starts[0] = maybe_contiguous_start;
    auto slice_type = tensorflow::GetTypeFromTFTensorShape(
        result_shape, buffer_type.getElementType());
    return TF::SliceOp::create(
        builder, loc, ArrayRef<Type>{slice_type},
        ArrayRef<Value>{buffer, GetR1Const(slice_starts, builder, loc),
                        GetR1Const(result_shape, builder, loc)});
  }
  auto result_type = tensorflow::GetTypeFromTFTensorShape(
      result_shape, buffer_type.getElementType());
  return TF::GatherV2Op::create(
      builder, loc, ArrayRef<Type>{result_type},
      ArrayRef<Value>{buffer, indices, CreateScalarConst(0, builder, loc)});
}

Value ScatterAccumulateElements(Value indices, Value updates, Value buffer,
                                OpBuilder builder, Location loc) {
  auto buffer_type = mlir::cast<RankedTensorType>(buffer.getType());
  auto updates_type = mlir::cast<RankedTensorType>(updates.getType());
  int64_t maybe_contiguous_start = GetFirstIfIndicesAreContiguous(indices);
  if (maybe_contiguous_start == 0 && buffer_type == updates_type) {
    return AccumulateBuffers(buffer, updates, builder, loc);
  }
  // We cannot simply use a TensorScatterUpdate, as it does not accumulate with
  // the old data; it is tricky to manually add the old data either, since there
  // could be duplicates in the index. We follow the old bridge's approach by
  // iterating through the indices.
  auto per_slice_shape = llvm::to_vector<8>(buffer_type.getShape());
  per_slice_shape[0] = 1;
  auto slice_sizes = GetR1Const(per_slice_shape, builder, loc);
  llvm::SmallVector<int64_t, 8> starts_in_update(buffer_type.getRank(), 0);
  for (int64_t i = 0; i < updates_type.getDimSize(0); ++i) {
    auto index = TF::SliceOp::create(
        builder, loc, ArrayRef<Type>{GetSizeType(builder)},
        ArrayRef<Value>{indices, GetR1Const({i}, builder, loc),
                        GetR1Const({1}, builder, loc)});
    auto old_slice =
        GetElement(index, buffer, builder, loc, /*keep_slice_shape=*/true);
    starts_in_update[0] = i;
    auto update_slice_starts = GetR1Const(starts_in_update, builder, loc);
    Value slice =
        TF::SliceOp::create(
            builder, loc, ArrayRef<Type>{old_slice.getType()},
            ArrayRef<Value>{updates, update_slice_starts, slice_sizes})
            .getOutput();
    slice = AccumulateBuffers(old_slice, slice, builder, loc);
    buffer = SetElement(index, buffer, slice, builder, loc);
  }
  return buffer;
}

}  // namespace collection_ops_util
}  // namespace TF
}  // namespace mlir
