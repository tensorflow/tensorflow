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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace mlir {
namespace {

namespace cutil = TF::collection_ops_util;

using std::string;

#define GEN_PASS_DEF_TENSORARRAYOPSDECOMPOSITIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// A pass that converts tensor array operations to tensor operations and
// read/assign ops on local variables. A later resource lifting pass can further
// remove the local variables.
//
// This pass requires that the full shape of the tensor array can be inferred:
// 1) the size needs to be a constant, 2) it specifies the full element shape,
// or that can be inferred from a later write, and 3) all elements have the same
// shape.
//
struct TensorArrayOpsDecompositionPass
    : public impl::TensorArrayOpsDecompositionPassBase<
          TensorArrayOpsDecompositionPass> {
  void runOnOperation() override;
};

// Infers the element type and count for a TensorArraySplitV3Op. Requires
// constant lengths and static shape on the input value.
LogicalResult GetSplitElementTypeAndCount(TF::TensorArraySplitV3Op split,
                                          RankedTensorType* elem_type,
                                          int64_t* count) {
  auto lengths_const =
      llvm::dyn_cast_or_null<TF::ConstOp>(split.getLengths().getDefiningOp());
  if (!lengths_const) return split.emitOpError("non-constant split lengths");
  *count = lengths_const.getValue().getNumElements();
  if (*count <= 0) return split.emitOpError("non-positive split count");
  auto buffer_type = split.getValue().getType().dyn_cast<RankedTensorType>();
  if (!buffer_type || !buffer_type.hasStaticShape() ||
      buffer_type.getRank() < 1) {
    return split.emitOpError("unknown or invalid split tensor shape");
  }
  int64_t length = buffer_type.getDimSize(0) / *count;
  for (const auto& len : lengths_const.getValue().getValues<APInt>()) {
    if (length == len.getSExtValue()) continue;
    return split.emitOpError("different split lengths are not supported");
  }
  llvm::SmallVector<int64_t, 8> elem_shape;
  elem_shape.push_back(length);
  for (int64_t dim : buffer_type.getShape().drop_front()) {
    elem_shape.push_back(dim);
  }
  *elem_type = RankedTensorType::get(elem_shape, buffer_type.getElementType());
  return success();
}

// Tries to infer the tensor array element shape.
std::optional<llvm::SmallVector<int64_t, 8>> GetTensorArrayElementShape(
    TF::TensorArrayV3Op ta, ModuleOp module) {
  auto element_shape = ta.getElementShapeAttr().cast<mlir::TF::ShapeAttr>();
  if (element_shape.hasStaticShape()) {
    auto shape = element_shape.getShape();
    // Convert int64 to int64_t.
    llvm::SmallVector<int64_t, 8> dims(shape.begin(), shape.end());
    return dims;
  }

  bool has_failure = false;
  auto elem_type = cutil::GetElementTypeFromAccess(
      ta.getHandle(), module, [&](Operation* user) -> std::optional<Type> {
        if (has_failure) return std::nullopt;
        if (auto write = llvm::dyn_cast<TF::TensorArrayWriteV3Op>(user)) {
          return write.getValue().getType();
        } else if (auto split =
                       llvm::dyn_cast<TF::TensorArraySplitV3Op>(user)) {
          if (!split.getLengths().getDefiningOp() ||
              !llvm::isa<TF::ConstOp>(split.getLengths().getDefiningOp())) {
            return std::nullopt;
          }
          RankedTensorType t;
          int64_t count;
          if (failed(GetSplitElementTypeAndCount(split, &t, &count))) {
            has_failure = true;
            return std::nullopt;
          }
          return t;
        } else if (auto scatter =
                       llvm::dyn_cast<TF::TensorArrayScatterV3Op>(user)) {
          // TensorArrayScatter writes vector of tensors to TensorArray. We can
          // deduce the shape of TensorArray by dropping the 0th dim of
          // TensorArrayScatter `value`.
          auto t = scatter.getValue().getType().dyn_cast<RankedTensorType>();
          if (!t || t.getShape().empty()) return std::nullopt;
          return RankedTensorType::get(t.getShape().drop_front(),
                                       t.getElementType());
        } else if (auto gather =
                       llvm::dyn_cast<TF::TensorArrayGatherV3Op>(user)) {
          // Try to infer from result type of gather.
          auto t = gather.getValue().getType().dyn_cast<RankedTensorType>();
          if (t && !t.getShape().empty())
            return RankedTensorType::get(t.getShape().drop_front(),
                                         t.getElementType());
          // Try to infer from `element_shape` attribute of gather.
          auto element_shape = gather.getElementShapeAttr()
                                   .dyn_cast_or_null<mlir::TF::ShapeAttr>();
          if (element_shape && element_shape.hasStaticShape()) {
            return RankedTensorType::get(element_shape.getShape(),
                                         gather.getDtype());
          }
        }
        return std::nullopt;
      });
  if (!elem_type) return std::nullopt;
  return llvm::to_vector<8>(elem_type->getShape());
}

void ReplaceAllUsesWithCast(Value old_val, Value new_val) {
  if (old_val.use_empty()) return;
  auto cast_op =
      OpBuilder(old_val.getDefiningOp())
          .create<tensor::CastOp>(old_val.getLoc(), old_val.getType(), new_val);
  old_val.replaceAllUsesWith(cast_op);
}

void ReplaceAllUsesExceptTerminator(Value old_val, Value new_val) {
  if (old_val.getType() == new_val.getType()) {
    old_val.replaceAllUsesWith(new_val);
    return;
  }
  Operation* old_op = old_val.getDefiningOp();
  Operation* terminator_op =
      old_op->getParentOfType<func::FuncOp>().front().getTerminator();
  llvm::SmallPtrSet<mlir::Operation*, 1> exceptions = {terminator_op};
  old_val.replaceAllUsesExcept(new_val, exceptions);
}

struct TensorArrayStats {
  // Whether a write op should accumulate with the old value. Set to true if
  // this is a gradient.
  bool accumulate_on_write;
  // Maps from a gradient source string to the local variable to the gradient.
  llvm::StringMap<Value> grads;
};

LogicalResult HandleTensorArrayV3Op(
    TF::TensorArrayV3Op ta, ModuleOp module,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats) {
  auto elem_shape = GetTensorArrayElementShape(ta, module);
  if (!elem_shape) return ta.emitOpError("unknown element shape");
  if (ta.getDynamicSize()) {
    return ta.emitOpError("dynamic tensor array size is unsupported");
  }
  Value buffer;
  OpBuilder builder(ta);
  if (failed(cutil::CreateInitBufferValue(*elem_shape, ta.getSize(), ta,
                                          ta.getDtype(), builder, &buffer))) {
    return failure();
  }
  auto var_type = RankedTensorType::get(
      {}, TF::ResourceType::get(
              ArrayRef<TensorType>{buffer.getType().cast<TensorType>()},
              ta.getContext()));
  auto local_var = builder.create<TF::MlirLocalVarOp>(
      ta.getLoc(), ArrayRef<Type>{var_type}, ArrayRef<Value>{});
  cutil::WriteLocalVariable(local_var, buffer, builder, ta.getLoc());
  ta.getHandle().replaceAllUsesWith(local_var);
  // The flow output is just a way for the front end to enforce ordering among
  // tensor array ops, but in the MLIR TF dialect they have sequential ordering.
  // Just create a constant to replace its uses.
  tensorflow::Tensor scalar_tensor(tensorflow::DT_FLOAT, {});
  scalar_tensor.scalar<float>()() = 0.0f;
  auto flow = builder.create<TF::ConstOp>(
      ta.getLoc(), tensorflow::ConvertTensor(scalar_tensor, &builder).value());
  ta.getFlow().replaceAllUsesWith(flow);
  ta.erase();
  (*stats)[local_var].accumulate_on_write = false;
  return success();
}

LogicalResult HandleTensorArrayReadV3Op(
    TF::TensorArrayReadV3Op read,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = read.getHandle();
  if (stats.count(local_var) == 0) {
    return read.emitOpError("unknown tensor array");
  }
  OpBuilder builder(read);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, read.getLoc());
  auto index_reshape =
      cutil::ReshapeScalarToSizeType(builder, read.getIndex(), read.getLoc());
  auto elem = cutil::GetElement(index_reshape, buffer, builder, read.getLoc());
  ReplaceAllUsesExceptTerminator(read.getValue(), elem);
  ReplaceAllUsesWithCast(read.getValue(), elem);
  read.erase();
  // The clear_after_read attribute does not mean setting the tensor to 0 after
  // read; instead it does not allow a second read before the next write. We
  // follow the old bridge's implementation not to do anything here.
  return success();
}

LogicalResult HandleTensorArrayWriteV3Op(
    TF::TensorArrayWriteV3Op write,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = write.getHandle();
  auto stat_it = stats.find(local_var);
  if (stat_it == stats.end()) return write.emitOpError("unknown tensor array");
  OpBuilder builder(write);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, write.getLoc());
  auto index_reshape =
      cutil::ReshapeScalarToSizeType(builder, write.getIndex(), write.getLoc());
  Value elem = write.getValue();
  if (stat_it->getSecond().accumulate_on_write) {
    // Get the old slice, and accumulate with it. We set keep_slice_shape
    // (keeping the leading size-1 dimension) because it avoids reshape back and
    // forth.
    auto original_elem =
        cutil::GetElement(index_reshape, buffer, builder, write.getLoc(),
                          /*keep_slice_shape=*/true);
    // Add a size-1 leading dimension to elem.
    auto slice_type = original_elem.getType().cast<RankedTensorType>();
    elem = builder.create<TF::ReshapeOp>(
        write.getLoc(), ArrayRef<Type>{slice_type},
        ArrayRef<Value>{elem, cutil::GetR1Const(slice_type.getShape(), builder,
                                                write.getLoc())});
    elem =
        cutil::AccumulateBuffers(elem, original_elem, builder, write.getLoc());
  }
  buffer =
      cutil::SetElement(index_reshape, buffer, elem, builder, write.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, write.getLoc());
  write.getFlowOut().replaceAllUsesWith(write.getFlowIn());
  write.erase();
  return success();
}

LogicalResult HandleTensorArrayConcatV3Op(
    TF::TensorArrayConcatV3Op concat,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = concat.getHandle();
  if (stats.count(local_var) == 0) {
    return concat.emitOpError("unknown tensor array");
  }
  OpBuilder builder(concat);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, concat.getLoc());
  auto buffer_type = buffer.getType().cast<RankedTensorType>();
  if (buffer_type.getShape().size() <= 1) {
    return concat.emitOpError("cannot concat on scalar-element tensor array");
  }
  // Merget he first two dimensions.
  auto shape = llvm::to_vector<8>(buffer_type.getShape().drop_front());
  shape[0] *= buffer_type.getDimSize(0);
  buffer = builder.create<TF::ReshapeOp>(
      concat.getLoc(),
      ArrayRef<Type>{
          RankedTensorType::get(shape, buffer_type.getElementType())},
      ArrayRef<Value>{buffer,
                      cutil::GetR1Const(shape, builder, concat.getLoc())});
  ReplaceAllUsesExceptTerminator(concat.getValue(), buffer);
  ReplaceAllUsesWithCast(concat.getValue(), buffer);

  // Create the lengths as a list of the same value (element size).
  tensorflow::Tensor lengths_tensor(tensorflow::DT_INT64,
                                    {buffer_type.getDimSize(0)});
  for (int64_t i = 0; i < buffer_type.getDimSize(0); ++i) {
    lengths_tensor.vec<int64_t>()(i) = buffer_type.getDimSize(1);
  }
  concat.getLengths().replaceAllUsesWith(builder.create<TF::ConstOp>(
      concat.getLoc(),
      tensorflow::ConvertTensor(lengths_tensor, &builder).value()));
  concat.erase();
  return success();
}

LogicalResult HandleTensorArraySplitV3Op(
    TF::TensorArraySplitV3Op split,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = split.getHandle();
  if (stats.count(local_var) == 0) {
    return split.emitOpError("unknown tensor array");
  }
  OpBuilder builder(split);
  int64_t count;
  RankedTensorType elem_type;
  if (failed(GetSplitElementTypeAndCount(split, &elem_type, &count))) {
    return failure();
  }
  llvm::SmallVector<int64_t, 8> buffer_shape;
  buffer_shape.push_back(count);
  for (int64_t dim : elem_type.getShape()) buffer_shape.push_back(dim);
  // Reshape the input to match the buffer of the tensor array.
  Value buffer =
      builder
          .create<TF::ReshapeOp>(
              split.getLoc(),
              ArrayRef<Type>{RankedTensorType::get(buffer_shape,
                                                   elem_type.getElementType())},
              ArrayRef<Value>{
                  split.getValue(),
                  cutil::GetR1Const(buffer_shape, builder, split.getLoc())})
          .getOutput();
  // Accumulate with the old buffer.
  auto old_buffer =
      cutil::ReadLocalVariable(local_var, builder, split.getLoc());
  buffer =
      cutil::AccumulateBuffers(old_buffer, buffer, builder, split.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, split.getLoc());
  split.getFlowOut().replaceAllUsesWith(split.getFlowIn());
  split.erase();
  return success();
}

LogicalResult HandleTensorArraySizeV3Op(
    TF::TensorArraySizeV3Op size,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = size.getHandle();
  if (stats.count(local_var) == 0) {
    return size.emitOpError("unknown tensor array");
  }
  auto buffer_type = getElementTypeOrSelf(local_var.getType())
                         .cast<TF::ResourceType>()
                         .getSubtypes()[0]
                         .cast<RankedTensorType>();
  OpBuilder builder(size);
  auto result = cutil::CreateScalarConst(buffer_type.getDimSize(0), builder,
                                         size.getLoc());
  size.getSize().replaceAllUsesWith(result);
  size.erase();
  return success();
}

LogicalResult CreateAndInitializeGradVariable(Type local_var_type,
                                              Operation* op, Value* var) {
  OpBuilder builder(op);
  *var = builder.create<TF::MlirLocalVarOp>(
      op->getLoc(), ArrayRef<Type>{local_var_type}, ArrayRef<Value>{});
  Value buffer;
  auto buffer_type = getElementTypeOrSelf(local_var_type)
                         .cast<TF::ResourceType>()
                         .getSubtypes()[0]
                         .cast<RankedTensorType>();
  if (failed(cutil::CreateInitBufferValue(
          buffer_type.getShape().drop_front(), buffer_type.getDimSize(0), op,
          buffer_type.getElementType(), builder, &buffer))) {
    return failure();
  }
  cutil::WriteLocalVariable(*var, buffer, builder, op->getLoc());
  return success();
}

LogicalResult HandleTensorArrayGradV3Op(
    TF::TensorArrayGradV3Op grad,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats) {
  auto local_var = grad.getHandle();
  OpBuilder builder(grad);
  Value grad_var;
  auto sit = stats->find(local_var);
  if (sit == stats->end()) return grad.emitOpError("unknown tensor array");
  auto emplace_res =
      sit->getSecond().grads.try_emplace(grad.getSource().str(), Value());
  if (!emplace_res.second) {
    // If the source has been assigned a grad, use it.
    grad_var = emplace_res.first->second;
  } else {
    if (failed(CreateAndInitializeGradVariable(local_var.getType(), grad,
                                               &grad_var))) {
      return failure();
    }
    emplace_res.first->second = grad_var;
    // Write to a grad accumulates with previous writes.
    (*stats)[grad_var].accumulate_on_write = true;
  }
  grad.getFlowOut().replaceAllUsesWith(grad.getFlowIn());
  grad.getGradHandle().replaceAllUsesWith(grad_var);
  grad.erase();
  return success();
}

LogicalResult HandleTensorArrayGatherV3Op(
    TF::TensorArrayGatherV3Op gather,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = gather.getHandle();
  if (stats.count(local_var) == 0) {
    return gather.emitOpError("unknown tensor array");
  }
  OpBuilder builder(gather);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, gather.getLoc());
  auto result = cutil::GatherElements(gather.getIndices(), buffer, builder,
                                      gather.getLoc());
  ReplaceAllUsesExceptTerminator(gather.getValue(), result);
  ReplaceAllUsesWithCast(gather.getValue(), result);
  gather.erase();
  return success();
}

LogicalResult HandleTensorArrayScatterV3Op(
    TF::TensorArrayScatterV3Op scatter,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = scatter.getHandle();
  if (stats.count(local_var) == 0) {
    return scatter.emitOpError("unknown tensor array");
  }
  OpBuilder builder(scatter);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, scatter.getLoc());
  buffer =
      cutil::ScatterAccumulateElements(scatter.getIndices(), scatter.getValue(),
                                       buffer, builder, scatter.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, scatter.getLoc());
  scatter.getFlowOut().replaceAllUsesWith(scatter.getFlowIn());
  scatter.erase();
  return success();
}

// Updates func's type according to its current arguments and return values.
void UpdateFuncType(func::FuncOp func) {
  llvm::SmallVector<Type, 8> arg_types;
  for (auto arg : func.getArguments()) arg_types.push_back(arg.getType());
  func.setType(
      FunctionType::get(func.getContext(), arg_types,
                        func.front().getTerminator()->getOperandTypes()));
}

// Finds the accessed gradient sources for each tensor array argument.
llvm::SmallDenseMap<int64_t, llvm::SmallVector<string, 4>> AccessedGradients(
    ArrayRef<func::FuncOp> funcs, ModuleOp module) {
  llvm::SmallDenseMap<int64_t, llvm::SmallVector<string, 4>> result;
  llvm::SmallDenseMap<int64_t, llvm::StringSet<>> result_sets;
  auto insert = [&](Value v, const string& source, const Block& func_block) {
    auto arg = v.dyn_cast<BlockArgument>();
    if (!arg || arg.getOwner() != &func_block) return;
    auto insert_res = result_sets[arg.getArgNumber()].insert(source);
    if (!insert_res.second) return;
    result[arg.getArgNumber()].push_back(source);
  };
  for (func::FuncOp func : funcs) {
    const Block& func_block = func.front();
    // Walk all operations and nested regions to find accessed gradient sources
    // for function arguments.
    func.walk([&](Operation* op) {
      if (llvm::isa<TF::IdentityOp, TF::IdentityNOp>(op)) {
        op->replaceAllUsesWith(op->getOperands());
        return;
      }
      if (auto grad = llvm::dyn_cast<TF::TensorArrayGradV3Op>(op)) {
        insert(grad.getHandle(), grad.getSource().str(), func_block);
      } else if (auto while_op = llvm::dyn_cast<TF::WhileOp>(op)) {
        for (const auto& entry : AccessedGradients(
                 {while_op.body_function(), while_op.cond_function()}, module))
          for (const string& source : entry.getSecond())
            insert(while_op.getOperand(entry.getFirst()), source, func_block);
      } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(op)) {
        for (const auto& entry : AccessedGradients(
                 {if_op.then_function(), if_op.else_function()}, module))
          for (const string& source : entry.getSecond())
            insert(if_op.getOperand(entry.getFirst() + 1), source, func_block);
      } else if (auto call = llvm::dyn_cast<CallOpInterface>(op)) {
        auto callee = dyn_cast<func::FuncOp>(call.resolveCallable());
        for (const auto& entry : AccessedGradients({callee}, module))
          for (const string& source : entry.getSecond())
            insert(call.getArgOperands()[entry.getFirst()], source, func_block);
      }
    });
  }
  return result;
}

// Contains cached information for decomposed callee functions for (stateful)
// partitioned call ops.
struct PartitionedCallTensorArrayOpsInfo {
  bool signature_change;
  func::FuncOp decomposed_callee;
  llvm::SmallVector<std::pair<int64_t, llvm::SmallVector<string, 4>>, 4>
      arg_grads;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> ret_forward_input;
};

// Updates a called function's input signature by adjusting resource types, and
// adding required gradient arguments.
void ChangeFunctionInputSignature(
    func::FuncOp func,
    const llvm::SmallDenseMap<int64_t, llvm::SmallVector<string, 4>>& grads,
    llvm::function_ref<Type(int64_t)> ta_arg_buffer_type,
    llvm::function_ref<bool(int64_t)> ta_accumulate_on_write,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats) {
  int64_t original_args = func.getNumArguments();
  for (int64_t argnum = 0; argnum < original_args; ++argnum) {
    auto arg = func.getArgument(argnum);
    Type t = ta_arg_buffer_type(argnum);
    if (!t) continue;
    arg.setType(t);
    auto grad_it = grads.find(argnum);
    if (grad_it == grads.end()) continue;
    llvm::StringMap<Value> grads_map;
    for (const string& source : grad_it->getSecond()) {
      auto g = func.front().addArgument(t, func.getLoc());
      (*stats)[g].accumulate_on_write = true;
      grads_map[source] = g;
    }
    auto& stat = (*stats)[arg];
    stat.accumulate_on_write = ta_accumulate_on_write(argnum);
    stat.grads = std::move(grads_map);
  }
  UpdateFuncType(func);
}

LogicalResult DecomposeTensorArrayOps(
    Block*, ModuleOp, llvm::SmallDenseMap<Value, TensorArrayStats>*,
    llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*);

LogicalResult HandleWhileOp(TF::WhileOp while_op, ModuleOp module,
                            llvm::SmallDenseMap<Value, TensorArrayStats>* stats,
                            llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*
                                decomposed_partitioned_call_callees) {
  auto body = while_op.body_function();
  auto cond = while_op.cond_function();
  auto grads = AccessedGradients({body, cond}, module);
  auto ta_arg_buffer_type = [&](int64_t index) -> Type {
    auto it = stats->find(while_op.getOperand(index));
    if (it == stats->end()) return nullptr;
    return it->getFirst().getType();
  };
  auto ta_accumulate_on_write = [&](int64_t index) {
    auto it = stats->find(while_op.getOperand(index));
    if (it == stats->end()) return false;
    return it->getSecond().accumulate_on_write;
  };
  llvm::SmallDenseMap<Value, TensorArrayStats> body_stats;
  ChangeFunctionInputSignature(body, grads, ta_arg_buffer_type,
                               ta_accumulate_on_write, &body_stats);
  llvm::SmallDenseMap<Value, TensorArrayStats> cond_stats;
  ChangeFunctionInputSignature(cond, grads, ta_arg_buffer_type,
                               ta_accumulate_on_write, &cond_stats);
  if (failed(DecomposeTensorArrayOps(&body.front(), module, &body_stats,
                                     decomposed_partitioned_call_callees)) ||
      failed(DecomposeTensorArrayOps(&cond.front(), module, &cond_stats,
                                     decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (body_stats.empty() && cond_stats.empty()) return success();
  auto old_body_ret = body.front().getTerminator();
  auto new_retvals = llvm::to_vector<8>(old_body_ret->getOperands());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    if (!ta_arg_buffer_type(i)) continue;
    auto retval = old_body_ret->getOperand(i);
    auto arg = retval.dyn_cast<BlockArgument>();
    if (!arg) {
      return while_op.emitOpError(
          "output tensor array does not alias input in a while loop");
    }
    for (const string& source : grads[i]) {
      new_retvals.push_back(body_stats[arg].grads[source]);
    }
  }
  OpBuilder(old_body_ret)
      .create<func::ReturnOp>(old_body_ret->getLoc(), new_retvals);
  old_body_ret->erase();
  UpdateFuncType(body);
  // Recreate the while op.
  auto operands = llvm::to_vector<8>(while_op.getOperands());
  for (int64_t i = 0; i < while_op.getNumOperands(); ++i) {
    auto grad_it = grads.find(i);
    auto& stat = (*stats)[operands[i]];
    if (grad_it == grads.end()) continue;
    for (const string& source : grad_it->getSecond()) {
      auto it = stat.grads.find(source);
      if (it != stat.grads.end()) {
        operands.push_back(it->second);
      } else {
        Value grad_var;
        if (failed(CreateAndInitializeGradVariable(operands[i].getType(),
                                                   while_op, &grad_var))) {
          return failure();
        }
        stat.grads[source] = grad_var;
        operands.push_back(grad_var);
        (*stats)[grad_var].accumulate_on_write = true;
      }
    }
  }
  OpBuilder builder(while_op);
  auto new_while = builder.create<TF::WhileOp>(
      while_op.getLoc(), body.getFunctionType().getInputs(), operands,
      while_op->getAttrs());
  for (int64_t i = 0; i < while_op.getNumOperands(); ++i) {
    if (ta_arg_buffer_type(i)) {
      while_op.getResult(i).replaceAllUsesWith(while_op.getOperand(i));
    } else {
      while_op.getResult(i).replaceAllUsesWith(new_while.getResult(i));
    }
  }
  while_op.erase();
  return success();
}

LogicalResult HandleIfOp(TF::IfOp if_op, ModuleOp module,
                         llvm::SmallDenseMap<Value, TensorArrayStats>* stats,
                         llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*
                             decomposed_partitioned_call_callees) {
  auto then_branch = if_op.then_function();
  auto else_branch = if_op.else_function();
  auto grads = AccessedGradients({then_branch, else_branch}, module);
  auto ta_arg_buffer_type = [&](int64_t index) -> Type {
    auto it = stats->find(if_op.getOperand(index + 1));
    if (it == stats->end()) return nullptr;
    return it->getFirst().getType();
  };
  auto ta_accumulate_on_write = [&](int64_t index) {
    auto it = stats->find(if_op.getOperand(index + 1));
    if (it == stats->end()) return false;
    return it->getSecond().accumulate_on_write;
  };
  llvm::SmallDenseMap<Value, TensorArrayStats> then_stats;
  ChangeFunctionInputSignature(then_branch, grads, ta_arg_buffer_type,
                               ta_accumulate_on_write, &then_stats);
  llvm::SmallDenseMap<Value, TensorArrayStats> else_stats;
  ChangeFunctionInputSignature(else_branch, grads, ta_arg_buffer_type,
                               ta_accumulate_on_write, &else_stats);
  if (failed(DecomposeTensorArrayOps(&then_branch.front(), module, &then_stats,
                                     decomposed_partitioned_call_callees)) ||
      failed(DecomposeTensorArrayOps(&else_branch.front(), module, &else_stats,
                                     decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (then_stats.empty() && else_stats.empty()) return success();
  // Recreate the if op.
  auto operands = llvm::to_vector<8>(if_op.getOperands());
  for (int64_t i = 0; i < if_op.getNumOperands() - 1; ++i) {
    auto grad_it = grads.find(i);
    auto& stat = (*stats)[operands[i + 1]];
    if (grad_it == grads.end()) continue;
    for (const string& source : grad_it->getSecond()) {
      auto it = stat.grads.find(source);
      if (it != stat.grads.end()) {
        operands.push_back(it->second);
      } else {
        Value grad_var;
        if (failed(CreateAndInitializeGradVariable(operands[i + 1].getType(),
                                                   if_op, &grad_var))) {
          return failure();
        }
        stat.grads[source] = grad_var;
        operands.push_back(grad_var);
        (*stats)[grad_var].accumulate_on_write = true;
      }
    }
  }
  OpBuilder builder(if_op);
  auto new_if = builder.create<TF::IfOp>(
      if_op.getLoc(), then_branch.getFunctionType().getResults(), operands,
      if_op->getAttrs());
  auto ret_forwards_input = [](func::FuncOp f, int64_t ret_ind) -> int64_t {
    auto retval = f.front().getTerminator()->getOperand(ret_ind);
    auto arg = retval.dyn_cast<BlockArgument>();
    if (!arg) return -1;
    return arg.getArgNumber();
  };
  for (int64_t i = 0; i < if_op.getNumResults(); ++i) {
    if (!getElementTypeOrSelf(if_op.getResult(i).getType())
             .isa<TF::ResourceType>()) {
      if_op.getResult(i).replaceAllUsesWith(new_if.getResult(i));
      continue;
    }
    int64_t then_forward_input = ret_forwards_input(then_branch, i);
    int64_t else_foward_input = ret_forwards_input(else_branch, i);
    if (then_forward_input != else_foward_input || then_forward_input < 0) {
      return if_op.emitOpError(
          "branches do not forward the same input resource");
    }
    if_op.getResult(i).replaceAllUsesWith(
        if_op.getOperand(then_forward_input + 1));
  }
  if_op.erase();
  return success();
}

template <typename CallOp>
LogicalResult HandlePartitionedCallOp(
    CallOp call, func::FuncOp callee, ModuleOp module,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats,
    llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto emplace_res = decomposed_partitioned_call_callees->try_emplace(
      callee.getName(), PartitionedCallTensorArrayOpsInfo());
  auto& info = emplace_res.first->second;
  // Recreates the call op with info.
  auto recreate_caller = [&]() -> LogicalResult {
    auto new_operands = llvm::to_vector<8>(call.getOperands());
    for (const auto& entry : info.arg_grads) {
      auto it = stats->find(call.getOperand(entry.first));
      if (it == stats->end()) return call.emitOpError("unknown tensor array");
      for (const string& source : entry.second) {
        auto grad_it = it->getSecond().grads.find(source);
        if (grad_it != it->getSecond().grads.end()) {
          new_operands.push_back(grad_it->second);
        } else {
          Value grad_var;
          if (failed(CreateAndInitializeGradVariable(it->getFirst().getType(),
                                                     call, &grad_var))) {
            return failure();
          }
          it->getSecond().grads[source] = grad_var;
          new_operands.push_back(grad_var);
        }
      }
    }
    OpBuilder builder(call);
    auto new_call = builder.create<CallOp>(
        call.getLoc(), info.decomposed_callee.getFunctionType().getResults(),
        new_operands, call->getAttrs());
    new_call->setAttr(
        "f", SymbolRefAttr::get(
                 builder.getContext(),
                 const_cast<func::FuncOp&>(info.decomposed_callee).getName()));
    for (const auto& entry : info.ret_forward_input) {
      call.getResult(entry.first)
          .replaceAllUsesWith(call.getOperand(entry.second));
    }
    call.replaceAllUsesWith(new_call);
    call.erase();
    return success();
  };
  if (!emplace_res.second) {
    // This callee was handled before.
    if (!info.signature_change) return success();
    return recreate_caller();
  }
  // Rewrite the callee.
  info.signature_change = false;
  auto ta_arg_buffer_type = [&](int64_t index) -> Type {
    auto it = stats->find(call.getOperand(index));
    if (it == stats->end()) return nullptr;
    info.signature_change = true;
    return it->getFirst().getType();
  };
  auto ta_accumulate_on_write = [&](int64_t index) {
    auto it = stats->find(call.getOperand(index));
    if (it == stats->end()) return false;
    return it->getSecond().accumulate_on_write;
  };
  func::FuncOp lowered_callee = callee;
  if (!callee.isPrivate()) {
    // Clone non-private callee in case of signature change.
    lowered_callee = callee.clone();
    lowered_callee.setPrivate();
  }
  auto grads = AccessedGradients({lowered_callee}, module);
  for (int64_t i = 0; i < lowered_callee.getNumArguments(); ++i) {
    auto it = grads.find(i);
    if (it == grads.end()) continue;
    info.arg_grads.emplace_back(i, it->getSecond());
  }
  llvm::SmallDenseMap<Value, TensorArrayStats> callee_stats;
  ChangeFunctionInputSignature(lowered_callee, grads, ta_arg_buffer_type,
                               ta_accumulate_on_write, &callee_stats);
  if (failed(DecomposeTensorArrayOps(&lowered_callee.front(), module,
                                     &callee_stats,
                                     decomposed_partitioned_call_callees))) {
    return failure();
  }
  for (int64_t i = 0; i < call.getNumResults(); ++i) {
    auto ret = lowered_callee.front().getTerminator()->getOperand(i);
    if (!getElementTypeOrSelf(ret.getType()).isa<TF::ResourceType>()) continue;
    auto arg = ret.dyn_cast<BlockArgument>();
    if (!arg) continue;
    info.ret_forward_input.emplace_back(i, arg.getArgNumber());
  }

  info.decomposed_callee = lowered_callee;
  if (lowered_callee != callee) {
    if (!info.signature_change) {
      // Signature is not modified. We do not need to keep two copies.
      lowered_callee.setName(
          StringAttr::get(callee->getContext(), callee.getName()));
      callee.erase();
    } else {
      // Add the clone with a new name.
      lowered_callee.setName(StringAttr::get(
          callee->getContext(),
          llvm::formatv("{0}_tensorarray_decomposed", callee.getName()).str()));
    }
    SymbolTable(module).insert(lowered_callee);
  }
  if (info.signature_change) return recreate_caller();
  return success();
}

LogicalResult HandleRegionControlFlowOps(
    Operation& op, ModuleOp module,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats,
    llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*
        decomposed_partitioned_call_callees) {
  for (OpOperand& operand : op.getOpOperands()) {
    if (getElementTypeOrSelf(operand.get().getType()).isa<TF::ResourceType>()) {
      return op.emitOpError()
             << "found unexpected type " << operand.get().getType()
             << " of operand #" << operand.getOperandNumber()
             << ", resource type operands are expected to have been "
                "canonicalized away for region based control flow ops";
    }
  }
  for (OpResult result : op.getResults()) {
    if (getElementTypeOrSelf(result.getType()).isa<TF::ResourceType>()) {
      return op.emitOpError()
             << "found unexpected type " << result.getType() << " of result #"
             << result.getResultNumber()
             << ", resource type results are expected to have been "
                "canonicalized away for region based control flow ops";
    }
  }

  for (Region& region : op.getRegions()) {
    if (failed(DecomposeTensorArrayOps(&region.front(), module, stats,
                                       decomposed_partitioned_call_callees)))
      return failure();
  }
  return success();
}

LogicalResult DecomposeTensorArrayOps(
    Block* block, ModuleOp module,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats,
    llvm::StringMap<PartitionedCallTensorArrayOpsInfo>*
        decomposed_partitioned_call_callees) {
  for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
    if (llvm::isa<TF::IdentityOp, TF::IdentityNOp>(&op)) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    } else if (auto ta = llvm::dyn_cast<TF::TensorArrayV3Op>(&op)) {
      if (failed(HandleTensorArrayV3Op(ta, module, stats))) {
        return failure();
      }
    } else if (auto read = llvm::dyn_cast<TF::TensorArrayReadV3Op>(&op)) {
      if (failed(HandleTensorArrayReadV3Op(read, *stats))) return failure();
    } else if (auto write = llvm::dyn_cast<TF::TensorArrayWriteV3Op>(&op)) {
      if (failed(HandleTensorArrayWriteV3Op(write, *stats))) return failure();
    } else if (auto concat = llvm::dyn_cast<TF::TensorArrayConcatV3Op>(&op)) {
      if (failed(HandleTensorArrayConcatV3Op(concat, *stats))) return failure();
    } else if (auto split = llvm::dyn_cast<TF::TensorArraySplitV3Op>(&op)) {
      if (failed(HandleTensorArraySplitV3Op(split, *stats))) return failure();
    } else if (auto size = llvm::dyn_cast<TF::TensorArraySizeV3Op>(&op)) {
      if (failed(HandleTensorArraySizeV3Op(size, *stats))) return failure();
    } else if (auto grad = llvm::dyn_cast<TF::TensorArrayGradV3Op>(&op)) {
      if (failed(HandleTensorArrayGradV3Op(grad, stats))) return failure();
    } else if (auto gather = llvm::dyn_cast<TF::TensorArrayGatherV3Op>(&op)) {
      if (failed(HandleTensorArrayGatherV3Op(gather, *stats))) return failure();
    } else if (auto scatter = llvm::dyn_cast<TF::TensorArrayScatterV3Op>(&op)) {
      if (failed(HandleTensorArrayScatterV3Op(scatter, *stats))) {
        return failure();
      }
    } else if (auto close = llvm::dyn_cast<TF::TensorArrayCloseV3Op>(&op)) {
      close.erase();
    } else if (auto while_op = llvm::dyn_cast<TF::WhileOp>(&op)) {
      if (failed(HandleWhileOp(while_op, module, stats,
                               decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(&op)) {
      if (failed(HandleIfOp(if_op, module, stats,
                            decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (llvm::isa<TF::CaseRegionOp>(op) ||
               llvm::isa<TF::IfRegionOp>(op) ||
               llvm::isa<TF::WhileRegionOp>(op)) {
      if (failed(HandleRegionControlFlowOps(
              op, module, stats, decomposed_partitioned_call_callees)))
        return failure();
    } else if (auto pcall = llvm::dyn_cast<TF::PartitionedCallOp>(&op)) {
      auto callee = pcall.func();
      if (!callee)
        return pcall.emitOpError(
            "TensorArray decomposition does not support call with nested "
            "references.");

      if (failed(
              HandlePartitionedCallOp(pcall, callee, module, stats,
                                      decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto spcall =
                   llvm::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      if (failed(
              HandlePartitionedCallOp(spcall, spcall.func(), module, stats,
                                      decomposed_partitioned_call_callees))) {
        return failure();
      }
    }
  }
  return success();
}

void TensorArrayOpsDecompositionPass::runOnOperation() {
  auto module = getOperation();
  auto main = module.lookupSymbol<func::FuncOp>("main");
  if (!main) return;
  llvm::SmallDenseMap<Value, TensorArrayStats> stats;
  llvm::StringMap<PartitionedCallTensorArrayOpsInfo>
      decomposed_partitioned_call_callees;
  if (failed(DecomposeTensorArrayOps(&main.front(), module, &stats,
                                     &decomposed_partitioned_call_callees))) {
    signalPassFailure();
  }
}

}  // namespace

namespace TF {

std::unique_ptr<OperationPass<ModuleOp>>
CreateTensorArrayOpsDecompositionPass() {
  return std::make_unique<TensorArrayOpsDecompositionPass>();
}

}  // namespace TF
}  // namespace mlir
