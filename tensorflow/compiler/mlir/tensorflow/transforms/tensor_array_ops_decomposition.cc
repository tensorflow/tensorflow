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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
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
    : public ModulePass<TensorArrayOpsDecompositionPass> {
  void runOnModule() override;
};

// Infers the element type and count for a TensorArraySplitV3Op. Requires
// constant lengths and static shape on the input value.
LogicalResult GetSplitElementTypeAndCount(TF::TensorArraySplitV3Op split,
                                          RankedTensorType* elem_type,
                                          int64_t* count) {
  auto lengths_const =
      llvm::dyn_cast_or_null<TF::ConstOp>(split.lengths().getDefiningOp());
  if (!lengths_const) return split.emitOpError("non-constant split lengths");
  *count = lengths_const.value().getNumElements();
  if (*count <= 0) return split.emitOpError("non-positive split count");
  auto buffer_type = split.value().getType().dyn_cast<RankedTensorType>();
  if (!buffer_type || !buffer_type.hasStaticShape() ||
      buffer_type.getRank() < 1) {
    return split.emitOpError("unknown or invalid split tensor shape");
  }
  int64_t length = buffer_type.getDimSize(0) / *count;
  for (auto len : lengths_const.value().getValues<APInt>()) {
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
llvm::Optional<llvm::SmallVector<int64_t, 8>> GetTensorArrayElementShape(
    TF::TensorArrayV3Op ta, ModuleOp module) {
  tensorflow::TensorShapeProto element_shape;
  if (tensorflow::mangling_util::DemangleShape(ta.element_shape().str(),
                                               &element_shape)
          .ok()) {
    tensorflow::PartialTensorShape shape(element_shape);
    if (shape.IsFullyDefined()) {
      // Convert int64 to int64_.
      auto int64_dims = shape.dim_sizes();
      llvm::SmallVector<int64_t, 8> dims(int64_dims.begin(), int64_dims.end());
      return dims;
    }
  }

  bool has_failure = false;
  auto elem_type = cutil::GetElementTypeFromAccess(
      ta.handle(), module, [&](Operation* user) -> llvm::Optional<Type> {
        if (has_failure) return llvm::None;
        if (auto write = llvm::dyn_cast<TF::TensorArrayWriteV3Op>(user)) {
          return write.value().getType();
        } else if (auto split =
                       llvm::dyn_cast<TF::TensorArraySplitV3Op>(user)) {
          if (!split.lengths().getDefiningOp() ||
              !llvm::isa<TF::ConstOp>(split.lengths().getDefiningOp())) {
            return llvm::None;
          }
          RankedTensorType t;
          int64_t count;
          if (failed(GetSplitElementTypeAndCount(split, &t, &count))) {
            has_failure = true;
            return llvm::None;
          }
          return t;
        }
        return llvm::None;
      });
  if (!elem_type) return llvm::None;
  return llvm::to_vector<8>(elem_type->getShape());
}

struct TensorArrayStats {
  // Whether a write op should accumulate with the old value. Set to true if
  // this is a gradient.
  bool accumulate_on_write;
  // Maps from a gradient source string to the local variable to the gradient.
  llvm::SmallDenseMap<llvm::StringRef, Value> grads;
};

LogicalResult HandleTensorArrayV3Op(
    TF::TensorArrayV3Op ta, ModuleOp module,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats) {
  auto elem_shape = GetTensorArrayElementShape(ta, module);
  if (!elem_shape) return ta.emitOpError("unknown element shape");
  if (ta.dynamic_size()) {
    return ta.emitOpError("dynamic tensor array size is unsupported");
  }
  Value buffer;
  OpBuilder builder(ta);
  if (failed(cutil::CreateInitBufferValue(*elem_shape, ta.size(), ta,
                                          ta.dtype(), builder, &buffer))) {
    return failure();
  }
  auto var_type = RankedTensorType::get(
      {}, TF::ResourceType::get(
              ArrayRef<TensorType>{buffer.getType().cast<TensorType>()},
              ta.getContext()));
  auto local_var = builder.create<TF::MlirLocalVarOp>(
      ta.getLoc(), ArrayRef<Type>{var_type}, ArrayRef<Value>{},
      ArrayRef<NamedAttribute>{});
  cutil::WriteLocalVariable(local_var, buffer, builder, ta.getLoc());
  ta.handle().replaceAllUsesWith(local_var);
  // The flow output is just a way for the front end to enforce ordering among
  // tensor array ops, but in the MLIR TF dialect they have sequential ordering.
  // Just create a constant to replace its uses.
  tensorflow::Tensor scalar_tensor(tensorflow::DT_FLOAT, {});
  scalar_tensor.scalar<float>()() = 0.0f;
  auto flow = builder.create<TF::ConstOp>(
      ta.getLoc(),
      tensorflow::ConvertTensor(scalar_tensor, &builder).ValueOrDie());
  ta.flow().replaceAllUsesWith(flow);
  ta.erase();
  (*stats)[local_var].accumulate_on_write = false;
  return success();
}

LogicalResult HandleTensorArrayReadV3Op(
    TF::TensorArrayReadV3Op read,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = read.handle();
  if (stats.count(local_var) == 0) {
    return read.emitOpError("unknown tensor array");
  }
  OpBuilder builder(read);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, read.getLoc());
  auto index_reshape =
      cutil::ReshapeScalarToSizeType(builder, read.index(), read.getLoc());
  auto elem = cutil::GetElement(index_reshape, buffer, builder, read.getLoc());
  read.value().replaceAllUsesWith(elem);
  read.erase();
  // The clear_after_read attribute does not mean setting the tensor to 0 after
  // read; instead it does not allow a second read before the next write. We
  // follow the old bridge's implementation not to do anything here.
  return success();
}

LogicalResult HandleTensorArrayWriteV3Op(
    TF::TensorArrayWriteV3Op write,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = write.handle();
  auto stat_it = stats.find(local_var);
  if (stat_it == stats.end()) return write.emitOpError("unknown tensor array");
  OpBuilder builder(write);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, write.getLoc());
  auto index_reshape =
      cutil::ReshapeScalarToSizeType(builder, write.index(), write.getLoc());
  auto elem = write.value();
  if (stat_it->getSecond().accumulate_on_write) {
    // Get the old slice, and accumulate with it. We set keep_slice_shape
    // (keeping the leading size-1 dimension) because it avoids reshape back and
    // forth.
    auto original_elem =
        cutil::GetElement(index_reshape, buffer, builder, write.getLoc(),
                          /*keep_slice_shape=*/true);
    // Add a size-1 leading dimension to elem.
    for (auto dim : buffer.getType().cast<RankedTensorType>().getShape())
      LOG(ERROR) << " buffer : " << dim;
    auto slice_type = original_elem.getType().cast<RankedTensorType>();
    for (auto dim : slice_type.getShape()) LOG(ERROR) << " resahpe : " << dim;
    elem = builder.create<TF::ReshapeOp>(
        write.getLoc(), ArrayRef<Type>{slice_type},
        ArrayRef<Value>{elem, cutil::GetR1Const(slice_type.getShape(), builder,
                                                write.getLoc())},
        ArrayRef<NamedAttribute>{});
    elem =
        cutil::AccumulateBuffers(elem, original_elem, builder, write.getLoc());
  }
  buffer =
      cutil::SetElement(index_reshape, buffer, elem, builder, write.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, write.getLoc());
  write.flow_out().replaceAllUsesWith(write.flow_in());
  write.erase();
  return success();
}

LogicalResult HandleTensorArrayConcatV3Op(
    TF::TensorArrayConcatV3Op concat,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = concat.handle();
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
                      cutil::GetR1Const(shape, builder, concat.getLoc())},
      ArrayRef<NamedAttribute>{});
  concat.value().replaceAllUsesWith(buffer);

  // Create the lengths as a list of the same value (element size).
  tensorflow::Tensor lengths_tensor(tensorflow::DT_INT64,
                                    {buffer_type.getDimSize(0)});
  for (int64_t i = 0; i < buffer_type.getDimSize(0); ++i) {
    lengths_tensor.vec<tensorflow::int64>()(i) = buffer_type.getDimSize(1);
  }
  concat.lengths().replaceAllUsesWith(builder.create<TF::ConstOp>(
      concat.getLoc(),
      tensorflow::ConvertTensor(lengths_tensor, &builder).ValueOrDie()));
  concat.erase();
  return success();
}

LogicalResult HandleTensorArraySplitV3Op(
    TF::TensorArraySplitV3Op split,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = split.handle();
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
  auto buffer = builder
                    .create<TF::ReshapeOp>(
                        split.getLoc(),
                        ArrayRef<Type>{RankedTensorType::get(
                            buffer_shape, elem_type.getElementType())},
                        ArrayRef<Value>{split.value(),
                                        cutil::GetR1Const(buffer_shape, builder,
                                                          split.getLoc())},
                        ArrayRef<NamedAttribute>{})
                    .output();
  // Accumulate with the old buffer.
  auto old_buffer =
      cutil::ReadLocalVariable(local_var, builder, split.getLoc());
  buffer =
      cutil::AccumulateBuffers(old_buffer, buffer, builder, split.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, split.getLoc());
  split.flow_out().replaceAllUsesWith(split.flow_in());
  split.erase();
  return success();
}

LogicalResult HandleTensorArraySizeV3Op(
    TF::TensorArraySizeV3Op size,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = size.handle();
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
  size.size().replaceAllUsesWith(result);
  size.erase();
  return success();
}

LogicalResult HandleTensorArrayGradV3Op(
    TF::TensorArrayGradV3Op grad,
    llvm::SmallDenseMap<Value, TensorArrayStats>* stats) {
  auto local_var = grad.handle();
  OpBuilder builder(grad);
  Value grad_var;
  auto sit = stats->find(local_var);
  if (sit == stats->end()) return grad.emitOpError("unknown tensor array");
  auto emplace_res = sit->getSecond().grads.try_emplace(grad.source(), Value());
  if (!emplace_res.second) {
    // If the source has been assigned a grad, use it.
    grad_var = emplace_res.first->getSecond();
  } else {
    grad_var = builder.create<TF::MlirLocalVarOp>(
        grad.getLoc(), ArrayRef<Type>{local_var.getType()}, ArrayRef<Value>{},
        ArrayRef<NamedAttribute>{});
    Value buffer;
    auto buffer_type = getElementTypeOrSelf(local_var.getType())
                           .cast<TF::ResourceType>()
                           .getSubtypes()[0]
                           .cast<RankedTensorType>();
    if (failed(cutil::CreateInitBufferValue(
            buffer_type.getShape().drop_front(), buffer_type.getDimSize(0),
            grad, buffer_type.getElementType(), builder, &buffer))) {
      return failure();
    }
    cutil::WriteLocalVariable(grad_var, buffer, builder, grad.getLoc());
    emplace_res.first->getSecond() = grad_var;
    // Write to a grad accumulates with previous writes.
    (*stats)[grad_var].accumulate_on_write = true;
  }
  grad.flow_out().replaceAllUsesWith(grad.flow_in());
  grad.grad_handle().replaceAllUsesWith(grad_var);
  grad.erase();
  return success();
}

LogicalResult HandleTensorArrayGatherV3Op(
    TF::TensorArrayGatherV3Op gather,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = gather.handle();
  if (stats.count(local_var) == 0) {
    return gather.emitOpError("unknown tensor array");
  }
  OpBuilder builder(gather);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, gather.getLoc());
  auto result =
      cutil::GatherElements(gather.indices(), buffer, builder, gather.getLoc());
  gather.value().replaceAllUsesWith(result);
  gather.erase();
  return success();
}

LogicalResult HandleTensorArrayScatterV3Op(
    TF::TensorArrayScatterV3Op scatter,
    const llvm::SmallDenseMap<Value, TensorArrayStats>& stats) {
  auto local_var = scatter.handle();
  if (stats.count(local_var) == 0) {
    return scatter.emitOpError("unknown tensor array");
  }
  OpBuilder builder(scatter);
  auto buffer = cutil::ReadLocalVariable(local_var, builder, scatter.getLoc());
  buffer = cutil::ScatterAccumulateElements(scatter.indices(), scatter.value(),
                                            buffer, builder, scatter.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, scatter.getLoc());
  scatter.flow_out().replaceAllUsesWith(scatter.flow_in());
  scatter.erase();
  return success();
}

LogicalResult DecomposeTensorArrayOps(Block* block, ModuleOp module) {
  llvm::SmallDenseMap<Value, TensorArrayStats> stats;
  for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
    if (llvm::isa<TF::IdentityOp>(&op) || llvm::isa<TF::IdentityNOp>(&op)) {
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    } else if (auto ta = llvm::dyn_cast<TF::TensorArrayV3Op>(&op)) {
      if (failed(HandleTensorArrayV3Op(ta, module, &stats))) {
        return failure();
      }
    } else if (auto read = llvm::dyn_cast<TF::TensorArrayReadV3Op>(&op)) {
      if (failed(HandleTensorArrayReadV3Op(read, stats))) return failure();
    } else if (auto write = llvm::dyn_cast<TF::TensorArrayWriteV3Op>(&op)) {
      if (failed(HandleTensorArrayWriteV3Op(write, stats))) return failure();
    } else if (auto concat = llvm::dyn_cast<TF::TensorArrayConcatV3Op>(&op)) {
      if (failed(HandleTensorArrayConcatV3Op(concat, stats))) return failure();
    } else if (auto split = llvm::dyn_cast<TF::TensorArraySplitV3Op>(&op)) {
      if (failed(HandleTensorArraySplitV3Op(split, stats))) return failure();
    } else if (auto size = llvm::dyn_cast<TF::TensorArraySizeV3Op>(&op)) {
      if (failed(HandleTensorArraySizeV3Op(size, stats))) return failure();
    } else if (auto grad = llvm::dyn_cast<TF::TensorArrayGradV3Op>(&op)) {
      if (failed(HandleTensorArrayGradV3Op(grad, &stats))) return failure();
    } else if (auto gather = llvm::dyn_cast<TF::TensorArrayGatherV3Op>(&op)) {
      if (failed(HandleTensorArrayGatherV3Op(gather, stats))) return failure();
    } else if (auto scatter = llvm::dyn_cast<TF::TensorArrayScatterV3Op>(&op)) {
      if (failed(HandleTensorArrayScatterV3Op(scatter, stats))) {
        return failure();
      }
    } else if (auto close = llvm::dyn_cast<TF::TensorArrayCloseV3Op>(&op)) {
      close.erase();
    }
  }
  return success();
}

void TensorArrayOpsDecompositionPass::runOnModule() {
  auto module = getModule();
  auto main = module.lookupSymbol<FuncOp>("main");
  if (!main) return;
  if (failed(DecomposeTensorArrayOps(&main.front(), module))) {
    signalPassFailure();
  }
}

static PassRegistration<TensorArrayOpsDecompositionPass> pass(
    "tf-tensor-array-ops-decomposition",
    "Decompose tensor array operations into local variable operations.");

}  // namespace

namespace TF {
std::unique_ptr<OpPassBase<ModuleOp>> CreateTensorArrayOpsDecompositionPass() {
  return std::make_unique<TensorArrayOpsDecompositionPass>();
}

}  // namespace TF
}  // namespace mlir
