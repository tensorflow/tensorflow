/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"

#include <memory>

#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/eval_util.h"

namespace mlir {
namespace quant {

bool HasQuantizedTensors(Operation* op) {
  if (IsOpNotQuantizable(op)) return false;
  for (Type operand_type : op->getOperandTypes()) {
    auto tensor_type = operand_type.dyn_cast<TensorType>();
    if (tensor_type && tensor_type.getElementType().isa<QuantizedType>()) {
      return true;
    }
  }
  for (Type result_type : op->getResultTypes()) {
    auto tensor_type = result_type.dyn_cast<TensorType>();
    if (tensor_type && tensor_type.getElementType().isa<QuantizedType>()) {
      return true;
    }
  }
  return false;
}

bool HasStaticShape(Value value) {
  auto shaped_type = value.getType().dyn_cast<ShapedType>();
  if (!shaped_type) return false;

  return shaped_type.hasStaticShape();
}

bool HasStaticShapeAtDims(Value value, llvm::ArrayRef<int> dims) {
  auto shaped_type = value.getType().dyn_cast<ShapedType>();
  if (!shaped_type) return false;

  for (auto dim : dims) {
    if (shaped_type.isDynamicDim(dim)) return false;
  }
  return true;
}

Type CloneTypeWithNewElementType(Type old_type, Type element_type) {
  if (!old_type.isa<ShapedType>()) return {};

  return old_type.cast<ShapedType>().clone(element_type);
}

// These constant folding utilities are forked from
// tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.cc.
// TODO(b/241488936): Remove these constant folding utility functions after
// adding a new constant folding pass to TensorFlow.
LogicalResult IsOperationFoldable(Operation* op) {
  if (!op->getDialect()->getNamespace().equals("tf") ||
      llvm::isa<TF::ConstOp>(op)) {
    return failure();
  }
  // Ops with `NoConstantFold` trait or side effects should not be constant
  // folded to preserve the original semantics.
  if (op->hasTrait<OpTrait::IsTerminator>() ||
      op->hasTrait<OpTrait::TF::NoConstantFold>() || op->getNumRegions() != 0 ||
      !MemoryEffectOpInterface::hasNoEffect(op)) {
    return failure();
  }

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (auto type : op->getResultTypes()) {
    if (auto tensor_type = type.dyn_cast<TensorType>()) {
      if (tensor_type.getElementType().isa<TF::VariantType>()) {
        return failure();
      }
    }
  }

  // Do not execute function calls.
  if (llvm::isa<TF::WhileOp, TF::CaseOp, TF::IfOp, CallOpInterface>(op)) {
    return failure();
  }

  // Check if the operands are constants or foldable as well.
  for (auto operand : op->getOperands()) {
    auto preceding_op = operand.getDefiningOp();
    if (!preceding_op || (!llvm::isa<TF::ConstOp>(preceding_op) &&
                          failed(IsOperationFoldable(preceding_op)))) {
      return failure();
    }
  }

  return success();
}

// Folds the operation recursively and return the results.
LogicalResult FoldOperation(TFE_Context* ctx, OpBuilder& builder, Operation* op,
                            llvm::SmallVector<Value>& results) {
  results.clear();
  builder.setInsertionPointAfter(op);

  bool has_empty_numerical_results =
      llvm::all_of(op->getResultTypes(), [](Type ty) {
        ShapedType shaped_ty = ty.cast<ShapedType>();
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results && op->isRegistered()) {
    for (Type ty : op->getResultTypes()) {
      auto shaped_ty = ty.cast<ShapedType>();
      results.push_back(builder.create<TF::ConstOp>(
          op->getLoc(),
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>())));
    }
    return success();
  }

  SmallVector<ElementsAttr, 4> inputs;
  for (auto operand : op->getOperands()) {
    auto preceding_const_op = operand.getDefiningOp<TF::ConstOp>();
    if (preceding_const_op) {
      inputs.push_back(preceding_const_op.value());
      continue;
    }

    Operation* preceding_op = operand.getDefiningOp();
    int preceding_result_id = -1;
    for (auto preceding_result : preceding_op->getResults()) {
      if (operand == preceding_result) {
        preceding_result_id = preceding_result.getResultNumber();
        break;
      }
    }
    llvm::SmallVector<Value> preceding_results;
    if (failed(FoldOperation(ctx, builder, preceding_op, preceding_results))) {
      return failure();
    }
    auto preceding_result = preceding_results[preceding_result_id];
    preceding_const_op = preceding_result.getDefiningOp<TF::ConstOp>();
    inputs.push_back(preceding_const_op.value());
  }

  // Avoid overlapping folds with the same context.
  static auto* mu = new tensorflow::mutex();
  tensorflow::mutex_lock l(*mu);
  SmallVector<Attribute, 8> constants;
  if (failed(tensorflow::EvaluateOperation(op, inputs, ctx, &constants))) {
    return failure();
  }
  for (const auto& constant : constants) {
    results.push_back(builder.create<TF::ConstOp>(op->getLoc(), constant));
  }
  return success();
}

TFE_Context* InitializeTFRuntime() {
  std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
      TF_NewStatus(), TF_DeleteStatus);
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  // Only initialize single CPU.
  tensorflow::ConfigProto config_proto;
  // This is conceptually equal to what we do in python/eager/context.py but
  // with all GPU devices ignored and CPU only set to 1.
  (*config_proto.mutable_device_count())["CPU"] = 1;
  (*config_proto.mutable_device_count())["GPU"] = 0;
  std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
      TF_NewBuffer(), TF_DeleteBuffer);
  DCHECK(config->data == nullptr);

  // Copy config_proto into config.
  {
    const size_t proto_size = config_proto.ByteSizeLong();
    void* buf = tensorflow::port::Malloc(proto_size);
    if (buf == nullptr) {
      LOG(ERROR) << "Failed to allocate memory to serialize ConfigProto "
                    "while creating context options for constant folding";
      return nullptr;
    }
    if (!config_proto.SerializeWithCachedSizesToArray(
            static_cast<uint8_t*>(buf))) {
      tensorflow::port::Free(buf);
      LOG(ERROR) << "Unable to serialize ConfigProto while creating context "
                    "options for constant folding";
      return nullptr;
    }
    config->data = buf;
    config->length = proto_size;
    config->data_deallocator = [](void* data, size_t length) {
      tensorflow::port::Free(data);
    };
  }

  TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                              status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to set context options for constant folding: "
               << status.get();
    return nullptr;
  }

  // Input tensors are placed on the host CPU so use the explicit device
  // policy to fail if no CPU kernels are available for the op.
  TFE_ContextOptionsSetDevicePlacementPolicy(opts.get(),
                                             TFE_DEVICE_PLACEMENT_EXPLICIT);
  auto ctx = TFE_NewContext(opts.get(), status.get());
  if (TF_GetCode(status.get()) != TF_OK) {
    LOG(ERROR) << "Failed to create context for constant folding: "
               << status.get();
    return nullptr;
  }
  return ctx;
}

llvm::SmallVector<Value> ConstantFoldOpIfPossible(Operation* op) {
  if (failed(IsOperationFoldable(op))) return op->getResults();

  static TFE_Context* ctx = InitializeTFRuntime();
  if (!ctx) return op->getResults();

  OpBuilder builder(op);
  llvm::SmallVector<Value> results;
  if (failed(FoldOperation(ctx, builder, op, results))) {
    return op->getResults();
  }
  return results;
}

}  // namespace quant
}  // namespace mlir
