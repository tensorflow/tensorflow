/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/fuse_mlrt_ops.h"

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

class FuseMlrtOpPass
    : public mlir::PassWrapper<FuseMlrtOpPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMlrtOpPass)

 private:
  llvm::StringRef getArgument() const final { return "tf-mlrt-fuse"; }

  llvm::StringRef getDescription() const final {
    return "Fuse consecutive mlrt ops of the same kind into one.";
  }

  void runOnOperation() override;
};

void FuseGetResourceOps(mlir::OpBuilder& builder, mlir::Block& block) {
  llvm::SmallVector<tf_mlrt::GetResourceOp> get_resource_ops;
  for (auto& op : llvm::make_early_inc_range(block)) {
    if (auto get_resource_op = llvm::dyn_cast<tf_mlrt::GetResourceOp>(&op)) {
      get_resource_ops.push_back(get_resource_op);
    }
  }

  if (get_resource_ops.empty()) return;

  // The last op is always a return op, so it is guaranteed to process all
  // groups of the candidate ops.
  auto first_get = get_resource_ops.front();

  builder.setInsertionPointAfter(first_get);

  llvm::SmallVector<mlir::Attribute> indices;
  llvm::SmallVector<mlir::Type> result_types;
  llvm::SmallVector<mlir::Value> old_values;

  indices.reserve(get_resource_ops.size());
  result_types.reserve(get_resource_ops.size());
  old_values.reserve(get_resource_ops.size());

  for (auto op : get_resource_ops) {
    auto indices_attr = op.getIndices();
    indices.append(indices_attr.begin(), indices_attr.end());
    result_types.append(op.result_type_begin(), op.result_type_end());
    old_values.append(op.result_begin(), op.result_end());
  }

  auto new_op = builder.create<tf_mlrt::GetResourceOp>(
      first_get.getLoc(), result_types, builder.getArrayAttr(indices));

  for (auto [old_value, new_value] :
       llvm::zip(old_values, new_op.getResults())) {
    old_value.replaceAllUsesWith(new_value);
  }

  for (auto get_resource_op : get_resource_ops) {
    get_resource_op->erase();
  }
}

template <typename AwaitOpType, typename AwaitAllOpType,
          typename ValueType = void>
void FuseAwaitOps(mlir::OpBuilder& builder, mlir::Block& block) {
  llvm::SmallVector<AwaitOpType> await_ops;
  for (auto& op : llvm::make_early_inc_range(block)) {
    if (auto await_op = llvm::dyn_cast<AwaitOpType>(&op)) {
      await_ops.push_back(await_op);
      continue;
    }

    // The last op is always a return op, so it is guaranteed to process all
    // groups of the candidate ops.
    if (await_ops.size() > 1) {
      auto last_await = await_ops.back();

      builder.setInsertionPointAfter(last_await);

      llvm::SmallVector<mlir::Value> futures;
      futures.reserve(await_ops.size());
      for (auto op : await_ops) {
        futures.push_back(op.getOperand());
      }

      llvm::SmallVector<mlir::Type> result_types;
      if constexpr (!std::is_same_v<ValueType, void>) {
        result_types.assign(futures.size(), builder.getType<ValueType>());
      }

      auto await_all =
          builder.create<AwaitAllOpType>(op.getLoc(), result_types, futures);

      if constexpr (!std::is_same_v<ValueType, void>) {
        for (auto [await_op, new_value] :
             llvm::zip(await_ops, await_all.getResults())) {
          await_op.getResult().replaceAllUsesWith(new_value);
        }
      }

      for (auto await_op : await_ops) {
        await_op->erase();
      }
    }

    await_ops.clear();
  }
}

void FusePromiseReturn(mlir::OpBuilder& builder, mlir::Block& block) {
  auto* terminator = block.getTerminator();
  auto return_op = llvm::dyn_cast<mlir::func::ReturnOp>(terminator);
  if (!return_op || return_op->getNumOperands() > 0) return;

  auto promise_op =
      llvm::dyn_cast_or_null<tf_mlrt::PromiseOp>(return_op->getPrevNode());
  if (!promise_op) return;

  builder.setInsertionPointAfter(return_op);
  builder.create<tf_mlrt::PromiseReturnOp>(return_op->getLoc(),
                                           promise_op->getResultTypes(),
                                           promise_op->getOperands());
  return_op->erase();
  promise_op->erase();
}

void FuseMlrtOpPass::runOnOperation() {
  auto func = getOperation();

  mlir::OpBuilder builder(func);

  FuseAwaitOps<tf_mlrt::AwaitOp, tf_mlrt::AwaitAllOp, tf_mlrt::TFTensorType>(
      builder, func.front());
  FuseAwaitOps<mlrt::compiler::AwaitHandleOp, mlrt::compiler::AwaitAllHandleOp>(
      builder, func.front());
  FuseAwaitOps<mlrt::compiler::AwaitControlOp,
               mlrt::compiler::AwaitAllControlOp>(builder, func.front());
  FuseGetResourceOps(builder, func.front());
  FusePromiseReturn(builder, func.front());
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateFuseMlrtOpPass() {
  return std::make_unique<FuseMlrtOpPass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
