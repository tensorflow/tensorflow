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

#include <cstdint>
#include <memory>
#include <optional>
#include <type_traits>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {
namespace {

#define GEN_PASS_DEF_TPUPARTITIONEDOPCONVERSIONPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct TPUPartitionedOpConversionPass
    : public impl::TPUPartitionedOpConversionPassBase<
          TPUPartitionedOpConversionPass> {
  void runOnOperation() override;
};

template <typename T>
LogicalResult ReplacePartitionedOp(IntegerAttr num_cores_per_replica, T op) {
  constexpr bool is_input =
      std::is_same_v<std::decay_t<T>, TF::TPUPartitionedInputOp>;
  static_assert(
      is_input || std::is_same_v<std::decay_t<T>, TF::TPUPartitionedOutputOp>,
      "operator should either be an input or output");

  OpBuilder builder(op);
  int partition_dim = op.getPartitionDim();
  bool is_replicated = partition_dim == -1;
  if (!(is_replicated || num_cores_per_replica)) return failure();

  Type first_operand_type;
  if constexpr (is_input) {
    first_operand_type = op.getOperand(0).getType();
  } else {
    first_operand_type = op.getOperand().getType();
  }

  auto element_type = getElementTypeOrSelf(first_operand_type);
  if (mlir::isa<TF::ResourceType>(element_type)) {
    first_operand_type =
        mlir::cast<TF::ResourceType>(element_type).getSubtypes().front();
  }

  auto tensor_type = mlir::dyn_cast_or_null<TensorType>(first_operand_type);
  if (!(tensor_type && tensor_type.hasRank())) {
    return op->emitError()
           << "cannot convert op with unranked or non-tensor input type "
           << tensor_type << ".";
  }

  int rank = tensor_type.getRank();
  if (rank <= partition_dim) {
    return op->emitError() << "cannot partition " << first_operand_type
                           << " (rank = " << rank << ") along dimension "
                           << partition_dim << ".";
  }

  llvm::SmallVector<int64_t, 4> partition_dims(is_replicated ? 0 : rank, 1);
  if (!is_replicated) {
    partition_dims[partition_dim] = num_cores_per_replica.getInt();
  }

  if constexpr (is_input) {
    auto pi = builder.create<TF::TPUPartitionedInputV2Op>(
        op.getLoc(), op.getType(), op.getOperands(),
        builder.getI64ArrayAttr(partition_dims), builder.getBoolAttr(false),
        op.get_XlaShardingAttr());
    op->replaceAllUsesWith(pi);
  } else {
    auto po = builder.create<TF::TPUPartitionedOutputV2Op>(
        op.getLoc(), op.getResultTypes(), op.getOperand(),
        builder.getI64ArrayAttr(partition_dims), op.get_XlaShardingAttr());
    op->replaceAllUsesWith(po);
  }

  return success();
}

void TPUPartitionedOpConversionPass::runOnOperation() {
  llvm::SmallVector<TF::TPUReplicateMetadataOp, 4> metadata;
  getOperation()->walk(
      [&metadata](TF::TPUReplicateMetadataOp op) { metadata.push_back(op); });

  IntegerAttr num_cores_per_replica;
  if (metadata.size() == 1) {
    num_cores_per_replica = metadata.front().getNumCoresPerReplicaAttr();
  }

  auto result = getOperation()->walk([&num_cores_per_replica](Operation* op) {
    std::optional<LogicalResult> status;
    if (auto partitioned_input =
            llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(op)) {
      status = ReplacePartitionedOp(num_cores_per_replica, partitioned_input);
    } else if (auto partitioned_output =
                   llvm::dyn_cast_or_null<TF::TPUPartitionedOutputOp>(op)) {
      status = ReplacePartitionedOp(num_cores_per_replica, partitioned_output);
    }

    if (status.has_value()) {
      if (failed(*status) || !op->use_empty()) return WalkResult::interrupt();

      op->erase();
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTPUPartitionedOpConversionPass() {
  return std::make_unique<TPUPartitionedOpConversionPass>();
}

}  // namespace TFTPU
}  // namespace mlir
